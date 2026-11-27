"""SQLite-backed encrypted templates for a local operator."""

import hashlib
import hmac
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .vision import unit_vector


def create_key(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
        out.write(AESGCM.generate_key(bit_length=256))


def subject_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("use a 1–64 character pseudonymous ID (letters, digits, ._-)")
    return value


class TemplateVault:
    def __init__(self, database: Path, key_file: Path, *, initialise: bool = False):
        key = key_file.read_bytes()
        if len(key) != 32:
            raise ValueError("vault key must contain exactly 32 bytes")
        self.marker = hmac.digest(key, b"edgeauth-v1/vault-key-check", "sha256")
        self.index_key = hmac.digest(key, b"edgeauth-v1/index-key", "sha256")
        self.cipher = AESGCM(hmac.digest(key, b"edgeauth-v1/template-key", "sha256"))
        self.database = database
        if initialise:
            database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(database, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(descriptor)
        if not database.is_file():
            raise ValueError("vault does not exist; run edgeauth init")
        with closing(self._connect()) as connection, connection:
            if initialise:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, "
                    "key_marker BLOB NOT NULL)"
                )
                connection.execute("INSERT OR IGNORE INTO metadata VALUES (1, ?)", (self.marker,))
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS templates ("
                    "tag BLOB PRIMARY KEY, nonce BLOB NOT NULL, data BLOB NOT NULL)"
                )
            marker = connection.execute("SELECT key_marker FROM metadata WHERE id = 1").fetchone()
            if marker is None or not hmac.compare_digest(marker[0], self.marker):
                raise ValueError("vault key does not match the database")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{self.database.resolve().as_uri()}?mode=rw", uri=True)
        connection.execute("PRAGMA secure_delete=ON")
        return connection

    def _tag(self, subject: str) -> bytes:
        return hmac.digest(self.index_key, subject_id(subject).encode(), hashlib.sha256)

    def put(self, subject: str, model_id: str, embedding: np.ndarray, *, replace: bool) -> None:
        tag = self._tag(subject)
        vector = unit_vector(embedding)
        payload = json.dumps({"model": model_id, "embedding": vector.tolist()}).encode()
        nonce = os.urandom(12)
        data = self.cipher.encrypt(nonce, payload, b"edgeauth-v1/template/" + tag)
        query = "INSERT INTO templates VALUES (?, ?, ?)"
        if replace:
            query += " ON CONFLICT(tag) DO UPDATE SET nonce = excluded.nonce, data = excluded.data"
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(query, (tag, nonce, data))
        except sqlite3.IntegrityError as error:
            raise ValueError("subject already enrolled; pass --replace to update") from error

    def get(self, subject: str) -> tuple[str, np.ndarray] | None:
        tag = self._tag(subject)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT nonce, data FROM templates WHERE tag = ?", (tag,)
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(self.cipher.decrypt(*row, b"edgeauth-v1/template/" + tag))
        except InvalidTag as error:
            raise ValueError("template integrity check failed") from error
        return payload["model"], unit_vector(payload["embedding"])

    def delete(self, subject: str) -> bool:
        with closing(self._connect()) as connection, connection:
            return (
                connection.execute(
                    "DELETE FROM templates WHERE tag = ?", (self._tag(subject),)
                ).rowcount
                == 1
            )
