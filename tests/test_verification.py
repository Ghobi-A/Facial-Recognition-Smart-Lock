import sqlite3
from dataclasses import dataclass

import numpy as np
import pytest

from edgeauth.vault import TemplateVault, create_key
from edgeauth.verification import Screen, build_template, verification_decision
from edgeauth.vision import CaptureRejected, FaceSample


@dataclass
class Matcher:
    model_id: str = "recognizer-a"

    def analyse(self, image):
        if not np.any(image):
            raise CaptureRejected("no detected face")
        return FaceSample(image, image)


@dataclass
class PAD:
    score: float
    model_id: str = "pad-a"

    def live_score(self, _):
        return self.score


@pytest.fixture
def vault(tmp_path):
    key = tmp_path / "vault.key"
    create_key(key)
    return TemplateVault(tmp_path / "templates.sqlite3", key, initialise=True)


def test_verifier_checks_claim_quality_comparison_and_pad(vault):
    genuine = np.array([1.0, 0.0])
    impostor = np.array([0.0, 1.0])
    vault.put("subject-a", "recognizer-a", genuine, replace=False)
    screen = Screen(Matcher(), PAD(0.95))
    assert verification_decision(vault, screen, "subject-a", genuine, 0.75)
    assert not verification_decision(vault, screen, "missing", genuine, 0.75)
    assert not verification_decision(vault, screen, "subject-a", impostor, 0.75)
    assert not verification_decision(vault, screen, "subject-a", np.zeros(2), 0.75)
    assert not verification_decision(vault, Screen(Matcher(), PAD(0.5)), "subject-a", genuine, 0.75)
    assert not verification_decision(
        vault,
        Screen(Matcher("recognizer-b"), PAD(0.95)),
        "subject-a",
        genuine,
        0.75,
    )


def test_enrolment_update_and_delete(vault):
    reference = build_template([np.array([1.0, offset]) for offset in (0, -0.1, 0.1)])
    vault.put("subject-a", "recognizer-a", reference, replace=False)
    with pytest.raises(ValueError, match="already enrolled"):
        vault.put("subject-a", "recognizer-b", reference, replace=False)
    assert vault.get("subject-a")[0] == "recognizer-a"
    vault.put("subject-a", "recognizer-b", reference, replace=True)
    assert np.allclose(vault.get("subject-a")[1], reference)
    assert vault.delete("subject-a")
    assert not vault.delete("subject-a")
    assert vault.get("subject-a") is None


def test_vault_rejects_wrong_key_and_swapped_ciphertexts(vault, tmp_path):
    key = tmp_path / "other.key"
    create_key(key)
    with pytest.raises(ValueError, match="key does not match"):
        TemplateVault(vault.database, key, initialise=True)
    vault.put("subject-a", "recognizer-a", np.array([1.0, 0]), replace=False)
    vault.put("subject-b", "recognizer-a", np.array([0, 1.0]), replace=False)
    with sqlite3.connect(vault.database) as connection:
        first, second = connection.execute("SELECT tag, nonce, data FROM templates").fetchall()
        connection.execute(
            "UPDATE templates SET nonce = ?, data = ? WHERE tag = ?", (*second[1:], first[0])
        )
    with pytest.raises(ValueError, match="integrity"):
        vault.get("subject-a" if first[0] == vault._tag("subject-a") else "subject-b")
    assert b"subject-a" not in vault.database.read_bytes()


def test_enrolment_rejects_inconsistent_template():
    with pytest.raises(CaptureRejected, match="inconsistent"):
        build_template([np.array([1.0, 0]), np.array([1.0, 0.1]), np.array([0, 1.0])])


def test_policy_cannot_cross_pad_or_matcher_configuration(tmp_path):
    import json

    screen = Screen(Matcher(), PAD(0.95))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(screen.policy(0.65, 0.01)), encoding="utf-8")
    assert screen.load_threshold(policy) == 0.65
    with pytest.raises(ValueError, match="does not match"):
        Screen(Matcher(), PAD(0.95), 0.9).load_threshold(policy)
    with pytest.raises(ValueError, match="does not match"):
        Screen(Matcher("recognizer-b"), PAD(0.95)).load_threshold(policy)
