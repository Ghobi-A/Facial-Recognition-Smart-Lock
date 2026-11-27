"""Evaluation datasets in one on-disk format, so every stage of the study runs unchanged on each.

A prepared dataset directory holds:

* ``manifest.csv`` - verification pairs (``lfw.MANIFEST_COLUMNS``), folds 1-5 / 6-10 forming two
  identity-disjoint halves;
* ``people.csv`` - per-person demographic labels (``sex``, ``ethnicity``; blank if unknown);
* ``images.csv`` - per-image capture conditions (``yes``/``no`` columns);
* ``dataset.json`` - name, description, image root and label provenance.
"""

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import lfw

DEMOGRAPHIC_KEYS = ("sex", "ethnicity")


@dataclass
class Dataset:
    name: str
    directory: Path
    meta: dict
    rows: list[lfw.PairRow]
    people: dict[str, dict[str, str]] = field(default_factory=dict)
    images: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def image_root(self) -> Path:
        return (self.directory / self.meta["image_root"]).resolve()

    @property
    def manifest(self) -> Path:
        return self.directory / "manifest.csv"


def half_of(fold: int) -> int:
    return 0 if fold <= 5 else 1


def fold_for(person: str, salt: str) -> int:
    """Deterministic fold 1-10 per person, for datasets without identity-disjoint folds."""
    return int(hashlib.sha256(f"{salt}:{person}".encode()).hexdigest(), 16) % 10 + 1


def check_disjoint(rows: list[lfw.PairRow]) -> None:
    halves: dict[str, int] = {}
    for row in rows:
        for person in (row.person_a, row.person_b):
            if halves.setdefault(person, half_of(row.fold)) != half_of(row.fold):
                raise ValueError(f"{person} appears in both halves: identity leakage")
        if row.genuine != (row.person_a == row.person_b):
            raise ValueError(f"pair {row.pair_id} genuine flag does not match its identities")


def write(
    directory: Path,
    meta: dict,
    rows: list[lfw.PairRow],
    people: dict[str, dict[str, str]],
    images: dict[str, dict[str, str]],
) -> None:
    check_disjoint(rows)
    directory.mkdir(parents=True, exist_ok=True)
    lfw.write_manifest(rows, directory / "manifest.csv")
    with (directory / "people.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("person", *DEMOGRAPHIC_KEYS))
        for person in sorted(people):
            writer.writerow((person, *(people[person].get(k, "") for k in DEMOGRAPHIC_KEYS)))
    keys = sorted({key for labels in images.values() for key in labels})
    with (directory / "images.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("image", *keys))
        for image in sorted(images):
            writer.writerow((image, *(images[image].get(k, "") for k in keys)))
    meta = {**meta, "pairs": len(rows), "genuine": sum(r.genuine for r in rows)}
    (directory / "dataset.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load(directory: Path) -> Dataset:
    meta = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    rows = lfw.read_manifest(directory / "manifest.csv")
    check_disjoint(rows)

    def table(path: Path) -> dict[str, dict[str, str]]:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            key = reader.fieldnames[0]
            return {r[key]: {k: v for k, v in r.items() if k != key and v} for r in reader}

    return Dataset(
        meta["name"],
        directory,
        meta,
        rows,
        table(directory / "people.csv"),
        table(directory / "images.csv"),
    )


def balanced_pairs(
    images_by_person: dict[str, list[str]], salt: str, per_person: int, seed: int
) -> list[lfw.PairRow]:
    """Genuine and impostor pairs for a dataset without a pair protocol.

    People are hashed to folds; each person with two or more images contributes up to
    ``per_person`` genuine pairs, and an equal number of impostor pairs is drawn between
    distinct people in the *same* half, so halves stay identity-disjoint.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    people = sorted(images_by_person)
    folds = {p: fold_for(p, salt) for p in people}
    rows: list[lfw.PairRow] = []
    for half in (0, 1):
        members = [p for p in people if half_of(folds[p]) == half]
        genuine = []
        for person in members:
            pics = sorted(images_by_person[person])
            if len(pics) < 2:
                continue
            seen = set()
            for _ in range(per_person * 4):
                if len(seen) == min(per_person, len(pics) * (len(pics) - 1) // 2):
                    break
                a, b = sorted(rng.choice(len(pics), size=2, replace=False))
                seen.add((pics[a], pics[b]))
            genuine += [(person, a, person, b) for a, b in sorted(seen)]
        impostor = set()
        while len(impostor) < len(genuine):
            a, b = rng.choice(len(members), size=2, replace=False)
            pa, pb = sorted((members[a], members[b]))
            impostor.add(
                (
                    pa,
                    str(rng.choice(images_by_person[pa])),
                    pb,
                    str(rng.choice(images_by_person[pb])),
                )
            )
        for person_a, image_a, person_b, image_b in genuine + sorted(impostor):
            rows.append(
                lfw.PairRow(
                    len(rows),
                    folds[person_a],
                    person_a == person_b,
                    person_a,
                    image_a,
                    person_b,
                    image_b,
                )
            )
    return rows
