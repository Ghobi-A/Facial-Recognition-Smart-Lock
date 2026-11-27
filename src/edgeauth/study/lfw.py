"""LFW download, the official 10-fold pairs protocol, and the Kumar et al. attribute labels.

LFW's ten folds are identity-disjoint, so calibrating a threshold on folds 1-5 and testing
on folds 6-10 (and the reverse) never scores a person whose pairs set the threshold.
"""

import csv
import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

# Mirrors used by scikit-learn's fetch_lfw_pairs; the original UMass host is often down.
SOURCES = {
    "lfw.tgz": "https://ndownloader.figshare.com/files/5976018",
    "pairs.txt": "https://ndownloader.figshare.com/files/5976006",
    "lfw_attributes.txt": (
        "https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt"
    ),
}
MANIFEST_COLUMNS = ("pair_id", "fold", "genuine", "person_a", "image_a", "person_b", "image_b")

# Attributes are classifier scores (positive = present), not self-reported labels.
ETHNICITY = ("Asian", "White", "Black", "Indian")
AGE = ("Child", "Youth", "Middle Aged", "Senior")
BINARY = {"eyeglasses": "Eyeglasses", "harsh_lighting": "Harsh Lighting", "blurry": "Blurry"}


@dataclass(frozen=True)
class PairRow:
    pair_id: int
    fold: int
    genuine: bool
    person_a: str
    image_a: str
    person_b: str
    image_b: str


def image_path(person: str, number: int) -> str:
    return f"{person}/{person}_{number:04d}.jpg"


def fetch(directory: Path) -> Path:
    """Download and extract LFW into ``directory`` (skips files already present)."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        target = directory / filename
        if target.exists():
            continue
        partial = target.with_suffix(target.suffix + ".part")
        request = Request(url, headers={"User-Agent": "EdgeAuth-Study/0.2"})
        with urlopen(request, timeout=120) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(target)
    if not (directory / "lfw").is_dir():
        with tarfile.open(directory / "lfw.tgz") as archive:
            archive.extractall(directory, filter="data")
    return directory


def parse_pairs(path: Path) -> list[PairRow]:
    lines = path.read_text(encoding="utf-8").split("\n")
    folds, per_fold = (int(value) for value in lines[0].split())
    rows, index = [], 1
    for fold in range(1, folds + 1):
        for genuine in (True, False):
            for _ in range(per_fold):
                fields = lines[index].split("\t")
                index += 1
                if genuine:
                    person, first, second = fields
                    a, b = (person, int(first)), (person, int(second))
                else:
                    a, b = (fields[0], int(fields[1])), (fields[2], int(fields[3]))
                rows.append(
                    PairRow(len(rows), fold, genuine, a[0], image_path(*a), b[0], image_path(*b))
                )
    return rows


def write_manifest(rows: list[PairRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow(
                (
                    row.pair_id,
                    row.fold,
                    int(row.genuine),
                    row.person_a,
                    row.image_a,
                    row.person_b,
                    row.image_b,
                )
            )


def read_manifest(path: Path) -> list[PairRow]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"expected columns: {', '.join(MANIFEST_COLUMNS)}")
        return [
            PairRow(
                int(r["pair_id"]),
                int(r["fold"]),
                r["genuine"] == "1",
                r["person_a"],
                r["image_a"],
                r["person_b"],
                r["image_b"],
            )
            for r in reader
        ]


def manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_attributes(path: Path) -> dict[str, dict[str, float]]:
    """Map ``Person_Name/Person_Name_0001.jpg`` to its attribute scores."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[1].lstrip("#").strip().split("\t")
    table = {}
    for line in lines[2:]:
        fields = line.split("\t")
        if len(fields) != len(header):
            continue
        person = fields[0].replace(" ", "_")
        scores = {name: float(value) for name, value in zip(header[2:], fields[2:])}
        table[image_path(person, int(fields[1]))] = scores
    return table


def image_groups(scores: dict[str, float]) -> dict[str, str]:
    """Per-image subgroup labels derived from attribute scores (estimated, not ground truth)."""
    return {
        "sex": "male" if scores["Male"] > 0 else "female",
        "ethnicity": max(ETHNICITY, key=scores.__getitem__).lower(),
        "age": max(AGE, key=scores.__getitem__).lower(),
        **{key: "yes" if scores[name] > 0 else "no" for key, name in BINARY.items()},
    }


def person_groups(attributes: dict[str, dict[str, float]]) -> dict[str, dict[str, str]]:
    """Person-level sex and ethnicity from mean image scores, so a person has one label."""
    totals: dict[str, dict[str, list[float]]] = {}
    for image, scores in attributes.items():
        person = image.split("/")[0]
        bucket = totals.setdefault(person, {})
        for name in ("Male", *ETHNICITY):
            bucket.setdefault(name, []).append(scores[name])
    labels = {}
    for person, bucket in totals.items():
        mean = {name: sum(values) / len(values) for name, values in bucket.items()}
        labels[person] = {
            "sex": "male" if mean["Male"] > 0 else "female",
            "ethnicity": max(ETHNICITY, key=mean.__getitem__).lower(),
        }
    return labels
