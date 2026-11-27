"""Identity-disjoint evaluation using local enrolment and probe manifests."""

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .metrics import Trial, calibrate, curve, curve_summary, operating_point
from .verification import Screen, build_template
from .vision import CaptureRejected, Condition, cosine, perturb, read_image

REPORT_SCHEMA = "edgeauth/verification-report/v1"
PAIR_COLUMNS = (
    "split",
    "reference_id",
    "probe_id",
    "reference_a",
    "reference_b",
    "reference_c",
    "probe",
)


@dataclass(frozen=True)
class Pair:
    split: str
    reference_id: str
    probe_id: str
    gallery: tuple[Path, Path, Path]
    probe: Path

    @property
    def genuine(self) -> bool:
        return self.reference_id == self.probe_id


def manifest_rows(path: Path, columns: tuple[str, ...]):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(columns):
            raise ValueError(f"expected CSV columns: {', '.join(columns)}")
        for line, row in enumerate(reader, start=2):
            if None in row or any(not value or not value.strip() for value in row.values()):
                raise ValueError(f"invalid CSV row {line}")
            yield {key: value.strip() for key, value in row.items()}


def file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def load_pairs(manifest: Path) -> list[Pair]:
    pairs = []
    identities: dict[str, str] = {}
    image_roles: dict[bytes, tuple[str, str]] = {}
    galleries: dict[str, frozenset[bytes]] = {}
    pair_keys = set()
    split_labels = defaultdict(set)
    digest = lru_cache(maxsize=None)(file_digest)
    for row in manifest_rows(manifest, PAIR_COLUMNS):
        split, reference, probe = (row["split"], row["reference_id"], row["probe_id"])
        if split not in ("calibration", "test"):
            raise ValueError("split must be calibration or test")
        for identity in (reference, probe):
            if identities.setdefault(identity, split) != split:
                raise ValueError("identity leakage between calibration and test")
        a, b, c, capture = ((manifest.parent / row[key]).resolve() for key in PAIR_COLUMNS[3:])
        gallery = frozenset(digest(path) for path in (a, b, c))
        if len(gallery) != 3:
            raise ValueError("reference gallery must contain three distinct files")
        if galleries.setdefault(reference, gallery) != gallery:
            raise ValueError("use one consistent reference gallery per claimed identity")
        capture_digest = digest(capture)
        roles = [(item, (reference, "reference")) for item in gallery]
        roles.append((capture_digest, (probe, "probe")))
        for content, role in roles:
            if image_roles.setdefault(content, role) != role:
                raise ValueError("image reused across identities or between enrolment and probes")
        key = reference, capture_digest
        if key in pair_keys:
            raise ValueError("duplicate verification trial")
        pair_keys.add(key)
        pair = Pair(split, reference, probe, tuple(sorted((a, b, c))), capture)
        split_labels[split].add(pair.genuine)
        pairs.append(pair)
    if any(split_labels[split] != {True, False} for split in ("calibration", "test")):
        raise ValueError("calibration and test each require genuine and impostor trials")
    return pairs


def summarise(trials: list[Trial], threshold: float | None) -> dict:
    points = curve(trials)
    return {
        **curve_summary(points),
        "curve": [{"fmr": point["fmr"], "fnmr": point["fnmr"]} for point in points],
        "attempts": len(trials),
        "screened_out": sum(trial.score is None for trial in trials),
        "operating_point": operating_point(trials, threshold) if threshold is not None else None,
    }


def evaluate(pairs: list[Pair], screen: Screen, target_fmr: float, *, robustness: bool) -> dict:
    @lru_cache(maxsize=512)
    def embed(path: Path, condition: Condition):
        try:
            return screen.embed(perturb(read_image(path), condition))
        except CaptureRejected:
            return None

    @lru_cache(maxsize=512)
    def gallery(paths: tuple[Path, Path, Path]):
        vectors = [embed(path, Condition.CLEAN) for path in paths]
        if any(vector is None for vector in vectors):
            return None
        try:
            return build_template(vectors)
        except CaptureRejected:
            return None

    def score_pairs(split: str, condition: Condition) -> list[Trial]:
        trials = []
        for pair in pairs:
            if pair.split != split:
                continue
            reference, probe = gallery(pair.gallery), embed(pair.probe, condition)
            score = (
                cosine(reference, probe) if reference is not None and probe is not None else None
            )
            trials.append(Trial(pair.genuine, score))
        return trials

    calibration = score_pairs("calibration", Condition.CLEAN)
    threshold = calibrate(calibration, target_fmr)
    conditions = list(Condition) if robustness else [Condition.CLEAN]
    report = {
        "schema": REPORT_SCHEMA,
        "source": "local consented manifest",
        "matcher": screen.matcher.model_id,
        "pad": screen.pad.model_id,
        "pad_cutoff": screen.pad_cutoff,
        "target_fmr": target_fmr,
        "calibration": summarise(calibration, threshold),
        "test": {
            condition.value: summarise(score_pairs("test", condition), threshold)
            for condition in conditions
        },
    }
    return report


def synthetic_report() -> dict:
    """Dashboard example from generated scores; no models or face data are used."""
    import numpy as np

    rng = np.random.default_rng(2022)

    def trials(mean: float) -> list[Trial]:
        genuine = np.clip(rng.normal(mean, 0.18, size=300), -1, 1)
        impostor = np.clip(rng.normal(0.1, 0.15, size=300), -1, 1)
        return [Trial(True, float(score)) for score in genuine] + [
            Trial(False, float(score)) for score in impostor
        ]

    target = 0.01
    calibration = trials(0.68)
    threshold = calibrate(calibration, target)
    return {
        "schema": REPORT_SCHEMA,
        "source": "synthetic example scores",
        "target_fmr": target,
        "calibration": summarise(calibration, threshold),
        "test": {
            name: summarise(trials(mean), threshold)
            for name, mean in (
                ("clean", 0.68),
                ("low-light", 0.55),
                ("blur", 0.52),
                ("low-resolution", 0.61),
            )
        },
    }


def evaluate_pad(manifest: Path, screen: Screen) -> dict:
    """Measure PAD on a separately labelled test manifest, by presentation species."""
    groups = defaultdict(
        lambda: {"presentations": 0, "quality_rejected": 0, "classified": 0, "classified_live": 0}
    )
    seen = set()
    for row in manifest_rows(manifest, ("capture", "label", "species")):
        label = row["label"]
        if label not in ("bona-fide", "attack"):
            raise ValueError("PAD labels must be bona-fide or attack")
        path = (manifest.parent / row["capture"]).resolve()
        fingerprint = file_digest(path)
        if fingerprint in seen:
            raise ValueError("duplicate PAD capture")
        seen.add(fingerprint)
        group = groups[label, row["species"]]
        group["presentations"] += 1
        try:
            image = read_image(path)
            sample = screen.matcher.analyse(image)
        except CaptureRejected:
            group["quality_rejected"] += 1
            continue
        group["classified"] += 1
        group["classified_live"] += screen.pad.live_score(sample.face_crop) >= screen.pad_cutoff
    if {label for label, _ in groups} != {"bona-fide", "attack"}:
        raise ValueError("PAD evaluation requires bona-fide and attack captures")
    results = []
    for (label, species), group in sorted(groups.items()):
        classified = group["classified"]
        live = group["classified_live"]
        attacks = label == "attack"
        errors = live if attacks else classified - live
        attempt_errors = errors if attacks else errors + group["quality_rejected"]
        results.append(
            {
                "label": label,
                "species": species,
                **group,
                "classifier_error_rate": errors / classified if classified else None,
                "pipeline_error_rate": attempt_errors / group["presentations"],
            }
        )
    return {
        "schema": "edgeauth/pad-report/v1",
        "pad": screen.pad.model_id,
        "pad_cutoff": screen.pad_cutoff,
        "results": results,
    }
