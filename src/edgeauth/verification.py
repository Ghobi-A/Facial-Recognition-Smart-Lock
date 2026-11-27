"""Shared capture screening and 1:1 verification decisions."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vault import TemplateVault
from .vision import CaptureRejected, FacePipeline, PresentationDetector, cosine, unit_vector

DEFAULT_PAD_CUTOFF = 0.8
ENROLMENT_MIN_COSINE = 0.45
POLICY_SCHEMA = "edgeauth/verification-policy/v1"


@dataclass(frozen=True)
class Screen:
    matcher: FacePipeline
    pad: PresentationDetector
    pad_cutoff: float = DEFAULT_PAD_CUTOFF

    def __post_init__(self) -> None:
        if not math.isfinite(self.pad_cutoff) or not 0.5 <= self.pad_cutoff <= 1:
            raise ValueError("PAD cutoff must be between 0.5 and 1")

    def embed(self, image: np.ndarray) -> np.ndarray:
        sample = self.matcher.analyse(image)
        if self.pad.live_score(sample.face_crop) < self.pad_cutoff:
            raise CaptureRejected("presentation screening rejected the capture")
        return sample.embedding

    def policy(self, threshold: float, target_fmr: float) -> dict:
        return {
            "schema": POLICY_SCHEMA,
            "matcher": self.matcher.model_id,
            "pad": self.pad.model_id,
            "pad_cutoff": self.pad_cutoff,
            "threshold": threshold,
            "calibration_target_fmr": target_fmr,
        }

    def load_threshold(self, path: Path) -> float:
        policy = json.loads(path.read_text(encoding="utf-8"))
        expected = self.policy(0.0, 0.0)
        fields = ("schema", "matcher", "pad", "pad_cutoff")
        if not isinstance(policy, dict) or any(policy.get(key) != expected[key] for key in fields):
            raise ValueError("policy does not match the inference or PAD configuration")
        threshold = policy.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("policy does not contain a numeric threshold")
        return validate_threshold(threshold)


def validate_threshold(threshold: float) -> float:
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("cosine threshold must be in [-1, 1]")
    return threshold


def build_template(embeddings: list[np.ndarray]) -> np.ndarray:
    if len(embeddings) != 3:
        raise ValueError("enrolment requires three captures")
    vectors = [unit_vector(embedding) for embedding in embeddings]
    for index, vector in enumerate(vectors):
        if any(cosine(vector, other) < ENROLMENT_MIN_COSINE for other in vectors[:index]):
            raise CaptureRejected("enrolment captures are inconsistent")
    return unit_vector(np.mean(vectors, axis=0))


def verification_decision(
    vault: TemplateVault,
    screen: Screen,
    subject: str,
    image: np.ndarray,
    threshold: float,
) -> bool:
    validate_threshold(threshold)
    enrolled = vault.get(subject)
    if enrolled is None or enrolled[0] != screen.matcher.model_id:
        return False
    try:
        probe = screen.embed(image)
    except CaptureRejected:
        return False
    return cosine(enrolled[1], probe) >= threshold
