"""Matcher and whole-attempt metrics for fixed 1:1 operating points."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trial:
    genuine: bool
    score: float | None


def operating_point(trials: list[Trial], threshold: float) -> dict:
    genuine = [trial for trial in trials if trial.genuine]
    impostor = [trial for trial in trials if not trial.genuine]
    scored_genuine = [trial.score for trial in genuine if trial.score is not None]
    scored_impostor = [trial.score for trial in impostor if trial.score is not None]
    false_matches = sum(score >= threshold for score in scored_impostor)
    genuine_matches = sum(score >= threshold for score in scored_genuine)
    return {
        "threshold": threshold,
        "genuine_attempts": len(genuine),
        "impostor_attempts": len(impostor),
        "genuine_scored": len(scored_genuine),
        "impostor_scored": len(scored_impostor),
        "false_matches": false_matches,
        "genuine_matches": genuine_matches,
        "fmr": false_matches / len(scored_impostor) if scored_impostor else None,
        "fnmr": 1 - genuine_matches / len(scored_genuine) if scored_genuine else None,
        "far": false_matches / len(impostor) if impostor else None,
        "frr": 1 - genuine_matches / len(genuine) if genuine else None,
    }


def curve(trials: list[Trial]) -> list[dict]:
    scored = sorted(
        ((trial.score, trial.genuine) for trial in trials if trial.score is not None),
        reverse=True,
    )
    genuine = sum(is_genuine for _, is_genuine in scored)
    impostor = len(scored) - genuine
    if not genuine or not impostor:
        return []
    thresholds = [{"threshold": None, "fmr": 0.0, "fnmr": 1.0}]
    genuine_accepted = impostor_accepted = 0
    for index, (score, is_genuine) in enumerate(scored):
        genuine_accepted += is_genuine
        impostor_accepted += not is_genuine
        if index + 1 == len(scored) or scored[index + 1][0] != score:
            thresholds.append(
                {
                    "threshold": score,
                    "fmr": impostor_accepted / impostor,
                    "fnmr": 1 - genuine_accepted / genuine,
                }
            )
    return thresholds


def curve_summary(points: list[dict]) -> dict:
    if not points:
        return {"roc_auc": None, "eer": None}
    fmr = np.array([point["fmr"] for point in points])
    fnmr = np.array([point["fnmr"] for point in points])
    tmr = 1 - fnmr
    auc = np.sum(np.diff(fmr) * (tmr[1:] + tmr[:-1]) / 2)
    return {"roc_auc": float(auc), "eer": float(np.interp(0.0, fmr - fnmr, fmr))}


def calibrate(trials: list[Trial], target_fmr: float) -> float | None:
    if not 0 <= target_fmr < 1:
        raise ValueError("target FMR must be in [0, 1)")
    candidates = [
        point["threshold"]
        for point in curve(trials)
        if point["threshold"] is not None and point["fmr"] <= target_fmr and point["fnmr"] < 1
    ]
    return min(candidates) if candidates else None
