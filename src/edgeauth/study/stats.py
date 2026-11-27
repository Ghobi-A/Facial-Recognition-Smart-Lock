"""Attempt-level error rates, cost-based thresholds, cross-calibration and bootstrap CIs.

Scores are similarities (higher = same person). NaN marks a failure to acquire; it can never
be accepted, so it counts against FRR for genuine attempts and as a correct rejection for
impostors. That is the attempt-level convention: what a person at the door experiences.
"""

from dataclasses import dataclass

import numpy as np

# A false accept (stranger let in) is taken to cost ten times a false reject (owner retries).
COST_FALSE_ACCEPT = 10.0
COST_FALSE_REJECT = 1.0
TARGET_FAR = 0.01


@dataclass(frozen=True)
class Scores:
    genuine: np.ndarray
    impostor: np.ndarray

    def subset(self, genuine_mask: np.ndarray, impostor_mask: np.ndarray) -> "Scores":
        return Scores(self.genuine[genuine_mask], self.impostor[impostor_mask])


def far(impostor: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.nan_to_num(impostor, nan=-np.inf) >= threshold))


def frr(genuine: np.ndarray, threshold: float) -> float:
    return float(np.mean(~(np.nan_to_num(genuine, nan=-np.inf) >= threshold)))


def sweep(scores: Scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every distinct threshold with its attempt-level FAR and FRR (thresholds descending)."""
    genuine = np.sort(scores.genuine[np.isfinite(scores.genuine)])
    impostor = np.sort(scores.impostor[np.isfinite(scores.impostor)])
    thresholds = np.unique(np.concatenate([genuine, impostor]))[::-1]
    thresholds = np.concatenate([[np.inf], thresholds])
    accepted_impostor = len(impostor) - np.searchsorted(impostor, thresholds, side="left")
    accepted_genuine = len(genuine) - np.searchsorted(genuine, thresholds, side="left")
    far_curve = accepted_impostor / len(scores.impostor)
    frr_curve = 1 - accepted_genuine / len(scores.genuine)
    return thresholds, far_curve, frr_curve


def eer(scores: Scores) -> tuple[float, float]:
    """Equal error rate and the threshold where FAR and FRR cross."""
    thresholds, far_curve, frr_curve = sweep(scores)
    index = int(np.argmin(np.abs(far_curve - frr_curve)))
    return float((far_curve[index] + frr_curve[index]) / 2), float(thresholds[index])


def auc(scores: Scores) -> float:
    _, far_curve, frr_curve = sweep(scores)
    tar = 1 - frr_curve
    return float(np.sum(np.diff(far_curve) * (tar[1:] + tar[:-1]) / 2))


def threshold_for_far(scores: Scores, target: float = TARGET_FAR) -> float:
    """Most permissive threshold whose FAR stays at or below ``target``."""
    thresholds, far_curve, _ = sweep(scores)
    return float(thresholds[far_curve <= target][-1])


def threshold_for_cost(
    scores: Scores, c_fa: float = COST_FALSE_ACCEPT, c_fr: float = COST_FALSE_REJECT
) -> float:
    """Threshold minimising c_fa * FAR + c_fr * FRR (equal attempt priors)."""
    thresholds, far_curve, frr_curve = sweep(scores)
    cost = c_fa * far_curve + c_fr * frr_curve
    return float(thresholds[int(np.argmin(cost))])


POLICIES = {
    "eer": lambda s: eer(s)[1],
    "far_1pct": threshold_for_far,
    "cost_10to1": threshold_for_cost,
}


def cross_calibrated(
    scores: Scores, genuine_half: np.ndarray, impostor_half: np.ndarray, policy: str
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Two-fold cross-calibration over identity-disjoint halves.

    Each half is scored with a threshold chosen on the other half, so every decision is
    out-of-sample. Returns per-attempt accept decisions and the two thresholds used.
    """
    choose = POLICIES[policy]
    genuine_accept = np.zeros(len(scores.genuine), dtype=bool)
    impostor_accept = np.zeros(len(scores.impostor), dtype=bool)
    used = []
    for half in (0, 1):
        calibration = scores.subset(genuine_half != half, impostor_half != half)
        threshold = choose(calibration)
        used.append(threshold)
        g, i = genuine_half == half, impostor_half == half
        genuine_accept[g] = np.nan_to_num(scores.genuine[g], nan=-np.inf) >= threshold
        impostor_accept[i] = np.nan_to_num(scores.impostor[i], nan=-np.inf) >= threshold
    return genuine_accept, impostor_accept, used


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple:
    """Point estimate and 95% percentile interval for a mean of 0/1 outcomes."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (None, None, None)
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(values.mean()), float(low), float(high))


def bootstrap_curve_stats(scores: Scores, rng: np.random.Generator, reps: int = 500) -> dict:
    """EER and AUC with 95% intervals from stratified resampling of attempts."""
    estimates = {"eer": [], "auc": []}
    for _ in range(reps):
        sample = Scores(
            scores.genuine[rng.integers(0, len(scores.genuine), len(scores.genuine))],
            scores.impostor[rng.integers(0, len(scores.impostor), len(scores.impostor))],
        )
        estimates["eer"].append(eer(sample)[0])
        estimates["auc"].append(auc(sample))
    point = {"eer": eer(scores)[0], "auc": auc(scores)}
    return {
        name: (point[name], *(float(v) for v in np.percentile(values, [2.5, 97.5])))
        for name, values in estimates.items()
    }
