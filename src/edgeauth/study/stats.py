"""Attempt-level error rates, thresholds, cross-calibration and identity-aware bootstrap CIs.

Scores are similarities (higher = same person). NaN marks a failure to acquire; it can never
be accepted, so it counts against FRR for genuine attempts and as a correct rejection for
impostors. That is the attempt-level convention: what a person at the door experiences.

Uncertainty is estimated with a *person-level* (pigeonhole) bootstrap (Owen, 2007): people are
resampled with replacement and every pair is weighted by how often its people were drawn.
Pairs that share a person are dependent, so resampling pairs alone understates the interval.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# A false accept (stranger let in) is taken to cost ten times a false reject (owner retries).
COST_FALSE_ACCEPT = 10.0
COST_FALSE_REJECT = 1.0
FAR_TARGETS = (1e-2, 1e-3, 1e-4)
CONFIDENCE = (2.5, 97.5)


@dataclass(frozen=True)
class Scores:
    genuine: np.ndarray
    impostor: np.ndarray
    genuine_weights: np.ndarray | None = None
    impostor_weights: np.ndarray | None = None

    def subset(self, genuine_mask: np.ndarray, impostor_mask: np.ndarray) -> "Scores":
        return Scores(self.genuine[genuine_mask], self.impostor[impostor_mask])


def accepted(scores: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    return np.nan_to_num(scores, nan=-np.inf) >= threshold


def far(impostor: np.ndarray, threshold: float) -> float:
    return float(np.mean(accepted(impostor, threshold)))


def frr(genuine: np.ndarray, threshold: float) -> float:
    return float(np.mean(~accepted(genuine, threshold)))


def _accept_curve(values: np.ndarray, weights: np.ndarray | None, thresholds: np.ndarray):
    """Weighted fraction of ``values`` at or above each threshold (NaN never accepted)."""
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    total = weights.sum()
    finite = np.isfinite(values)
    order = np.argsort(values[finite])
    sorted_values = values[finite][order]
    # cumulative weight from the top: weight of values >= threshold
    tail = np.concatenate([np.cumsum(weights[finite][order][::-1])[::-1], [0.0]])
    return tail[np.searchsorted(sorted_values, thresholds, side="left")] / total


def sweep(scores: Scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every distinct threshold with its attempt-level FAR and FRR (thresholds descending)."""
    finite = np.concatenate(
        [scores.genuine[np.isfinite(scores.genuine)], scores.impostor[np.isfinite(scores.impostor)]]
    )
    thresholds = np.concatenate([[np.inf], np.unique(finite)[::-1]])
    far_curve = _accept_curve(scores.impostor, scores.impostor_weights, thresholds)
    frr_curve = 1 - _accept_curve(scores.genuine, scores.genuine_weights, thresholds)
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


def threshold_for_cost(
    scores: Scores, c_fa: float = COST_FALSE_ACCEPT, c_fr: float = COST_FALSE_REJECT
) -> float:
    """Threshold minimising c_fa * FAR + c_fr * FRR (equal attempt priors)."""
    thresholds, far_curve, frr_curve = sweep(scores)
    return float(thresholds[int(np.argmin(c_fa * far_curve + c_fr * frr_curve))])


def threshold_at_far(impostor: np.ndarray, target: float) -> float:
    """Lowest threshold whose FAR on ``impostor`` stays at or below ``target``.

    Intended for a dense impostor set (millions of scores), where FAR targets of 1e-3 or 1e-4
    are resolvable. Ties are handled by returning the next float above the k-th score.
    """
    values = np.sort(np.nan_to_num(impostor, nan=-np.inf))[::-1]
    allowed = int(np.floor(target * len(values)))
    if allowed >= len(values):
        return float(-np.inf)
    return float(np.nextafter(values[allowed], np.inf))


PairPolicy = Callable[[Scores], float]
POLICIES: dict[str, PairPolicy] = {
    "eer": lambda s: eer(s)[1],
    "cost_10to1": threshold_for_cost,
}


def cross_calibrated(
    scores: Scores,
    genuine_half: np.ndarray,
    impostor_half: np.ndarray,
    policy: str | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Two-fold cross-calibration over identity-disjoint halves.

    ``policy`` names a pair-based policy, or gives the two thresholds directly (index = the
    half they will be applied to, chosen elsewhere on the other half). Every decision is
    out-of-sample. Returns per-attempt accept decisions and the two thresholds used.
    """
    genuine_accept = np.zeros(len(scores.genuine), dtype=bool)
    impostor_accept = np.zeros(len(scores.impostor), dtype=bool)
    used = []
    for half in (0, 1):
        if isinstance(policy, str):
            threshold = POLICIES[policy](scores.subset(genuine_half != half, impostor_half != half))
        else:
            threshold = float(policy[half])
        used.append(threshold)
        g, i = genuine_half == half, impostor_half == half
        genuine_accept[g] = accepted(scores.genuine[g], threshold)
        impostor_accept[i] = accepted(scores.impostor[i], threshold)
    return genuine_accept, impostor_accept, used


# ---------------------------------------------------------------- bootstrap ---------------


def person_draws(n_people: int, rng: np.random.Generator, reps: int) -> np.ndarray:
    """reps x n_people multiplicities: how often each person is drawn in each resample."""
    return rng.multinomial(n_people, np.full(n_people, 1 / n_people), size=reps).astype(float)


def pair_weights(draws: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pigeonhole weight per pair: count[a] for a genuine pair, count[a]*count[b] otherwise."""
    weights = draws[:, a]
    stranger = a != b
    weights[:, stranger] *= draws[:, b[stranger]]
    return weights


def _interval(point: float, estimates: np.ndarray) -> tuple:
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) == 0:
        return (point, None, None)
    low, high = np.percentile(estimates, CONFIDENCE)
    return (point, float(low), float(high))


def person_bootstrap(
    outcomes: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    draws: np.ndarray,
) -> tuple:
    """Rate of 0/1 ``outcomes`` with a person-level 95% interval.

    ``a``/``b`` are integer person ids per pair (equal for genuine pairs) indexing the columns
    of ``draws`` from :func:`person_draws`. Resamples in which no pair survives are dropped.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    if len(outcomes) == 0:
        return (None, None, None)
    weights = pair_weights(draws, a, b)
    total = weights.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        estimates = (weights @ outcomes) / total
    return _interval(float(outcomes.mean()), estimates)


def pair_bootstrap(values: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple:
    """Naive pair-level interval, kept for comparison with the person-level one."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (None, None, None)
    means = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return _interval(float(values.mean()), means)


def person_curve_stats(
    scores: Scores,
    genuine_people: tuple[np.ndarray, np.ndarray],
    impostor_people: tuple[np.ndarray, np.ndarray],
    draws: np.ndarray,
) -> dict:
    """EER and AUC with person-level 95% intervals (weighted sweeps per resample)."""
    estimates = {"eer": [], "auc": []}
    genuine_w = pair_weights(draws, *genuine_people)
    impostor_w = pair_weights(draws, *impostor_people)
    for rep in range(len(draws)):
        sample = Scores(scores.genuine, scores.impostor, genuine_w[rep], impostor_w[rep])
        if sample.genuine_weights.sum() == 0 or sample.impostor_weights.sum() == 0:
            continue
        estimates["eer"].append(eer(sample)[0])
        estimates["auc"].append(auc(sample))
    point = {"eer": eer(scores)[0], "auc": auc(scores)}
    return {name: _interval(point[name], np.array(estimates[name])) for name in point}


def block_bootstrap(
    accepted_counts: np.ndarray, attempt_counts: np.ndarray, draws: np.ndarray
) -> tuple:
    """FAR over a dense person x person impostor block, with a person-level interval.

    ``accepted_counts[i, j]`` / ``attempt_counts[i, j]`` count image-pair accepts and attempts
    between people i and j (zero diagonal). For draw vector c the resampled rate is
    c^T A c / c^T N c, which equals weighting every image pair by count[i] * count[j].
    """
    total = attempt_counts.sum()
    if total == 0:
        return (None, None, None)
    point = float(accepted_counts.sum() / total)
    numerator = np.einsum("ri,ij,rj->r", draws, accepted_counts, draws, optimize=True)
    denominator = np.einsum("ri,ij,rj->r", draws, attempt_counts, draws, optimize=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _interval(point, numerator / denominator)
