"""Dense impostor sets: every cross-person image pair within each identity-disjoint half.

LFW's 3,000 official impostor pairs cannot resolve a FAR of 1e-3 (three errors) or 1e-4. Each
half of LFW has ~3,850 images, so it holds ~7 million stranger comparisons. Thresholds are set
on one half's dense set and tested on the other's.

For the person-level bootstrap, accepted image pairs are aggregated to a sparse
person x person count matrix A, and attempts have the closed form N = n n^T - diag(n^2), where
n counts images per person. A resample with person multiplicities c then has
FAR = c^T A c / c^T N c - the pigeonhole bootstrap without materialising 7M weights.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .stats import CONFIDENCE, threshold_at_far

CHUNK = 1024


@dataclass
class Block:
    """One half: embeddings, the local person index of each image, and person global ids."""

    vectors: np.ndarray  # images x dims (NaN rows = failed to acquire)
    person: np.ndarray  # local person index per image
    people: np.ndarray  # global person id per local person
    scoring: str  # "cosine" or "negative euclidean"

    @property
    def counts(self) -> np.ndarray:
        return np.bincount(self.person, minlength=len(self.people)).astype(float)

    def rows(self):
        """(start, similarity rows) chunks for upper-triangle stranger pairs (cached)."""
        if getattr(self, "_chunks", None) is None:
            self._chunks = list(self._compute())
        return self._chunks

    def _compute(self):
        valid = np.all(np.isfinite(self.vectors), axis=1)
        vectors = np.where(valid[:, None], self.vectors, 0).astype(np.float32)
        squared = np.sum(vectors**2, axis=1)
        for start in range(0, len(vectors), CHUNK):
            stop = min(start + CHUNK, len(vectors))
            dot = vectors[start:stop] @ vectors.T
            if self.scoring == "cosine":
                scores = dot
            else:
                scores = -np.sqrt(np.maximum(squared[start:stop, None] + squared - 2 * dot, 0))
            scores[~valid[start:stop]] = -np.inf
            scores[:, ~valid] = -np.inf
            columns = np.arange(len(vectors))
            keep = (columns[None, :] > np.arange(start, stop)[:, None]) & (
                self.person[start:stop, None] != self.person[None, :]
            )
            scores[~keep] = np.nan  # not a stranger pair (or below the diagonal)
            yield start, scores

    def impostor_scores(self) -> np.ndarray:
        parts = [scores[np.isfinite(scores) | np.isneginf(scores)] for _, scores in self.rows()]
        return np.concatenate(parts)


def thresholds_at_far(blocks: list[Block], targets) -> dict[float, list[float]]:
    """Per target FAR: the threshold for half h, chosen on the *other* half's dense set."""
    scores = [block.impostor_scores() for block in blocks]
    return {target: [threshold_at_far(scores[1 - h], target) for h in (0, 1)] for target in targets}


def accepted_matrix(block: Block, threshold: float, image_mask: np.ndarray | None = None):
    """Symmetric sparse person x person counts of accepted stranger image pairs.

    With ``image_mask``, only pairs where at least one image is flagged are counted.
    """
    rows_, cols_ = [], []
    for start, scores in block.rows():
        hits = np.nan_to_num(scores, nan=-np.inf) >= threshold
        i, j = np.nonzero(hits)
        i += start
        if image_mask is not None:
            flagged = image_mask[i] | image_mask[j]
            i, j = i[flagged], j[flagged]
        rows_.append(block.person[i])
        cols_.append(block.person[j])
    r, c = np.concatenate(rows_), np.concatenate(cols_)
    size = len(block.people)
    upper = sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(size, size)).tocsr()
    return upper + upper.T


def _attempt_terms(draws: np.ndarray, n: np.ndarray, n_clear: np.ndarray | None):
    """c^T N c for N = n n^T - m m^T - diag(n^2 - m^2), m = unflagged image counts."""
    m = np.zeros_like(n) if n_clear is None else n_clear
    return (draws @ n) ** 2 - (draws @ m) ** 2 - (draws**2) @ (n**2 - m**2)


def far_with_ci(
    blocks: list[Block],
    accepted: list[sparse.csr_matrix],
    draws: np.ndarray,
    people_mask: np.ndarray | None = None,
    image_masks: list[np.ndarray] | None = None,
) -> tuple:
    """Dense FAR over both halves with a person-level 95% interval.

    ``people_mask`` (global ids) restricts to strangers who *both* belong to a group;
    ``image_masks`` (per block) restricts to pairs where either image has a condition.
    ``draws`` is reps x global people, as from :func:`stats.person_draws`.
    """
    numerator = np.zeros(len(draws))
    denominator = np.zeros(len(draws))
    total_accepted = total_attempts = 0.0
    for index, (block, matrix) in enumerate(zip(blocks, accepted)):
        keep = np.ones(len(block.people), dtype=bool)
        if people_mask is not None:
            keep = people_mask[block.people]
        n = block.counts
        n_clear = None
        if image_masks is not None:
            n_clear = np.bincount(
                block.person, weights=~image_masks[index], minlength=len(block.people)
            ).astype(float)
        sub = matrix[keep][:, keep]
        c = draws[:, block.people[keep]]
        numerator += np.asarray(((sub @ c.T).T * c).sum(axis=1)).ravel()
        n_k = n[keep]
        m_k = None if n_clear is None else n_clear[keep]
        denominator += _attempt_terms(c, n_k, m_k)
        total_accepted += sub.sum()
        total_attempts += _attempt_terms(np.ones((1, keep.sum())), n_k, m_k)[0]
    if total_attempts == 0:
        return (None, None, None)
    point = float(total_accepted / total_attempts)
    with np.errstate(invalid="ignore", divide="ignore"):
        estimates = numerator / denominator
    estimates = estimates[np.isfinite(estimates)]
    low, high = np.percentile(estimates, CONFIDENCE)
    return (point, float(low), float(high))
