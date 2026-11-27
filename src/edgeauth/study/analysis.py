"""Turn cached embeddings into the study's results, for any prepared dataset.

Per model:
* EER / AUC on the dataset's pairs, with person-level bootstrap intervals;
* operating points: pair-calibrated (EER, cost 10:1, the 2022 default for dlib) and
  security-constrained (FAR = 1e-2, 1e-3, 1e-4, calibrated on the other half's dense impostors);
  each reports out-of-sample FRR (genuine pairs) and FAR (dense stranger pairs);
* subgroups at the cost-based and FAR=1e-3 policies: FRR from genuine pairs, FAR from *every*
  same-group stranger pair in the dense sets.
"""

from pathlib import Path

import numpy as np

from . import dense, embedders, lfw
from .datasets import DEMOGRAPHIC_KEYS, Dataset, half_of
from .stats import (
    FAR_TARGETS,
    Scores,
    cross_calibrated,
    pair_bootstrap,
    person_bootstrap,
    person_curve_stats,
    person_draws,
    sweep,
)

YAW_TURNED = 0.2  # |nose offset| above 0.2 inter-ocular distances counts as a turned head
LEGACY_DLIB_DISTANCE = 0.6  # face_recognition's default tolerance, used unchanged in 2022
SUBGROUP_POLICIES = ("cost_10to1", "far_1e-03")
BOOTSTRAP_REPS = 1000


def policy_name(target: float) -> str:
    return f"far_{target:.0e}"


def image_conditions(dataset: Dataset, yaw: dict[str, float]) -> dict[str, dict[str, bool]]:
    """Per condition key, image -> flagged. Pose comes from detector landmarks (SFace run)."""
    conditions: dict[str, dict[str, bool]] = {}
    for image, labels in dataset.images.items():
        for key, value in labels.items():
            conditions.setdefault(key, {})[image] = value == "yes"
    if yaw:
        conditions["pose_turned"] = {
            image: bool(abs(value) > YAW_TURNED)
            for image, value in yaw.items()
            if np.isfinite(value)
        }
    return conditions


def analyse(dataset: Dataset, embedding_dir: Path, seed: int = 2022) -> dict:
    rows = dataset.rows
    genuine = np.array([row.genuine for row in rows])
    half = np.array([half_of(row.fold) for row in rows])
    people = sorted({r.person_a for r in rows} | {r.person_b for r in rows})
    person_id = {person: i for i, person in enumerate(people)}
    pa = np.array([person_id[r.person_a] for r in rows])
    pb = np.array([person_id[r.person_b] for r in rows])
    draws = person_draws(len(people), np.random.default_rng(seed), BOOTSTRAP_REPS)

    yaw = {}
    if (embedding_dir / "sface-fp32.npz").exists():
        index, _, yaw_values = embedders.load(embedding_dir / "sface-fp32.npz")
        if yaw_values is not None:
            yaw = {image: float(yaw_values[i]) for image, i in index.items()}
    conditions = image_conditions(dataset, yaw)

    # images of each half, for the dense impostor sets
    half_images: list[dict[str, str]] = [{}, {}]
    for row in rows:
        h = half_of(row.fold)
        half_images[h][row.image_a] = row.person_a
        half_images[h][row.image_b] = row.person_b

    results = {
        "dataset": dataset.name,
        "description": dataset.meta.get("description"),
        "labels": dataset.meta.get("labels"),
        "manifest_sha256": lfw.manifest_digest(dataset.manifest),
        "people": len(people),
        "pairs": len(rows),
        "genuine_pairs": int(genuine.sum()),
        "protocol": "two-fold cross-calibration over identity-disjoint halves (folds 1-5 <-> 6-10)",
        "uncertainty": f"95% person-level (pigeonhole) bootstrap, {BOOTSTRAP_REPS} resamples; "
        "pair-level intervals kept as *_pair_ci for comparison",
        "seed": seed,
        "models": {},
    }

    for model in embedders.MODELS:
        path = embedding_dir / f"{model}.npz"
        if not path.exists():
            continue
        rng = np.random.default_rng(seed)
        index, vectors, _ = embedders.load(path)
        ia = np.array([index[r.image_a] for r in rows])
        ib = np.array([index[r.image_b] for r in rows])
        raw = embedders.pair_scores(model, vectors, ia, ib)
        scores = Scores(raw[genuine], raw[~genuine])
        g_people, i_people = (pa[genuine], pb[genuine]), (pa[~genuine], pb[~genuine])
        used_images = sorted(set(index) & ({r.image_a for r in rows} | {r.image_b for r in rows}))
        acquired = np.all(np.isfinite(vectors[[index[i] for i in used_images]]), axis=1)

        blocks = []
        for h in (0, 1):
            images = sorted(half_images[h])
            local = {p: i for i, p in enumerate(sorted(set(half_images[h].values())))}
            blocks.append(
                dense.Block(
                    vectors[[index[i] for i in images]],
                    np.array([local[half_images[h][i]] for i in images]),
                    np.array([person_id[p] for p in sorted(local, key=local.get)]),
                    "cosine" if embedders.SCORING[model] == "cosine" else "euclidean",
                )
            )
        block_images = [sorted(half_images[h]) for h in (0, 1)]

        thresholds, far_curve, frr_curve = sweep(scores)
        keep = np.unique(np.linspace(0, len(thresholds) - 1, 400).astype(int))
        curve_stats = person_curve_stats(scores, g_people, i_people, draws[:500])
        entry = {
            "scoring": embedders.SCORING[model],
            "failure_to_acquire": float(1 - acquired.mean()),
            "dense_impostor_pairs": [int(len(b.impostor_scores())) for b in blocks],
            **curve_stats,
            "curve": {"far": far_curve[keep].tolist(), "frr": frr_curve[keep].tolist()},
            "operating_points": {},
            "subgroups": {},
        }

        policies: dict[str, list[float]] = {}
        for name in ("eer", "cost_10to1"):
            _, _, used = cross_calibrated(scores, half[genuine], half[~genuine], name)
            policies[name] = used
        for target, used in dense.thresholds_at_far(blocks, FAR_TARGETS).items():
            policies[policy_name(target)] = used
        if model == "dlib-legacy":
            policies["legacy_default_0.6"] = [-LEGACY_DLIB_DISTANCE] * 2

        accepted_blocks = {}
        for name, used in policies.items():
            g_accept, i_accept, _ = cross_calibrated(scores, half[genuine], half[~genuine], used)
            accepted_blocks[name] = [dense.accepted_matrix(b, t) for b, t in zip(blocks, used)]
            entry["operating_points"][name] = {
                "thresholds": used,
                "frr": person_bootstrap(~g_accept, *g_people, draws),
                "frr_pair_ci": pair_bootstrap(~g_accept, rng),
                "far": dense.far_with_ci(blocks, accepted_blocks[name], draws),
                "far_protocol_pairs": person_bootstrap(i_accept, *i_people, draws),
            }

        for name in SUBGROUP_POLICIES:
            used = policies[name]
            g_accept, _, _ = cross_calibrated(scores, half[genuine], half[~genuine], used)
            groups: dict[str, dict] = {}
            for key in DEMOGRAPHIC_KEYS:
                values = sorted(
                    {dataset.people[p][key] for p in people if key in dataset.people.get(p, {})}
                )
                for value in values:
                    member = np.array([dataset.people.get(p, {}).get(key) == value for p in people])
                    gm = member[g_people[0]]
                    groups.setdefault(key, {})[value] = {
                        "people": int(member.sum()),
                        "genuine_pairs": int(gm.sum()),
                        "frr": person_bootstrap(
                            ~g_accept[gm], g_people[0][gm], g_people[1][gm], draws
                        ),
                        "far": dense.far_with_ci(
                            blocks, accepted_blocks[name], draws, people_mask=member
                        ),
                    }
            for key, flags in conditions.items():
                for value in (True, False):
                    pair_flag = np.array(
                        [flags.get(r.image_a, False) or flags.get(r.image_b, False) for r in rows]
                    )
                    known = np.array([r.image_a in flags and r.image_b in flags for r in rows])
                    gm = (pair_flag == value)[genuine] & known[genuine]
                    cell = {
                        "genuine_pairs": int(gm.sum()),
                        "frr": person_bootstrap(
                            ~g_accept[gm], g_people[0][gm], g_people[1][gm], draws
                        ),
                    }
                    if value:  # dense FAR for pairs where either image has the condition
                        masks = [
                            np.array([flags.get(i, False) for i in imgs]) for imgs in block_images
                        ]
                        matrices = [
                            dense.accepted_matrix(b, t, m) for b, t, m in zip(blocks, used, masks)
                        ]
                        cell["far"] = dense.far_with_ci(blocks, matrices, draws, image_masks=masks)
                    else:  # neither image flagged: complement, flag the clean images instead
                        masks = [
                            np.array([not flags.get(i, True) for i in imgs])
                            for imgs in block_images
                        ]
                        both_clean = [
                            dense.accepted_matrix(b, t) - dense.accepted_matrix(b, t, ~m)
                            for b, t, m in zip(blocks, used, masks)
                        ]
                        cell["far"] = _far_both(blocks, both_clean, masks, draws)
                    groups.setdefault(key, {})["yes" if value else "no"] = cell
            entry["subgroups"][name] = groups
        results["models"][model] = entry
    return results


def _far_both(blocks, matrices, clean_masks, draws) -> tuple:
    """FAR over stranger pairs where *both* images are clean (complement of 'either flagged')."""
    numerator = np.zeros(len(draws))
    denominator = np.zeros(len(draws))
    accepted_total = attempts_total = 0.0
    for block, matrix, clean in zip(blocks, matrices, clean_masks):
        m = np.bincount(block.person, weights=clean, minlength=len(block.people)).astype(float)
        c = draws[:, block.people]
        numerator += np.asarray(((matrix @ c.T).T * c).sum(axis=1)).ravel()
        denominator += (c @ m) ** 2 - (c**2) @ (m**2)
        accepted_total += matrix.sum()
        attempts_total += m.sum() ** 2 - (m**2).sum()
    if attempts_total == 0:
        return (None, None, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        estimates = numerator / denominator
    estimates = estimates[np.isfinite(estimates)]
    low, high = np.percentile(estimates, dense.CONFIDENCE)
    return (float(accepted_total / attempts_total), float(low), float(high))
