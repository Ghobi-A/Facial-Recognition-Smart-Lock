"""Turn cached embeddings into the study's results: overall metrics, operating points, subgroups."""

from pathlib import Path

import numpy as np

from . import embedders, lfw
from .stats import (
    POLICIES,
    Scores,
    bootstrap_curve_stats,
    bootstrap_mean,
    cross_calibrated,
    sweep,
)

YAW_TURNED = 0.2  # |nose offset| above 0.2 inter-ocular distances counts as a turned head
LEGACY_DLIB_DISTANCE = 0.6  # face_recognition's default tolerance, used unchanged in 2022
SUBGROUP_POLICY = "cost_10to1"


def pair_groups(
    rows: list[lfw.PairRow],
    attributes: dict[str, dict[str, float]],
    people: dict[str, dict[str, str]],
    yaw: dict[str, float],
) -> list[dict[str, str]]:
    """Subgroup labels per pair.

    Demographic labels (sex, ethnicity) are per person; an impostor pair belongs to a group only
    when both people do, because same-demographic impostors are the hard, relevant case. Capture
    conditions (glasses, lighting, blur, pose) are per image; a pair has a condition if either
    image does.
    """
    labelled = []
    for row in rows:
        groups = {}
        a, b = people.get(row.person_a), people.get(row.person_b)
        for key in ("sex", "ethnicity"):
            if a and b and a[key] == b[key]:
                groups[key] = a[key]
        images = [attributes.get(row.image_a), attributes.get(row.image_b)]
        if all(images):
            labels = [lfw.image_groups(scores) for scores in images]
            for key in lfw.BINARY:
                groups[key] = "yes" if "yes" in (labels[0][key], labels[1][key]) else "no"
        yaws = [yaw.get(row.image_a, np.nan), yaw.get(row.image_b, np.nan)]
        if all(np.isfinite(yaws)):
            groups["pose"] = "turned" if max(abs(v) for v in yaws) > YAW_TURNED else "frontal"
        labelled.append(groups)
    return labelled


MATCHED_IMPOSTORS_PER_GROUP = 2000


def matched_impostors(
    rows: list[lfw.PairRow], people: dict[str, dict[str, str]], seed: int
) -> list[tuple[int, str, str, str, str]]:
    """Supplementary same-demographic impostor pairs: (half, key, value, image_a, image_b).

    LFW's official impostor pairs are random, so same-ethnicity strangers are rare (a handful
    for minority groups). Matched strangers are the realistic threat and where demographic
    false-match differentials show up, so we sample them explicitly. Sampling stays inside
    each identity-disjoint half, so each pair is scored with the *other* half's threshold.
    """
    rng = np.random.default_rng(seed)
    images: dict[tuple[int, str], set[str]] = {}
    for row in rows:
        half = 0 if row.fold <= 5 else 1
        images.setdefault((half, row.person_a), set()).add(row.image_a)
        images.setdefault((half, row.person_b), set()).add(row.image_b)
    pairs = []
    for half in (0, 1):
        persons = sorted(p for h, p in images if h == half and p in people)
        for key in ("sex", "ethnicity"):
            for value in sorted({people[p][key] for p in persons}):
                group = [p for p in persons if people[p][key] == value]
                if len(group) < 2:
                    continue
                target = min(MATCHED_IMPOSTORS_PER_GROUP // 2, len(group) * (len(group) - 1) // 2)
                seen = set()
                while len(seen) < target:
                    a, b = sorted(rng.choice(len(group), size=2, replace=False))
                    seen.add((group[a], group[b]))
                for a, b in sorted(seen):
                    pick = lambda person: rng.choice(sorted(images[half, person]))  # noqa: E731
                    pairs.append((half, key, value, str(pick(a)), str(pick(b))))
    return pairs


def model_scores(model: str, rows: list[lfw.PairRow], embedding_dir: Path):
    index, vectors, yaw = embedders.load(embedding_dir / f"{model}.npz")
    a = np.array([index[row.image_a] for row in rows])
    b = np.array([index[row.image_b] for row in rows])
    acquired = np.all(np.isfinite(vectors), axis=1)
    return embedders.pair_scores(model, vectors, a, b), acquired, index, yaw


def analyse(manifest: Path, attributes_path: Path, embedding_dir: Path, seed: int = 2022) -> dict:
    rows = lfw.read_manifest(manifest)
    genuine = np.array([row.genuine for row in rows])
    half = np.array([0 if row.fold <= 5 else 1 for row in rows])
    attributes = lfw.parse_attributes(attributes_path)
    people = lfw.person_groups(attributes)

    _, _, sface_index, sface_yaw = model_scores("sface-fp32", rows, embedding_dir)
    yaw = {image: float(sface_yaw[i]) for image, i in sface_index.items()}
    groups = pair_groups(rows, attributes, people, yaw)
    matched = matched_impostors(rows, people, seed)

    results = {
        "dataset": "LFW, official 10-fold pairs (6,000 pairs: 3,000 genuine, 3,000 impostor)",
        "manifest_sha256": lfw.manifest_digest(manifest),
        "protocol": "two-fold cross-calibration: folds 1-5 <-> 6-10 (identity-disjoint)",
        "seed": seed,
        "matched_impostors": "up to 2,000 same-sex / same-ethnicity stranger pairs per group, "
        "sampled within each half and scored with the other half's threshold",
        "models": {},
    }
    for model in embedders.MODELS:
        if not (embedding_dir / f"{model}.npz").exists():
            continue
        rng = np.random.default_rng(seed)
        raw, acquired, index, _ = model_scores(model, rows, embedding_dir)
        scores = Scores(raw[genuine], raw[~genuine])
        used_images = sorted({row.image_a for row in rows} | {row.image_b for row in rows})
        fta = 1 - np.mean([acquired[index[image]] for image in used_images])

        thresholds, far_curve, frr_curve = sweep(scores)
        keep = np.unique(np.linspace(0, len(thresholds) - 1, 400).astype(int))
        entry = {
            "scoring": embedders.SCORING[model],
            "failure_to_acquire": float(fta),
            "pairs_with_fta": int(np.sum(~np.isfinite(raw))),
            **bootstrap_curve_stats(scores, rng),
            "curve": {
                "far": far_curve[keep].tolist(),
                "frr": frr_curve[keep].tolist(),
            },
            "operating_points": {},
            "subgroups": {},
        }

        decisions = {}
        for policy in POLICIES:
            g_accept, i_accept, used = cross_calibrated(
                scores, half[genuine], half[~genuine], policy
            )
            decisions[policy] = (g_accept, i_accept, used)
            entry["operating_points"][policy] = {
                "thresholds": used,
                "far": bootstrap_mean(i_accept, rng),
                "frr": bootstrap_mean(~g_accept, rng),
            }
        if model == "dlib-legacy":
            g_accept = np.nan_to_num(scores.genuine, nan=-np.inf) >= -LEGACY_DLIB_DISTANCE
            i_accept = np.nan_to_num(scores.impostor, nan=-np.inf) >= -LEGACY_DLIB_DISTANCE
            entry["operating_points"]["legacy_default_0.6"] = {
                "thresholds": [-LEGACY_DLIB_DISTANCE],
                "far": bootstrap_mean(i_accept, rng),
                "frr": bootstrap_mean(~g_accept, rng),
            }

        g_accept, i_accept, used = decisions[SUBGROUP_POLICY]
        _, vectors, _ = embedders.load(embedding_dir / f"{model}.npz")
        ma = np.array([index[m[3]] for m in matched])
        mb = np.array([index[m[4]] for m in matched])
        matched_scores = embedders.pair_scores(model, vectors, ma, mb)
        matched_threshold = np.array([used[m[0]] for m in matched])
        matched_accept = np.nan_to_num(matched_scores, nan=-np.inf) >= matched_threshold
        genuine_groups = [g for g, is_genuine in zip(groups, genuine) if is_genuine]
        impostor_groups = [g for g, is_genuine in zip(groups, genuine) if not is_genuine]
        keys = sorted({key for g in groups for key in g})
        for key in keys:
            values = sorted({g[key] for g in groups if key in g})
            table = {}
            for value in values:
                gm = np.array([g.get(key) == value for g in genuine_groups])
                im = np.array([g.get(key) == value for g in impostor_groups])
                table[value] = {
                    "genuine_pairs": int(gm.sum()),
                    "impostor_pairs": int(im.sum()),
                    "frr": bootstrap_mean(~g_accept[gm], rng),
                    "far": bootstrap_mean(i_accept[im], rng),
                }
                if key in ("sex", "ethnicity"):
                    mm = np.array([m[1] == key and m[2] == value for m in matched])
                    table[value]["matched_impostor_pairs"] = int(mm.sum())
                    table[value]["far_matched"] = bootstrap_mean(matched_accept[mm], rng)
            entry["subgroups"][key] = table
        results["models"][model] = entry
    return results
