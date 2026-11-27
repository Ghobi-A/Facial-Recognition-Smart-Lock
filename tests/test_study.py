import numpy as np
import pytest

from edgeauth.study import datasets, dense, embedders, lfw
from edgeauth.study.analysis import image_conditions
from edgeauth.study.stats import (
    Scores,
    auc,
    cross_calibrated,
    eer,
    far,
    frr,
    pair_bootstrap,
    person_bootstrap,
    person_draws,
    sweep,
    threshold_at_far,
    threshold_for_cost,
)


def separable() -> Scores:
    return Scores(np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.1, 0.2, 0.3, 0.4]))


def test_failure_to_acquire_never_accepted():
    scores = Scores(np.array([0.9, np.nan]), np.array([np.nan, 0.1]))
    assert frr(scores.genuine, 0.5) == 0.5  # the missing genuine probe is a false reject
    assert far(scores.impostor, -1.0) == 0.5  # a missing impostor can never get in


def test_perfect_separation():
    scores = separable()
    assert eer(scores)[0] == 0
    assert auc(scores) == pytest.approx(1.0)
    threshold = threshold_at_far(scores.impostor, 0.0)
    assert far(scores.impostor, threshold) == 0 and frr(scores.genuine, threshold) == 0


def test_eer_on_overlapping_scores():
    scores = Scores(np.array([0.9, 0.8, 0.7, 0.3]), np.array([0.6, 0.2, 0.1, 0.05]))
    value, threshold = eer(scores)
    assert value == pytest.approx(0.25)
    assert far(scores.impostor, threshold) == pytest.approx(frr(scores.genuine, threshold))


def test_weighted_sweep_with_unit_weights_matches_unweighted():
    rng = np.random.default_rng(3)
    plain = Scores(rng.normal(0.6, 0.2, 50), np.r_[rng.normal(0.2, 0.2, 49), np.nan])
    weighted = Scores(plain.genuine, plain.impostor, np.ones(50), np.ones(50))
    for left, right in zip(sweep(plain), sweep(weighted)):
        np.testing.assert_allclose(left, right)


def test_weights_act_like_duplicates():
    scores = Scores(np.array([0.9, 0.5]), np.array([0.6, 0.1]))
    duplicated = Scores(np.array([0.9, 0.5, 0.5]), np.array([0.6, 0.1]))
    weighted = Scores(scores.genuine, scores.impostor, np.array([1.0, 2.0]), np.ones(2))
    assert eer(weighted)[0] == pytest.approx(eer(duplicated)[0])


def test_cost_threshold_prefers_rejecting_strangers():
    rng = np.random.default_rng(0)
    scores = Scores(rng.normal(0.6, 0.15, 2000), rng.normal(0.2, 0.15, 2000))
    balanced = threshold_for_cost(scores, 1, 1)
    strict = threshold_for_cost(scores, 10, 1)
    assert strict > balanced
    assert far(scores.impostor, strict) < far(scores.impostor, balanced)


def test_dense_threshold_hits_target_far():
    impostor = np.random.default_rng(1).normal(0.0, 0.1, 200_000)
    for target in (1e-2, 1e-3, 1e-4):
        threshold = threshold_at_far(impostor, target)
        assert far(impostor, threshold) <= target
        assert far(impostor, threshold) == pytest.approx(target, rel=0.05)


def test_cross_calibration_is_out_of_sample():
    genuine = np.array([0.9, 0.8, 0.5, 0.4])
    impostor = np.array([0.2, 0.1, 0.45, 0.35])
    half = np.array([0, 0, 1, 1])
    _, _, used = cross_calibrated(Scores(genuine, impostor), half, half, "eer")
    # half 1 is judged by half 0's EER threshold and vice versa
    assert used[1] == pytest.approx(eer(Scores(genuine[:2], impostor[:2]))[1])
    assert used[0] == pytest.approx(eer(Scores(genuine[2:], impostor[2:]))[1])
    g, i, fixed = cross_calibrated(Scores(genuine, impostor), half, half, [0.85, 0.42])
    assert fixed == [0.85, 0.42]
    assert list(g) == [True, False, True, False] and list(i) == [False, False, True, False]


def test_person_bootstrap_matches_pair_bootstrap_for_singletons():
    rng = np.random.default_rng(5)
    outcomes = (rng.random(400) < 0.1).astype(float)
    people = np.arange(400)
    draws = person_draws(400, np.random.default_rng(6), 2000)
    person = person_bootstrap(outcomes, people, people, draws)
    pair = pair_bootstrap(outcomes, np.random.default_rng(7))
    assert person[0] == pair[0]
    assert person[2] - person[1] == pytest.approx(pair[2] - pair[1], rel=0.15)


def test_person_bootstrap_widens_for_repeated_people():
    # 40 people with 10 pairs each, outcomes perfectly correlated within a person
    people = np.repeat(np.arange(40), 10)
    outcomes = (people < 8).astype(float)
    draws = person_draws(40, np.random.default_rng(1), 2000)
    person = person_bootstrap(outcomes, people, people, draws)
    pair = pair_bootstrap(outcomes, np.random.default_rng(1))
    assert person[2] - person[1] > 2 * (pair[2] - pair[1])


def test_dense_far_matches_brute_force():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(30, 4))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors[3] = np.nan  # failure to acquire: an attempt that can never be accepted
    person = np.repeat(np.arange(10), 3)
    block = dense.Block(vectors, person, np.arange(10), "cosine")
    threshold = 0.3
    brute_hits = brute_attempts = 0
    for i in range(30):
        for j in range(i + 1, 30):
            if person[i] != person[j]:
                brute_attempts += 1
                brute_hits += np.nan_to_num(vectors[i] @ vectors[j], nan=-np.inf) >= threshold
    assert len(block.impostor_scores()) == brute_attempts
    draws = person_draws(10, np.random.default_rng(0), 200)
    matrix = dense.accepted_matrix(block, threshold)
    point, low, high = dense.far_with_ci([block], [matrix], draws)
    assert point == pytest.approx(brute_hits / brute_attempts)
    assert low <= point <= high
    flagged = np.zeros(30, dtype=bool)
    flagged[:6] = True
    masked = dense.accepted_matrix(block, threshold, flagged)
    hits = attempts = 0
    for i in range(30):
        for j in range(i + 1, 30):
            if person[i] != person[j] and (flagged[i] or flagged[j]):
                attempts += 1
                hits += np.nan_to_num(vectors[i] @ vectors[j], nan=-np.inf) >= threshold
    assert dense.far_with_ci([block], [masked], draws, image_masks=[flagged])[0] == (
        pytest.approx(hits / attempts)
    )


def test_pairs_protocol_parsing(tmp_path):
    text = "1\t2\nAnn\t1\t2\nBob\t1\t3\nAnn\t1\tBob\t2\nCat\t4\tDan\t1\n"
    path = tmp_path / "pairs.txt"
    path.write_text(text)
    rows = lfw.parse_pairs(path)
    assert [r.genuine for r in rows] == [True, True, False, False]
    assert rows[0].image_b == "Ann/Ann_0002.jpg"
    assert rows[3].person_b == "Dan" and rows[3].image_a == "Cat/Cat_0004.jpg"
    manifest = tmp_path / "m.csv"
    lfw.write_manifest(rows, manifest)
    assert lfw.read_manifest(manifest) == rows


def test_dataset_round_trip_and_leakage_check(tmp_path):
    rows = [
        lfw.PairRow(0, 1, True, "Ann", "Ann/1.jpg", "Ann", "Ann/2.jpg"),
        lfw.PairRow(1, 2, False, "Ann", "Ann/1.jpg", "Bob", "Bob/1.jpg"),
    ]
    people = {"Ann": {"sex": "female"}, "Bob": {"sex": "male", "ethnicity": "white"}}
    images = {"Ann/1.jpg": {"blurry": "yes"}, "Bob/1.jpg": {"blurry": "no"}}
    datasets.write(tmp_path, {"name": "toy", "image_root": "."}, rows, people, images)
    loaded = datasets.load(tmp_path)
    assert loaded.rows == rows and loaded.people == people and loaded.images == images
    leaky = rows + [lfw.PairRow(2, 7, True, "Bob", "Bob/1.jpg", "Bob", "Bob/2.jpg")]
    with pytest.raises(ValueError, match="identity leakage"):
        datasets.write(tmp_path, {"name": "toy", "image_root": "."}, leaky, people, images)


def test_balanced_pairs_stay_inside_halves():
    images = {f"p{i}": [f"p{i}/{j}.jpg" for j in range(4)] for i in range(40)}
    rows = datasets.balanced_pairs(images, "salt", per_person=3, seed=0)
    datasets.check_disjoint(rows)
    genuine = sum(r.genuine for r in rows)
    assert genuine == 120 and len(rows) == 240


def test_image_conditions_include_pose():
    data = datasets.Dataset("toy", None, {}, [], {}, {"a.jpg": {"blurry": "yes"}})
    flags = image_conditions(data, {"a.jpg": 0.5, "b.jpg": 0.0, "c.jpg": float("nan")})
    assert flags["blurry"] == {"a.jpg": True}
    assert flags["pose_turned"] == {"a.jpg": True, "b.jpg": False}


def test_pair_scores_orientation():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]])
    a, b = np.array([0, 0, 0]), np.array([0, 1, 2])
    cosine = embedders.pair_scores("sface-fp32", vectors, a, b)
    assert cosine[0] > cosine[1] and np.isnan(cosine[2])
    distance = embedders.pair_scores("dlib-legacy", vectors, a, b)
    assert distance[0] > distance[1]  # negated distance: higher still means more similar


def test_central_face_selection():
    boxes = [(0, 0, 20, 20), (40, 40, 20, 20), (80, 0, 20, 20)]
    assert embedders.central(boxes, 100, 100) == 1
