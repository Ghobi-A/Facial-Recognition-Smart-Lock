import numpy as np
import pytest

from edgeauth.study import embedders, lfw
from edgeauth.study.analysis import pair_groups
from edgeauth.study.stats import (
    Scores,
    auc,
    bootstrap_mean,
    cross_calibrated,
    eer,
    far,
    frr,
    threshold_for_cost,
    threshold_for_far,
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
    threshold = threshold_for_far(scores, 0.0)
    assert far(scores.impostor, threshold) == 0 and frr(scores.genuine, threshold) == 0


def test_eer_on_overlapping_scores():
    scores = Scores(np.array([0.9, 0.8, 0.7, 0.3]), np.array([0.6, 0.2, 0.1, 0.05]))
    value, threshold = eer(scores)
    assert value == pytest.approx(0.25)
    assert far(scores.impostor, threshold) == pytest.approx(frr(scores.genuine, threshold))


def test_cost_threshold_prefers_rejecting_strangers():
    rng = np.random.default_rng(0)
    scores = Scores(rng.normal(0.6, 0.15, 2000), rng.normal(0.2, 0.15, 2000))
    balanced = threshold_for_cost(scores, 1, 1)
    strict = threshold_for_cost(scores, 10, 1)
    assert strict > balanced
    assert far(scores.impostor, strict) < far(scores.impostor, balanced)


def test_cross_calibration_is_out_of_sample():
    # Half 0 is easy, half 1 is shifted: each half must be judged by the other's threshold.
    genuine = np.array([0.9, 0.8, 0.5, 0.4])
    impostor = np.array([0.2, 0.1, 0.45, 0.35])
    half = np.array([0, 0, 1, 1])
    _, _, used = cross_calibrated(Scores(genuine, impostor), half, half, "far_1pct")
    assert used[0] == pytest.approx(0.5)  # chosen on half 1 only
    assert used[1] == pytest.approx(0.8)  # chosen on half 0 only


def test_bootstrap_interval_brackets_estimate():
    point, low, high = bootstrap_mean(np.r_[np.ones(10), np.zeros(90)], np.random.default_rng(1))
    assert point == pytest.approx(0.1)
    assert low <= point <= high
    assert bootstrap_mean(np.array([]), np.random.default_rng(1)) == (None, None, None)


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


def attribute_scores(male: float, white: float, glasses: float = -1.0) -> dict:
    scores = {name: -1.0 for name in ("Asian", "Black", "Indian", *lfw.AGE, *lfw.BINARY.values())}
    scores.update({"Male": male, "White": white, "Eyeglasses": glasses, "Youth": 1.0})
    return scores


def test_subgroups_require_both_impostors_in_group():
    attributes = {
        "Ann/Ann_0001.jpg": attribute_scores(-1, 2, glasses=1),
        "Ann/Ann_0002.jpg": attribute_scores(-1, 2),
        "Bob/Bob_0001.jpg": attribute_scores(1, 2),
    }
    people = lfw.person_groups(attributes)
    rows = [
        lfw.PairRow(0, 1, True, "Ann", "Ann/Ann_0001.jpg", "Ann", "Ann/Ann_0002.jpg"),
        lfw.PairRow(1, 1, False, "Ann", "Ann/Ann_0002.jpg", "Bob", "Bob/Bob_0001.jpg"),
    ]
    yaw = {"Ann/Ann_0001.jpg": 0.05, "Ann/Ann_0002.jpg": 0.4, "Bob/Bob_0001.jpg": 0.0}
    genuine, impostor = pair_groups(rows, attributes, people, yaw)
    assert genuine["sex"] == "female" and genuine["eyeglasses"] == "yes"
    assert genuine["pose"] == "turned"
    assert "sex" not in impostor  # female vs male impostor is not a same-group attempt
    assert impostor["ethnicity"] == "white"


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
