import csv

import pytest

from edgeauth.evaluation import PAIR_COLUMNS, load_pairs, synthetic_report
from edgeauth.metrics import Trial, calibrate, curve, curve_summary, operating_point


def test_tied_scores_enter_roc_together():
    trials = [Trial(True, 0.5), Trial(False, 0.5)]
    points = curve(trials)
    assert len(points) == 2
    assert curve_summary(points) == {"roc_auc": 0.5, "eer": 0.5}
    assert calibrate(trials, target_fmr=0) is None


def test_operating_point_separates_matcher_failures_from_attempt_failures():
    trials = [
        Trial(True, 0.9),
        Trial(True, 0.75),
        Trial(True, None),
        Trial(False, 0.75),
        Trial(False, 0.3),
        Trial(False, None),
    ]
    threshold = calibrate(trials, target_fmr=0.0)
    assert threshold == 0.9
    point = operating_point(trials, threshold)
    assert point["fmr"] == 0
    assert point["fnmr"] == 0.5
    assert point["far"] == 0
    assert point["frr"] == pytest.approx(2 / 3)
    assert point["impostor_scored"] == 2


def make_pairs(tmp_path, *, leak=False, duplicate=False, reuse=False):
    path = tmp_path / "pairs.csv"
    rows = [
        ("calibration", "cal-a", "cal-a", "cal-a1", "cal-a2", "cal-a3", "cal-a4"),
        ("calibration", "cal-a", "cal-b", "cal-a1", "cal-a2", "cal-a3", "cal-b4"),
        ("test", "test-a", "test-a", "test-a1", "test-a2", "test-a3", "test-a4"),
        (
            "test",
            "test-a",
            "cal-b" if leak else "test-b",
            "test-a1",
            "test-a2",
            "test-a3",
            "test-b4",
        ),
    ]
    if duplicate:
        rows.append(rows[-1])
    for row in rows:
        for filename in row[3:]:
            (tmp_path / filename).write_text(filename, encoding="utf-8")
    if reuse:
        (tmp_path / "test-b4").write_text("test-a1", encoding="utf-8")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(PAIR_COLUMNS)
        writer.writerows(rows)
    return path


def test_loads_identity_disjoint_trial_manifest(tmp_path):
    pairs = load_pairs(make_pairs(tmp_path))
    assert len(pairs) == 4
    assert sum(pair.genuine for pair in pairs) == 2


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("leak", "identity leakage"),
        ("duplicate", "duplicate verification trial"),
        ("reuse", "image reused"),
    ],
)
def test_rejects_manifest_contamination(tmp_path, corruption, message):
    with pytest.raises(ValueError, match=message):
        load_pairs(make_pairs(tmp_path, **{corruption: True}))


def test_synthetic_example_uses_one_fixed_threshold():
    report = synthetic_report()
    threshold = report["calibration"]["operating_point"]["threshold"]
    assert report["source"] == "synthetic example scores"
    assert len(report["test"]) == 4
    assert all(
        item["operating_point"]["threshold"] == threshold for item in report["test"].values()
    )
    assert "threshold" not in report["test"]["clean"]["curve"][1]
