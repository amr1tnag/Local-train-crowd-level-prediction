"""Metrics, checked against hand-computable cases."""
from __future__ import annotations

import numpy as np
import pytest

from mumbai_crowd.config import COST_MATRIX, density_to_band
from mumbai_crowd.metrics import (
    asymmetric_cost,
    danger_reliability_table,
    band_confusion,
    bias,
    critical_miss_rate,
    danger_precision,
    danger_recall,
    dangerous_miss_rate,
    expected_cost,
    false_alarm_rate,
    leaderboard,
    mae,
    pinball,
    r2,
    regression_report,
    rmse,
    tail_under_error,
    under_prediction_rate,
)


def test_band_edges_partition_the_line():
    d = np.array([0.0, 3.99, 4.0, 7.99, 8.0, 11.99, 12.0, 50.0])
    assert list(density_to_band(d)) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_basic_error_metrics():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 2.0])
    assert rmse(y, p) == pytest.approx(np.sqrt(2 / 3))
    assert mae(y, p) == pytest.approx(2 / 3)
    assert bias(y, p) == pytest.approx(0.0)
    assert r2(y, y) == pytest.approx(1.0)


def test_asymmetric_cost_weights_the_two_directions_differently():
    y = np.array([10.0])
    assert asymmetric_cost(y, np.array([8.0]), 6.0, 1.0) == pytest.approx(12.0)   # 2 short
    assert asymmetric_cost(y, np.array([12.0]), 6.0, 1.0) == pytest.approx(2.0)   # 2 over


def test_under_prediction_rate():
    y = np.array([5.0, 5.0, 5.0, 5.0])
    p = np.array([4.0, 4.0, 6.0, 5.0])
    assert under_prediction_rate(y, p) == pytest.approx(0.5)


def test_tail_under_error_only_looks_at_dangerous_coaches():
    y = np.array([1.0, 13.0, 14.0])
    p = np.array([0.0, 10.0, 16.0])          # 3 short, then over
    assert tail_under_error(y, p) == pytest.approx(1.5)
    assert np.isnan(tail_under_error(np.array([1.0]), np.array([1.0])))


def test_expected_cost_reads_the_matrix_the_right_way_round():
    """Row = truth, column = prediction.  Transposing this is a classic bug."""
    truly_dangerous, predicted_comfortable = np.array([13.0]), np.array([1.0])
    truly_comfortable, predicted_dangerous = np.array([1.0]), np.array([13.0])
    assert expected_cost(truly_dangerous, predicted_comfortable) == COST_MATRIX[3, 0]
    assert expected_cost(truly_comfortable, predicted_dangerous) == COST_MATRIX[0, 3]
    assert COST_MATRIX[3, 0] > COST_MATRIX[0, 3] * 10


def test_cost_matrix_shape_and_zero_diagonal():
    assert COST_MATRIX.shape == (4, 4)
    assert np.allclose(np.diag(COST_MATRIX), 0.0)
    # Under-prediction (lower triangle) must dominate over-prediction.
    assert COST_MATRIX[np.tril_indices(4, -1)].sum() > 5 * COST_MATRIX[np.triu_indices(4, 1)].sum()


def test_safety_metrics_on_a_hand_built_case():
    #                    truth:  DANGEROUS  DANGEROUS  DANGEROUS  COMFORTABLE
    y = np.array([13.0, 13.0, 13.0, 1.0])
    p = np.array([1.0, 5.0, 13.0, 9.0])
    #             COMF  BUSY  DANGER   CRUSH
    assert critical_miss_rate(y, p) == pytest.approx(1 / 3)
    assert dangerous_miss_rate(y, p) == pytest.approx(2 / 3)
    assert danger_recall(y, p) == pytest.approx(1 / 3)
    assert danger_precision(y, p) == pytest.approx(1.0)
    assert false_alarm_rate(y, p) == pytest.approx(1.0)


def test_perfect_prediction_is_free():
    rng = np.random.default_rng(0)
    y = rng.gamma(2, 3, 500)
    assert expected_cost(y, y) == pytest.approx(0.0)
    assert dangerous_miss_rate(y, y) == pytest.approx(0.0)


def test_confusion_rows_sum_to_one_when_normalised():
    rng = np.random.default_rng(1)
    y = rng.gamma(2, 3, 3000)
    cm = band_confusion(y, y + rng.normal(0, 2, 3000), normalize="true")
    assert cm.shape == (4, 4)
    assert np.allclose(cm.sum(axis=1), 1.0)


def test_pinball_reduces_to_half_mae_at_the_median():
    rng = np.random.default_rng(2)
    y, p = rng.normal(size=400), rng.normal(size=400)
    assert pinball(y, p, 0.5) == pytest.approx(0.5 * mae(y, p))


def test_report_and_leaderboard_round_trip():
    rng = np.random.default_rng(3)
    y = rng.gamma(2, 3, 1000)
    board = leaderboard({
        "good": regression_report(y, y + rng.normal(0, 0.2, 1000)),
        "bad": regression_report(y, y * 0.2),
    })
    assert board.index[0] == "good"
    assert board.loc["good", "exp_cost_inr"] < board.loc["bad", "exp_cost_inr"]


# --- reliability diagnostic ----------------------------------------------

def test_reliability_table_recovers_a_known_relationship():
    """A perfectly calibrated score must land on the diagonal."""
    rng = np.random.default_rng(9)
    n = 60000
    p = rng.uniform(0.0, 0.9, n)
    dangerous = rng.random(n) < p
    # 13 is inside DANGEROUS, 1 is inside COMFORTABLE.
    y = np.where(dangerous, 13.0, 1.0)
    t = danger_reliability_table(p, y)
    assert len(t) > 4
    assert np.abs(t["predicted"] - t["observed"]).max() < 0.03


def test_reliability_table_detects_over_confidence():
    rng = np.random.default_rng(10)
    n = 40000
    p = rng.uniform(0.0, 0.9, n)
    dangerous = rng.random(n) < p / 2.0        # truth is half what is claimed
    y = np.where(dangerous, 13.0, 1.0)
    t = danger_reliability_table(p, y)
    assert (t["observed"] < t["predicted"]).all()


def test_reliability_table_drops_thin_bins():
    rng = np.random.default_rng(11)
    p = np.concatenate([rng.uniform(0, 0.01, 5000), np.array([0.9] * 3)])
    y = np.concatenate([np.ones(5000), np.full(3, 13.0)])
    t = danger_reliability_table(p, y, min_count=30)
    assert (t["n"] >= 30).all()
    assert t["bin_low"].max() < 0.75


def test_reliability_table_columns_and_ordering():
    rng = np.random.default_rng(12)
    p = rng.uniform(0, 1, 20000)
    y = np.where(rng.random(20000) < p, 13.0, 1.0)
    t = danger_reliability_table(p, y)
    assert list(t.columns) == ["bin_low", "bin_high", "predicted", "observed", "n"]
    assert t["bin_low"].is_monotonic_increasing
