"""The decision layer: thresholds and Bayes actions."""
from __future__ import annotations

import numpy as np
import pytest

from mumbai_crowd.config import BAND_EDGES, COST_MATRIX, density_to_band
from mumbai_crowd.decision import (
    DistributionalPolicy,
    NaivePolicy,
    ThresholdPolicy,
    action_confusion,
    policy_report,
)


@pytest.fixture
def noisy_forecast():
    rng = np.random.default_rng(17)
    y = rng.gamma(2.0, 3.0, size=30000)
    pred = np.maximum(y + rng.normal(0.0, 2.0, size=y.shape), 0.0)
    return y, pred


def test_naive_policy_is_just_the_band_edges(noisy_forecast):
    y, pred = noisy_forecast
    assert np.array_equal(NaivePolicy().decide(pred), density_to_band(pred))


def test_threshold_policy_beats_the_naive_edges(noisy_forecast):
    """The whole point of the decision layer, asserted."""
    y, pred = noisy_forecast
    tp = ThresholdPolicy().fit(y, pred)
    naive_cost = float(np.mean(COST_MATRIX[density_to_band(y), NaivePolicy().decide(pred)]))
    tuned_cost = float(np.mean(COST_MATRIX[density_to_band(y), tp.decide(pred)]))
    assert tuned_cost <= naive_cost


def test_fitted_thresholds_sit_below_the_physical_edges(noisy_forecast):
    """Under an asymmetric cost you must worry earlier than the physics says."""
    y, pred = noisy_forecast
    tp = ThresholdPolicy().fit(y, pred)
    assert (tp.thresholds_ <= np.asarray(BAND_EDGES) + 1e-9).all()


def test_thresholds_stay_ordered(noisy_forecast):
    y, pred = noisy_forecast
    t = ThresholdPolicy().fit(y, pred).thresholds_
    assert t[0] <= t[1] <= t[2]


def test_threshold_policy_reduces_dangerous_misses(noisy_forecast):
    y, pred = noisy_forecast
    tp = ThresholdPolicy().fit(y, pred)
    before = policy_report(y, NaivePolicy().decide(pred))
    after = policy_report(y, tp.decide(pred))
    assert after["dangerous_miss"] < before["dangerous_miss"]
    assert after["danger_recall"] > before["danger_recall"]
    # ...and is honest about paying for it in false alarms.
    assert after["false_alarm"] >= before["false_alarm"]


def test_decide_before_fit_raises():
    with pytest.raises(RuntimeError):
        ThresholdPolicy().decide(np.array([1.0]))


def test_a_harsher_cost_matrix_pushes_the_thresholds_down(noisy_forecast):
    y, pred = noisy_forecast
    mild = ThresholdPolicy(cost_matrix=np.array([
        [0.0, 60.0, 400.0, 1500.0],
        [350.0, 0.0, 180.0, 900.0],
        [500.0, 300.0, 0.0, 500.0],
        [900.0, 700.0, 400.0, 0.0],
    ])).fit(y, pred)
    harsh = ThresholdPolicy().fit(y, pred)      # the project's real matrix
    assert harsh.thresholds_[2] <= mild.thresholds_[2]


# --- distributional -------------------------------------------------------

TAUS = np.array([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.995])


def _quantile_matrix(samples: np.ndarray) -> np.ndarray:
    return np.quantile(samples, TAUS, axis=1).T


def test_band_probabilities_are_a_valid_distribution():
    rng = np.random.default_rng(4)
    q = _quantile_matrix(rng.gamma(2.0, 3.0, size=(500, 3000)))
    probs = DistributionalPolicy(taus=TAUS).band_probabilities(q)
    assert probs.shape == (500, 4)
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_band_probabilities_recover_the_truth():
    """Against a distribution we can compute exactly, the estimate must be close."""
    rng = np.random.default_rng(5)
    samples = rng.gamma(2.0, 3.0, size=(300, 6000))
    q = _quantile_matrix(samples)
    est = DistributionalPolicy(taus=TAUS).band_probabilities(q)
    truth = np.stack([
        (samples < 4).mean(axis=1),
        ((samples >= 4) & (samples < 8)).mean(axis=1),
        ((samples >= 8) & (samples < 12)).mean(axis=1),
        (samples >= 12).mean(axis=1),
    ], axis=1)
    assert np.abs(est - truth).mean() < 0.02


def test_quantile_crossing_is_repaired():
    dp = DistributionalPolicy(taus=TAUS)
    crossed = np.tile(np.array([5.0, 4.0, 6.0, 5.5, 9.0, 8.0, 11.0, 13.0, 12.0]), (3, 1))
    probs = dp.band_probabilities(crossed)
    assert np.all(probs >= 0) and np.allclose(probs.sum(axis=1), 1.0)


def test_bayes_action_follows_the_cost_matrix():
    """A near-certain dangerous coach must trigger the relief-rake action."""
    dp = DistributionalPolicy(taus=TAUS)
    certain_danger = np.tile(np.linspace(18.0, 25.0, len(TAUS)), (5, 1))
    certain_calm = np.tile(np.linspace(0.05, 0.6, len(TAUS)), (5, 1))
    assert (dp.decide(certain_danger) == 3).all()
    assert (dp.decide(certain_calm) == 0).all()


def test_expected_costs_are_consistent_with_the_chosen_action():
    rng = np.random.default_rng(6)
    q = _quantile_matrix(rng.gamma(2.0, 3.0, size=(200, 2000)))
    dp = DistributionalPolicy(taus=TAUS)
    costs = dp.expected_costs(q)
    assert np.array_equal(dp.decide(q), np.argmin(costs, axis=1))


def test_wrong_quantile_matrix_shape_is_rejected():
    with pytest.raises(ValueError):
        DistributionalPolicy(taus=TAUS).band_probabilities(np.zeros((10, 3)))


def test_policy_report_and_confusion_agree():
    rng = np.random.default_rng(7)
    y = rng.gamma(2.0, 3.0, 5000)
    action = density_to_band(y + rng.normal(0, 2, 5000))
    rep = policy_report(y, action)
    cm = action_confusion(y, action, normalize="true")
    assert rep["danger_recall"] == pytest.approx(cm.loc["true_DANGEROUS", "act_DANGEROUS"])
    assert rep["dangerous_miss"] == pytest.approx(
        cm.loc["true_DANGEROUS", ["act_COMFORTABLE", "act_BUSY"]].sum()
    )
