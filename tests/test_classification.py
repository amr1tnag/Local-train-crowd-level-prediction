"""The band classifier, its calibrators, and the shared Bayes rule.

These exist because the classifier route ends up producing the cheapest policy
in the project, so its probability machinery is load-bearing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mumbai_crowd.classification import (
    N_BANDS,
    BandClassifier,
    CalibratedBandClassifier,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    expected_class_probabilities,
    split_validation_by_date,
)
from mumbai_crowd.config import COST_MATRIX, ModelConfig, SimConfig, density_to_band
from mumbai_crowd.decision import ProbabilityPolicy, bayes_action, policy_report
from mumbai_crowd.features import TARGET, build_design, prepare_matrix
from mumbai_crowd.simulate import simulate

CFG = ModelConfig(n_estimators=120, val_days=6, test_days=6)


# --- the shared Bayes rule ------------------------------------------------

def test_bayes_action_matches_the_cost_arithmetic():
    probs = np.array([
        [1.0, 0.0, 0.0, 0.0],       # certainly comfortable
        [0.0, 0.0, 0.0, 1.0],       # certainly dangerous
    ])
    assert list(bayes_action(probs)) == [0, 3]


def test_bayes_action_is_argmin_of_expected_cost():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(N_BANDS), size=500)
    assert np.array_equal(bayes_action(probs), np.argmin(probs @ COST_MATRIX, axis=1))


def test_bayes_action_acts_on_a_minority_of_danger_probability():
    """The asymmetry means a coach does not need to be *probably* dangerous."""
    probs = np.array([[0.0, 0.0, 0.70, 0.30]])
    assert bayes_action(probs)[0] == 3


def test_bayes_action_rejects_the_wrong_shape():
    with pytest.raises(ValueError):
        bayes_action(np.zeros((10, 3)))


def test_probability_policy_wraps_the_rule():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(N_BANDS), size=200)
    pol = ProbabilityPolicy()
    assert np.array_equal(pol.decide(probs), bayes_action(probs))
    assert np.allclose(pol.expected_costs(probs), probs @ COST_MATRIX)


# --- calibrators ----------------------------------------------------------

@pytest.fixture
def miscalibrated():
    """Scores that rank correctly but are systematically over-confident."""
    rng = np.random.default_rng(7)
    n = 8000
    truth = rng.random(n) < 0.15
    raw = np.where(truth, rng.beta(6, 2, n), rng.beta(2, 6, n))
    probs = np.column_stack([1 - raw, np.zeros(n), np.zeros(n), raw])
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = np.where(truth, 3, 0)
    return probs, y


@pytest.mark.parametrize("cal", [IdentityCalibrator(), TemperatureCalibrator(), IsotonicCalibrator()])
def test_calibrators_return_valid_distributions(cal, miscalibrated):
    probs, y = miscalibrated
    out = cal.fit(probs, y).transform(probs)
    assert out.shape == probs.shape
    assert np.all(out >= 0)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_identity_calibrator_changes_nothing(miscalibrated):
    probs, y = miscalibrated
    assert np.allclose(IdentityCalibrator().fit(probs, y).transform(probs), probs)


def test_isotonic_calibration_reduces_the_calibration_error(miscalibrated):
    """On the set it was fitted on, isotonic must improve reliability."""
    probs, y = miscalibrated
    cal = IsotonicCalibrator().fit(probs, y)
    out = cal.transform(probs)
    truth = (y == 3).astype(float)

    def ece(p):
        bins = np.clip((p * 10).astype(int), 0, 9)
        err = 0.0
        for b in range(10):
            m = bins == b
            if m.sum() > 20:
                err += m.mean() * abs(p[m].mean() - truth[m].mean())
        return err

    assert ece(out[:, 3]) < ece(probs[:, 3])


def test_temperature_calibration_finds_a_sensible_temperature(miscalibrated):
    probs, y = miscalibrated
    cal = TemperatureCalibrator().fit(probs, y)
    assert 0.05 < cal.temperature_ < 10.0


def test_temperature_above_one_softens_and_below_one_sharpens():
    probs = np.array([[0.7, 0.2, 0.07, 0.03]])
    cal = TemperatureCalibrator()
    cal.temperature_ = 3.0
    softened = cal.transform(probs)
    cal.temperature_ = 0.4
    sharpened = cal.transform(probs)
    assert softened[0].max() < probs[0].max()
    assert sharpened[0].max() > probs[0].max()


def test_calibrators_reject_use_before_fit(miscalibrated):
    probs, _ = miscalibrated
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().transform(probs)
    with pytest.raises(RuntimeError):
        TemperatureCalibrator().transform(probs)


def test_isotonic_rejects_a_wrong_shaped_matrix():
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit(np.zeros((10, 3)), np.zeros(10, dtype=int))


def test_isotonic_leaves_a_class_alone_when_it_has_no_positives():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(N_BANDS), size=400)
    y = np.zeros(400, dtype=int)          # nothing but COMFORTABLE
    cal = IsotonicCalibrator().fit(probs, y)
    assert cal.models_[3] is None
    out = cal.transform(probs)
    assert np.allclose(out.sum(axis=1), 1.0)


# --- the classifier end to end -------------------------------------------

@pytest.fixture(scope="module")
def design():
    obs = simulate(SimConfig(n_days=26, monitored_service_fraction=0.20), verbose=False).coach_observations
    split, cols, _ = build_design(obs, "schedule", val_days=CFG.val_days, test_days=CFG.test_days)
    return split, cols


def test_validation_split_is_temporal_and_disjoint(design):
    split, _ = design
    es, cal = split_validation_by_date(split.val)
    assert len(es) and len(cal)
    assert es["date"].max() < cal["date"].min()
    assert len(es) + len(cal) == len(split.val)
    assert not set(es["date"]) & set(cal["date"])


def test_validation_split_needs_two_dates():
    one_day = pd.DataFrame({"date": pd.to_datetime(["2024-06-01"] * 5)})
    with pytest.raises(ValueError):
        split_validation_by_date(one_day)


@pytest.fixture(scope="module")
def fitted(design):
    split, cols = design
    es, cal = split_validation_by_date(split.val)
    out = {}
    for weight in (None, "balanced"):
        m = CalibratedBandClassifier(
            BandClassifier(cfg=CFG, class_weight=weight), calibrator=IdentityCalibrator()
        )
        m.fit(
            prepare_matrix(split.train, cols), split.train[TARGET].to_numpy(float),
            prepare_matrix(es, cols), es[TARGET].to_numpy(float),
            prepare_matrix(cal, cols), cal[TARGET].to_numpy(float),
        )
        out["balanced" if weight else "none"] = m.predict_proba(prepare_matrix(split.test, cols))
    return out


def test_predicted_probabilities_are_well_formed(design, fitted):
    split, _ = design
    for probs in fitted.values():
        assert probs.shape == (len(split.test), N_BANDS)
        assert np.all(probs >= 0)
        assert np.allclose(probs.sum(axis=1), 1.0)


def test_unweighted_classifier_is_roughly_calibrated_on_the_base_rate(design, fitted):
    """Trained on log-loss with no reweighting, mean P(band) should track the
    observed frequency -- this is what makes post-hoc calibration unnecessary."""
    split, _ = design
    observed = np.bincount(
        density_to_band(split.test[TARGET].to_numpy(float)), minlength=N_BANDS
    ) / len(split.test)
    predicted = fitted["none"].mean(axis=0)
    assert np.abs(predicted - observed).max() < 0.06


def test_class_weighting_inflates_the_danger_probability(design, fitted):
    """Reweighting breaks the probability scale by construction; assert it."""
    assert fitted["balanced"][:, 3].mean() > 2.0 * fitted["none"][:, 3].mean()


def test_class_weighting_trades_misses_for_false_alarms(design, fitted):
    split, _ = design
    y = split.test[TARGET].to_numpy(float)
    plain = policy_report(y, bayes_action(fitted["none"]))
    heavy = policy_report(y, bayes_action(fitted["balanced"]))
    assert heavy["dangerous_miss"] <= plain["dangerous_miss"]
    assert heavy["false_alarm"] >= plain["false_alarm"]


def test_classifier_beats_the_naive_edges_baseline(design, fitted):
    """The whole reason the classifier route is in the project."""
    from mumbai_crowd.metrics import expected_cost
    from mumbai_crowd.regression import BoostedRegressor, fit_predict
    from mumbai_crowd.losses import SquaredError

    split, cols = design
    y = split.test[TARGET].to_numpy(float)
    _, preds = fit_predict(BoostedRegressor(loss=SquaredError(), cfg=CFG), split, cols)
    l2_naive = expected_cost(y, preds["test"])
    clf = float(np.mean(COST_MATRIX[density_to_band(y), bayes_action(fitted["none"])]))
    assert clf < l2_naive


def test_expected_class_probabilities_labels_the_bands(fitted):
    s = expected_class_probabilities(fitted["none"])
    assert list(s.index) == ["COMFORTABLE", "BUSY", "CRUSH", "DANGEROUS"]
    assert s.sum() == pytest.approx(1.0)


def test_unsupported_class_weight_is_rejected(design):
    split, cols = design
    with pytest.raises(ValueError):
        BandClassifier(cfg=CFG, class_weight="inverse").fit(
            prepare_matrix(split.train, cols), split.train[TARGET].to_numpy(float)
        )
