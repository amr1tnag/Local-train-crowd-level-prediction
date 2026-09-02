"""End-to-end behaviour of the estimators.

The central experimental claim of the project is asserted here: holding the
data and the model family fixed and changing only the loss must move the
forecast in the safe direction and reduce the operator's expected cost.

Every model is fitted once, in a module-scoped fixture, and the individual
tests read the cached predictions.  Fitting per test made this file take an
hour.
"""
from __future__ import annotations

import numpy as np
import pytest

from mumbai_crowd.config import COST_OVER, COST_UNDER, ModelConfig, SimConfig, optimal_quantile
from mumbai_crowd.decision import ThresholdPolicy
from mumbai_crowd.features import TARGET, build_design, prepare_matrix
from mumbai_crowd.losses import AsymmetricSquaredError, LinexLoss, PinballLoss, SquaredError
from mumbai_crowd.metrics import (
    asymmetric_cost,
    expected_cost,
    regression_report,
    rmse,
    tail_under_error,
)
from mumbai_crowd.regression import (
    BoostedRegressor,
    HistoricalBaseline,
    LinearAsymmetric,
    MarginBaseline,
    MeanBaseline,
    QuantileEnsemble,
    fit_predict,
)
from mumbai_crowd.simulate import simulate

VAL_DAYS, TEST_DAYS = 3, 4
# Enough rounds for the bounded-gradient losses to actually converge; see the
# PinballLoss docstring on why 100 rounds would libel them.
CFG = ModelConfig(n_estimators=600, val_days=VAL_DAYS, test_days=TEST_DAYS)


@pytest.fixture(scope="module")
def design():
    obs = simulate(SimConfig(n_days=20, monitored_service_fraction=0.20), verbose=False).coach_observations
    split, cols, _ = build_design(obs, "schedule", val_days=VAL_DAYS, test_days=TEST_DAYS)
    return split, cols


@pytest.fixture(scope="module")
def fits(design):
    """Fit one model per loss, once, and cache the predictions."""
    split, cols = design
    losses = {
        "l2": SquaredError(),
        "asym_l2": AsymmetricSquaredError(COST_UNDER, COST_OVER),
        "pinball": PinballLoss(optimal_quantile()),
        "linex": LinexLoss(0.30),
    }
    out = {}
    for key, loss in losses.items():
        model, preds = fit_predict(BoostedRegressor(loss=loss, cfg=CFG), split, cols)
        out[key] = {"model": model, "preds": preds}
    return out


@pytest.fixture(scope="module")
def y_test(design):
    return design[0].test[TARGET].to_numpy(float)


# --- baselines ------------------------------------------------------------

def test_mean_baseline_predicts_a_constant(design):
    split, cols = design
    m = MeanBaseline().fit(prepare_matrix(split.train, cols), split.train[TARGET])
    p = m.predict(prepare_matrix(split.test, cols))
    assert len(set(np.round(p, 9))) == 1
    assert p[0] == pytest.approx(split.train[TARGET].mean())


def test_historical_baseline_needs_its_column(design):
    split, cols = design
    with pytest.raises(KeyError):
        HistoricalBaseline(column="not_a_column").fit(
            prepare_matrix(split.train, cols), split.train[TARGET]
        )


def test_boosting_beats_the_constant_baseline(design, fits, y_test):
    split, _ = design
    const = split.train[TARGET].mean()
    assert rmse(y_test, fits["l2"]["preds"]["test"]) < rmse(y_test, np.full_like(y_test, const))


# --- the central claim ----------------------------------------------------

def test_asymmetric_losses_shift_predictions_upward(fits, y_test):
    """Only the loss changes, and every asymmetric one moves the forecast up."""
    l2_bias = float(np.mean(fits["l2"]["preds"]["test"] - y_test))
    for key in ("asym_l2", "pinball", "linex"):
        bias = float(np.mean(fits[key]["preds"]["test"] - y_test))
        assert bias > l2_bias, f"{key} did not shift the forecast upward"


def test_asymmetric_losses_reduce_the_asymmetric_cost(fits, y_test):
    """The direct consequence of optimising the objective."""
    base = asymmetric_cost(y_test, fits["l2"]["preds"]["test"])
    for key in ("asym_l2", "pinball"):
        assert asymmetric_cost(y_test, fits[key]["preds"]["test"]) < base


def test_asymmetric_losses_reduce_the_operator_cost(fits, y_test):
    """The metric a railway would actually be handed."""
    base = expected_cost(y_test, fits["l2"]["preds"]["test"])
    for key in ("asym_l2", "pinball"):
        assert expected_cost(y_test, fits[key]["preds"]["test"]) < base


def test_asymmetric_losses_shrink_the_shortfall_on_dangerous_coaches(fits, y_test):
    base = tail_under_error(y_test, fits["l2"]["preds"]["test"])
    for key in ("asym_l2", "pinball"):
        assert tail_under_error(y_test, fits[key]["preds"]["test"]) < base


def test_asymmetric_losses_pay_for_it_in_rmse(fits, y_test):
    """Honesty check: the trade-off is real and must show up."""
    base = rmse(y_test, fits["l2"]["preds"]["test"])
    for key in ("asym_l2", "pinball", "linex"):
        assert rmse(y_test, fits[key]["preds"]["test"]) > base


def test_a_harsher_ratio_produces_a_higher_forecast(design):
    split, cols = design
    y = split.test[TARGET].to_numpy(float)
    biases = []
    for ratio in (1.0, 4.0, 20.0):
        _, preds = fit_predict(
            BoostedRegressor(loss=PinballLoss(ratio / (ratio + 1.0)), cfg=CFG), split, cols
        )
        biases.append(float(np.mean(preds["test"] - y)))
    assert biases == sorted(biases)


# --- numerical robustness -------------------------------------------------

def test_linex_does_not_diverge(fits, y_test):
    """Regression test for the unfloored-hessian blow-up."""
    p = fits["linex"]["preds"]["test"]
    assert np.isfinite(p).all()
    assert p.min() > -5.0
    assert p.max() < 60.0
    assert abs(float(np.mean(p - y_test))) < 10.0


def test_bounded_gradient_losses_actually_converge(fits):
    """Pinball must not still be climbing when boosting stops.

    With a unit pseudo-hessian it would be; the hessian_scale default exists
    precisely so early stopping, not the round cap, decides when to stop.
    """
    assert fits["pinball"]["model"].best_iteration_ < CFG.n_estimators


def test_init_score_is_added_back_on_predict(design):
    """With a custom objective LightGBM returns raw scores; forgetting the
    init score silently shifts every prediction by a constant."""
    split, cols = design
    m = BoostedRegressor(
        loss=PinballLoss(0.9), cfg=ModelConfig(n_estimators=1, val_days=VAL_DAYS, test_days=TEST_DAYS)
    )
    X = prepare_matrix(split.train, cols)
    m.fit(X, split.train[TARGET].to_numpy(float))
    assert m.init_score_ == pytest.approx(np.quantile(split.train[TARGET], 0.9), rel=1e-3)
    raw = m.booster_.predict(X, num_iteration=m.best_iteration_)
    assert np.allclose(m.predict(X) - raw, m.init_score_)


# --- the strawman and the linear model ------------------------------------

def test_margin_baseline_needs_validation_data(design):
    split, cols = design
    base = MeanBaseline().fit(prepare_matrix(split.train, cols), split.train[TARGET])
    with pytest.raises(ValueError):
        MarginBaseline(base).fit(prepare_matrix(split.train, cols), split.train[TARGET])


def test_margin_baseline_finds_a_positive_margin(design, fits):
    split, cols = design
    mb = MarginBaseline(fits["l2"]["model"])
    mb.fit(prepare_matrix(split.train, cols), split.train[TARGET],
           prepare_matrix(split.val, cols), split.val[TARGET].to_numpy(float))
    assert mb.margin_ > 0.0
    assert len(mb.tuning_curve_) > 10


def test_margin_baseline_buys_safety_at_a_worse_rmse(design, fits, y_test):
    """The strawman works, and its cost is visible: a flat margin inflates
    every prediction, including the empty 03:00 ones."""
    split, cols = design
    mb = MarginBaseline(fits["l2"]["model"])
    mb.fit(prepare_matrix(split.train, cols), split.train[TARGET],
           prepare_matrix(split.val, cols), split.val[TARGET].to_numpy(float))
    p = mb.predict(prepare_matrix(split.test, cols))
    assert expected_cost(y_test, p) < expected_cost(y_test, fits["l2"]["preds"]["test"])
    assert rmse(y_test, p) > rmse(y_test, fits["l2"]["preds"]["test"])


def test_linear_asymmetric_optimises_its_own_loss(design):
    split, cols = design
    Xtr = prepare_matrix(split.train, cols)
    ytr = split.train[TARGET].to_numpy(float)
    loss = AsymmetricSquaredError(COST_UNDER, COST_OVER)
    m = LinearAsymmetric(loss).fit(Xtr, ytr)
    fitted = loss(ytr, m.predict(Xtr))
    constant = loss(ytr, np.full_like(ytr, loss.optimal_constant(ytr)))
    assert fitted < constant
    assert len(m.coefficients()) > 5


# --- distributional -------------------------------------------------------

def test_quantile_ensemble_is_broadly_monotone(design):
    split, cols = design
    taus = (0.2, 0.5, 0.8, 0.95)
    cfg = ModelConfig(n_estimators=200, val_days=VAL_DAYS, test_days=TEST_DAYS)
    qe = QuantileEnsemble(taus=taus, cfg=cfg)
    qe.fit(prepare_matrix(split.train, cols), split.train[TARGET].to_numpy(float),
           prepare_matrix(split.val, cols), split.val[TARGET].to_numpy(float))
    Xte = prepare_matrix(split.test, cols)
    q = qe.predict_quantiles(Xte)
    assert q.shape == (len(split.test), len(taus))
    # Independently fitted quantiles can cross on individual rows, but the
    # column means must be ordered or something is badly wrong.
    assert list(q.mean(axis=0)) == sorted(q.mean(axis=0))
    assert np.allclose(qe.predict(Xte), q[:, 1])


def test_full_pipeline_regression_report(design, fits):
    split, _ = design
    rep = regression_report(split.test[TARGET], fits["pinball"]["preds"]["test"])
    assert rep["rmse"] > 0
    assert 0.0 <= rep["under_rate"] <= 1.0
    tp = ThresholdPolicy().fit(split.val[TARGET], fits["pinball"]["preds"]["val"])
    assert tp.thresholds_.shape == (3,)
