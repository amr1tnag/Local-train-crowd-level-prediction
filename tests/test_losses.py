"""The losses are the core claim of the project, so they get the most tests."""
from __future__ import annotations

import numpy as np
import pytest

from mumbai_crowd.losses import (
    AsymmetricHuber,
    AsymmetricSquaredError,
    LinexLoss,
    PinballLoss,
    SquaredError,
    build_loss,
)

ALL_LOSSES = [
    SquaredError(),
    AsymmetricSquaredError(6.0, 1.0),
    PinballLoss(0.857),
    LinexLoss(0.30),
    AsymmetricHuber(2.0, 6.0, 1.0),
]


@pytest.fixture
def sample():
    rng = np.random.default_rng(11)
    y = rng.gamma(2.0, 3.0, size=4000)
    p = y + rng.normal(0.0, 2.0, size=4000)
    return y, p


@pytest.mark.parametrize("loss", ALL_LOSSES, ids=lambda l: l.name)
def test_gradient_matches_central_difference(loss, sample):
    """Analytic dL/dy_pred must match a numerical derivative."""
    y, p = sample
    eps = 1e-5
    grad, _ = loss.grad_hess(y, p)
    numeric = (loss.elementwise(y, p + eps) - loss.elementwise(y, p - eps)) / (2 * eps)
    # Skip points within eps of a kink, where the derivative is undefined.
    smooth = np.abs(p - y) > 1e-2
    assert np.max(np.abs(grad - numeric)[smooth]) < 1e-4


@pytest.mark.parametrize("loss", ALL_LOSSES, ids=lambda l: l.name)
def test_loss_is_non_negative_and_zero_at_truth(loss, sample):
    y, _ = sample
    assert np.all(loss.elementwise(y, y) < 1e-9)
    assert np.all(loss.elementwise(y, y + 1.0) >= 0)
    assert np.all(loss.elementwise(y, y - 1.0) >= 0)


@pytest.mark.parametrize("loss", ALL_LOSSES[1:], ids=lambda l: l.name)
def test_under_prediction_costs_more_than_over_prediction(loss):
    """The defining property: being short is worse than being over."""
    y = np.full(500, 8.0)
    under = loss(y, y - 2.0)
    over = loss(y, y + 2.0)
    assert under > over, f"{loss.name} is not asymmetric in the required direction"


def test_squared_error_is_symmetric():
    loss = SquaredError()
    y = np.full(100, 8.0)
    assert loss(y, y - 2.0) == pytest.approx(loss(y, y + 2.0))


@pytest.mark.parametrize("loss", ALL_LOSSES, ids=lambda l: l.name)
def test_hessian_is_strictly_positive(loss, sample):
    """A non-positive hessian makes gradient boosting produce garbage leaves."""
    y, p = sample
    _, hess = loss.grad_hess(y, p)
    assert np.all(hess > 0)


@pytest.mark.parametrize("loss", ALL_LOSSES, ids=lambda l: l.name)
def test_newton_step_is_bounded(loss, sample):
    """Regression test for the LINEX/Huber divergence: -g/h must stay sane.

    Without a hessian floor the LINEX Newton step reached ~1e5 on the
    over-prediction side and boosting never recovered.
    """
    y, p = sample
    grad, hess = loss.grad_hess(y, p)
    step = -grad / hess
    assert np.max(np.abs(step)) < 50.0


def test_optimal_constant_recovers_known_statistics():
    rng = np.random.default_rng(3)
    y = rng.gamma(2.0, 3.0, size=20000)
    assert SquaredError().optimal_constant(y) == pytest.approx(y.mean(), rel=1e-6)
    assert PinballLoss(0.85).optimal_constant(y) == pytest.approx(np.quantile(y, 0.85), rel=1e-6)
    # An asymmetric squared loss sits between the mean and the high quantile.
    c = AsymmetricSquaredError(6.0, 1.0).optimal_constant(y)
    assert y.mean() < c < np.quantile(y, 0.95)


def test_optimal_constant_actually_minimises():
    rng = np.random.default_rng(4)
    y = rng.gamma(2.0, 3.0, size=5000)
    for loss in ALL_LOSSES:
        c = loss.optimal_constant(y)
        here = loss(y, np.full_like(y, c))
        for delta in (-0.5, -0.1, 0.1, 0.5):
            assert here <= loss(y, np.full_like(y, c + delta)) + 1e-6


def test_pinball_from_costs_matches_the_decision_theory():
    """tau = c_under / (c_under + c_over) is the whole bridge; pin it down."""
    loss = PinballLoss.from_costs(cost_under=6.0, cost_over=1.0)
    assert loss.tau == pytest.approx(6.0 / 7.0)


def test_pinball_minimiser_is_the_quantile():
    rng = np.random.default_rng(5)
    y = rng.normal(10.0, 3.0, size=50000)
    for tau in (0.5, 0.8, 0.95):
        assert PinballLoss(tau).optimal_constant(y) == pytest.approx(np.quantile(y, tau), abs=0.05)


def test_asymmetry_scales_with_the_weight_ratio():
    rng = np.random.default_rng(6)
    y = rng.gamma(2.0, 3.0, size=20000)
    constants = [AsymmetricSquaredError(r, 1.0).optimal_constant(y) for r in (1, 2, 6, 20)]
    assert constants == sorted(constants), "a harsher under-prediction penalty must raise the forecast"


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        PinballLoss(0.0)
    with pytest.raises(ValueError):
        PinballLoss(1.0)
    with pytest.raises(ValueError):
        LinexLoss(-1.0)
    with pytest.raises(ValueError):
        AsymmetricSquaredError(0.0, 1.0)
    with pytest.raises(ValueError):
        AsymmetricHuber(delta=0.0)


def test_shape_mismatch_is_caught():
    with pytest.raises(ValueError):
        SquaredError().elementwise(np.zeros(5), np.zeros(6))


def test_build_loss_registry():
    assert isinstance(build_loss("pinball", tau=0.9), PinballLoss)
    assert isinstance(build_loss("l2"), SquaredError)
    with pytest.raises(KeyError):
        build_loss("nonsense")


def test_lgb_objective_uses_the_native_argument_order():
    """lgb.train calls fobj(preds, dataset); getting this backwards is silent."""

    class FakeDataset:
        def __init__(self, label):
            self._label = label

        def get_label(self):
            return self._label

    loss = AsymmetricSquaredError(6.0, 1.0)
    y = np.array([5.0, 5.0])
    p = np.array([3.0, 7.0])          # one under, one over
    grad, hess = loss.lgb_objective()(p, FakeDataset(y))
    direct_g, direct_h = loss.grad_hess(y, p)
    assert np.allclose(grad, direct_g)
    assert np.allclose(hess, direct_h)
    # The under-predicted row must carry the heavier weight.
    assert abs(grad[0]) > abs(grad[1])
