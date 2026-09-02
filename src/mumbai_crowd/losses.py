"""Asymmetric loss functions -- the heart of the project.

Why symmetric loss is the wrong objective here
----------------------------------------------
Squared error says a coach that is 3 standees/m^2 *emptier* than predicted is
exactly as bad as one that is 3 standees/m^2 *fuller*.  On the Harbour Line
those two errors are not remotely comparable:

* **Over-prediction** ("we said CRUSH, it was BUSY").  A relief rake is held,
  four RPF constables walk to the foot-over-bridge, a PIS board tells people
  to wait for the next train.  Cost: some money and a little credibility.
* **Under-prediction** ("we said COMFORTABLE, it was super-dense crush").
  Nobody is sent.  The footboard fills.  The Mumbai suburban network records
  roughly 2,000 deaths a year, a large share of them falls from moving trains
  and platform-gap incidents in exactly this regime.  Cost: unbounded.

So the estimator we want is not ``argmin E[(y - yhat)^2]``.  It is
``argmin E[C(y, yhat)]`` for a cost ``C`` that is steeper on the
under-prediction side.  Everything in this module is a way of writing such a
``C`` down so that a gradient-boosted tree ensemble can optimise it directly,
rather than fitting the conditional mean and then patching the output with a
fudge factor.

Three families are provided
---------------------------
``AsymmetricSquaredError``  weighted squared error, weight ``w_under`` below
                            the truth and ``w_over`` above it.  Smooth, easy
                            to reason about, and its minimiser is a weighted
                            mean -- a *shifted* central tendency.
``PinballLoss``             piecewise-linear, slope ``tau`` below and
                            ``1-tau`` above.  Its minimiser is the
                            ``tau``-quantile, which is *exactly* the
                            cost-optimal point forecast under a piecewise
                            linear cost -- the principled choice, with
                            ``tau = c_under / (c_under + c_over)``.
``LinexLoss``               linear-exponential: roughly linear for
                            over-prediction, exponential for under-prediction.
                            The most aggressive of the three; use when the
                            tail genuinely is catastrophic.

Each loss exposes ``grad_hess`` so it can be handed straight to LightGBM as a
custom objective, and ``optimal_constant`` so training can start from the
loss-minimising constant instead of zero (LightGBM's custom-objective path
does not do this for you, and starting from zero on an asymmetric loss wastes
a large number of boosting rounds).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize_scalar

ArrayLike = np.ndarray


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


class AsymmetricLoss(ABC):
    """Base class: elementwise loss plus first and second derivatives.

    Sign convention used everywhere in this module::

        residual r = y_pred - y_true
        r < 0  ->  UNDER-prediction  ->  the dangerous direction
        r > 0  ->  OVER-prediction   ->  the merely expensive direction

    Derivatives are taken with respect to ``y_pred``, which is what a
    gradient-boosting library needs.
    """

    name: str = "loss"

    @abstractmethod
    def elementwise(self, y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
        """Per-sample loss."""

    @abstractmethod
    def grad_hess(self, y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """(dL/dy_pred, d2L/dy_pred2), elementwise."""

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        return float(np.mean(self.elementwise(y_true, y_pred)))

    # -- integration helpers ------------------------------------------------

    def lgb_objective(self):
        """Return a callable usable as LightGBM's ``objective=`` parameter.

        The native ``lgb.train`` API calls ``fobj(current_scores, dataset)``
        and expects ``(grad, hess)``.  Note the argument order is the reverse
        of the sklearn wrapper's, which is a classic source of silently wrong
        gradients -- hence :meth:`lgb_objective_sklearn` as a separate method
        rather than one function that tries to guess.
        """

        def _obj(y_pred, dataset):
            return self.grad_hess(dataset.get_label(), y_pred)

        return _obj

    def lgb_objective_sklearn(self):
        """Objective for ``LGBMRegressor(objective=...)``: ``f(y_true, y_pred)``."""

        def _obj(y_true, y_pred):
            return self.grad_hess(y_true, y_pred)

        return _obj

    def lgb_eval(self):
        """Return a callable usable as ``lgb.train(feval=...)``."""

        def _eval(y_pred, dataset):
            return self.name, self(dataset.get_label(), y_pred), False  # lower is better

        return _eval

    def optimal_constant(self, y_true: ArrayLike) -> float:
        """The single number that minimises this loss on ``y_true``.

        Used as the boosting ``init_score``.  For squared error this is the
        mean; for pinball it is the ``tau``-quantile; for the others there is
        no closed form, so we solve it numerically once.
        """
        y = np.asarray(y_true, dtype=float).ravel()
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi - lo < 1e-12:
            return lo
        span = hi - lo
        res = minimize_scalar(
            lambda c: self(y, np.full_like(y, c)),
            bounds=(lo - 0.05 * span, hi + 0.05 * span),
            method="bounded",
            options={"xatol": 1e-4},
        )
        return float(res.x)


# ---------------------------------------------------------------------------


class SquaredError(AsymmetricLoss):
    """Plain L2.  The symmetric baseline every other loss is compared against."""

    name = "l2"

    def elementwise(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        return (y_pred - y_true) ** 2

    def grad_hess(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        r = y_pred - y_true
        return 2.0 * r, np.full_like(r, 2.0)

    def optimal_constant(self, y_true):
        return float(np.mean(np.asarray(y_true, dtype=float)))


class AsymmetricSquaredError(AsymmetricLoss):
    r"""Weighted squared error.

    .. math::

        L(y, \hat y) = \begin{cases}
            w_\text{under}\,(y - \hat y)^2 & \hat y < y \quad\text{(dangerous)}\\
            w_\text{over}\,(y - \hat y)^2  & \hat y \ge y
        \end{cases}

    With ``w_under = 6`` and ``w_over = 1`` the model is told that being three
    standees/m^2 short is as bad as being seven-and-a-half over.  The
    minimiser is a weighted mean that sits above the conditional mean -- a
    deliberate, quantified safety margin rather than an arbitrary one.
    """

    def __init__(self, w_under: float = 6.0, w_over: float = 1.0):
        if w_under <= 0 or w_over <= 0:
            raise ValueError("weights must be positive")
        self.w_under = float(w_under)
        self.w_over = float(w_over)
        self.name = f"asym_l2(u={w_under:g},o={w_over:g})"

    def _weights(self, y_true, y_pred):
        return np.where(y_pred < y_true, self.w_under, self.w_over)

    def elementwise(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        return self._weights(y_true, y_pred) * (y_pred - y_true) ** 2

    def grad_hess(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        w = self._weights(y_true, y_pred)
        r = y_pred - y_true
        return 2.0 * w * r, 2.0 * w

    def optimal_constant(self, y_true):
        # Closed form: the weighted mean fixed point. Solve by bisection on
        # the monotone gradient rather than a generic optimiser.
        y = np.sort(np.asarray(y_true, dtype=float).ravel())
        lo, hi = y[0], y[-1]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            g = float(np.mean(self.grad_hess(y, np.full_like(y, mid))[0]))
            if g > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)


class PinballLoss(AsymmetricLoss):
    r"""Quantile (pinball) loss -- the decision-theoretically correct one.

    .. math::

        L(y, \hat y) = \begin{cases}
            \tau\,(y - \hat y)      & \hat y < y\\
            (1-\tau)\,(\hat y - y)  & \hat y \ge y
        \end{cases}

    Minimised in expectation by the ``tau``-quantile of ``p(y | x)``.  If the
    real cost of being one standee/m^2 short is ``c_under`` and of being one
    over is ``c_over``, then the cost-minimising forecast *is* the
    ``tau``-quantile with ``tau = c_under / (c_under + c_over)``; see
    :func:`from_costs`.  That identity is why this loss is the project's
    headline model rather than a heuristic.

    The loss is piecewise linear, so its true second derivative is 0 almost
    everywhere.  Gradient boosting needs a positive hessian to form leaf
    values, so we supply a constant pseudo-hessian, exactly as LightGBM's own
    ``objective="quantile"`` does internally.

    Why ``hessian_scale`` exists
    ----------------------------
    The gradient of this loss is *bounded*: it is exactly ``-tau`` or
    ``1-tau``, never larger, however wrong the prediction is.  With a
    pseudo-hessian of 1 the Newton step for a leaf is therefore at most 1, and
    after the learning rate at most ``lr``.  To travel ten standees/m^2 at
    ``lr = 0.05`` the ensemble needs at least two hundred trees doing nothing
    but climbing -- which is why a pinball model that looks hopeless at 100
    rounds becomes the best model in the zoo at 500.  Squared error does not
    have this problem because its gradient grows with the residual.

    Shrinking the pseudo-hessian enlarges the step proportionally.  At the
    default of 0.25 the model reaches the same solution in roughly a quarter
    of the boosting rounds, with no measurable difference in the fit; it is
    the same device as raising the learning rate for this loss alone, but it
    keeps one learning rate across the whole comparison so that the loss stays
    the only thing that varies between models.
    """

    def __init__(self, tau: float = 0.85, hessian_scale: float = 0.25):
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must lie strictly between 0 and 1")
        if hessian_scale <= 0:
            raise ValueError("hessian_scale must be positive")
        self.tau = float(tau)
        self.hessian_scale = float(hessian_scale)
        self.name = f"pinball(tau={tau:g})"

    @classmethod
    def from_costs(cls, cost_under: float, cost_over: float, **kwargs) -> "PinballLoss":
        return cls(tau=cost_under / (cost_under + cost_over), **kwargs)

    def elementwise(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        d = y_true - y_pred
        return np.maximum(self.tau * d, (self.tau - 1.0) * d)

    def grad_hess(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        under = y_pred < y_true
        grad = np.where(under, -self.tau, 1.0 - self.tau)
        hess = np.full_like(grad, self.hessian_scale)
        return grad, hess

    def optimal_constant(self, y_true):
        return float(np.quantile(np.asarray(y_true, dtype=float), self.tau))


class LinexLoss(AsymmetricLoss):
    r"""Linear-exponential loss (Varian, 1975).

    .. math::

        L(y, \hat y) = e^{a(y - \hat y)} - a(y - \hat y) - 1,\qquad a > 0

    Approximately quadratic near zero, approximately *linear* for
    over-prediction and *exponential* for under-prediction.  It is the right
    shape when the bad tail is not merely expensive but catastrophic, which is
    arguably the case at super-dense crush.

    Two numerical guards, both necessary, both stated rather than hidden:

    * **exponent clipping.**  ``e^{a e}`` overflows long before the residuals
      do anything interesting, so the exponent is clipped at
      :attr:`MAX_EXPONENT`.
    * **a hessian floor.**  This is the one that actually bites.  On the
      *over*-prediction side the true hessian decays like ``e^{-a|e|}``
      towards zero, and a Newton step ``-g/h`` with a vanishing ``h`` is
      unbounded: with no floor, boosting takes one leaf step of order 10^5,
      the predictions leave the real line's useful part, and the model never
      recovers.  Flooring bounds the step at ``a / hess_floor`` and costs
      nothing statistically, because the region it touches is the one this
      loss barely cares about anyway.  The default of 0.25 bounds the step at
      ``1.2`` -- safe, and loose enough that boosting converges in a few
      hundred rounds rather than stalling the way a floor of 1.0 does.
    """

    #: Exponent clip.  ``exp(6) ~ 400`` is already more asymmetry than any
    #: real cost structure justifies; past it the gradient carries no
    #: information beyond "this is very wrong".
    MAX_EXPONENT: float = 6.0

    def __init__(self, a: float = 0.30, hess_floor: float = 0.25):
        if a <= 0:
            raise ValueError("a must be positive so that under-prediction is the costly side")
        if hess_floor <= 0:
            raise ValueError("hess_floor must be positive; see the class docstring")
        self.a = float(a)
        self.hess_floor = float(hess_floor)
        self.name = f"linex(a={a:g})"

    def _z(self, y_true, y_pred):
        e = y_true - y_pred                      # >0 means under-prediction
        return np.clip(self.a * e, -self.MAX_EXPONENT, self.MAX_EXPONENT), e

    def elementwise(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        z, e = self._z(y_true, y_pred)
        return np.exp(z) - self.a * e - 1.0

    def grad_hess(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        z, _ = self._z(y_true, y_pred)
        ez = np.exp(z)
        grad = self.a * (1.0 - ez)
        hess = self.a * self.a * ez
        return grad, np.maximum(hess, self.hess_floor)


class AsymmetricHuber(AsymmetricLoss):
    r"""Huberised asymmetric loss: quadratic core, linear tails, unequal slopes.

    Useful when the target has genuine outliers (a jammed door sensor, a
    mis-counted CCTV frame) and you do not want a handful of them to dominate
    the fit the way :class:`AsymmetricSquaredError` would, but you still want
    the safety asymmetry.
    """

    def __init__(self, delta: float = 2.0, w_under: float = 6.0, w_over: float = 1.0):
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)
        self.w_under = float(w_under)
        self.w_over = float(w_over)
        self.name = f"asym_huber(d={delta:g},u={w_under:g},o={w_over:g})"

    def _weights(self, y_true, y_pred):
        return np.where(y_pred < y_true, self.w_under, self.w_over)

    def elementwise(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        w = self._weights(y_true, y_pred)
        r = np.abs(y_pred - y_true)
        small = r <= self.delta
        return w * np.where(small, 0.5 * r**2, self.delta * (r - 0.5 * self.delta))

    def grad_hess(self, y_true, y_pred):
        y_true, y_pred = _as_arrays(y_true, y_pred)
        w = self._weights(y_true, y_pred)
        r = y_pred - y_true
        small = np.abs(r) <= self.delta
        grad = w * np.where(small, r, self.delta * np.sign(r))
        # Outside the quadratic core the true second derivative is zero.  The
        # usual boosting surrogate is delta/|r|, which keeps the Newton step
        # equal to the residual and stays bounded; a tiny constant floor here
        # would produce the same runaway leaf values as an unfloored LINEX.
        denom = np.maximum(np.abs(r), self.delta)
        hess = w * np.where(small, 1.0, np.maximum(self.delta / denom, 0.1))
        return grad, hess


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_loss(kind: str, **kwargs) -> AsymmetricLoss:
    """Factory used by the training scripts and the CLI."""
    kind = kind.lower()
    table = {
        "l2": SquaredError,
        "squared": SquaredError,
        "asym_l2": AsymmetricSquaredError,
        "pinball": PinballLoss,
        "quantile": PinballLoss,
        "linex": LinexLoss,
        "asym_huber": AsymmetricHuber,
    }
    if kind not in table:
        raise KeyError(f"unknown loss {kind!r}; choose from {sorted(table)}")
    return table[kind](**kwargs)


__all__ = [
    "AsymmetricHuber",
    "AsymmetricLoss",
    "AsymmetricSquaredError",
    "LinexLoss",
    "PinballLoss",
    "SquaredError",
    "build_loss",
]
