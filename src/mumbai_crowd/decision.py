"""Turning a density forecast into an operational decision.

A regression gives a number.  A station master needs an *action*: do nothing,
put an advisory on the boards, marshal the platform, or hold a relief rake.
The step between the two is where most of the remaining safety is won, and
doing it by simply comparing the point forecast to the band edges throws that
safety away.

The reason is a fact about point forecasts, not a bug in the model.  A point
forecast is a summary of ``p(y | x)``; comparing it to a threshold implicitly
assumes the whole distribution sits on one side.  For a coach whose predicted
density is 9.5 with a wide right tail, the *point* is CRUSH but the
*probability* of super-dense crush may be 30% -- and 30% of a Rs 22,000
outcome dominates the Rs 500 cost of over-reacting.  The Bayes-optimal action
is therefore

.. math::

    a^*(x) = \\arg\\min_a \\sum_b P(\\text{band}=b \\mid x)\\, C[b, a]

which is what :class:`DistributionalPolicy` computes.  :class:`ThresholdPolicy`
is the cheap, deployable approximation: keep the point forecast, but move the
alert cut-points down to wherever validation says expected cost is minimised.

Both are strictly better than comparing a point forecast to the physical band
edges, and both are honest about what they cost in false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BAND_EDGES, BAND_LABELS, COST_MATRIX, density_to_band


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------


class Policy:
    """Maps model output to one of the four operational bands."""

    name: str = "policy"

    def decide(self, pred) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class NaivePolicy(Policy):
    """Compare the point forecast to the physical band edges.

    The default anyone writes first.  Kept as the comparison point, because
    every improvement below has to be measured against it.
    """

    name = "naive_edges"

    def decide(self, pred) -> np.ndarray:
        return density_to_band(np.asarray(pred, dtype=float))


@dataclass
class ThresholdPolicy(Policy):
    """Cost-optimal alert thresholds on a point forecast.

    Three cut-points ``t1 <= t2 <= t3`` decide BUSY / CRUSH / DANGEROUS.  They
    are fitted by coordinate descent on a validation set against the operator
    cost matrix, sweeping each cut-point over a fine grid while holding the
    others fixed and repeating until nothing moves.  The objective is a step
    function of the thresholds, so gradients are useless and a direct sweep is
    both simpler and exact on the grid.

    The fitted thresholds always come out *below* the physical band edges: to
    catch a dangerous coach you have to start worrying before your point
    forecast says it is dangerous.  How far below is a quantitative statement
    about the cost ratio, and it is the most directly deployable output of
    this whole project -- it is three numbers a control room can act on.
    """

    cost_matrix: np.ndarray = field(default_factory=lambda: COST_MATRIX)
    grid_step: float = 0.05
    max_rounds: int = 8
    name: str = "cost_optimal_thresholds"

    def fit(self, y_true, pred) -> "ThresholdPolicy":
        y_true = np.asarray(y_true, dtype=float)
        pred = np.asarray(pred, dtype=float)
        true_band = density_to_band(y_true)

        lo = float(np.min(pred))
        hi = float(np.quantile(pred, 0.9995))
        grid = np.arange(lo, hi + self.grid_step, self.grid_step)
        if len(grid) < 4:
            grid = np.linspace(lo, hi if hi > lo else lo + 1.0, 50)

        thresholds = np.asarray(BAND_EDGES, dtype=float).copy()
        best = self._cost(true_band, pred, thresholds)
        self.history_ = [(thresholds.copy(), best)]

        for _ in range(self.max_rounds):
            improved = False
            for k in range(3):
                current = thresholds[k]
                candidates = grid
                # Keep the cut-points ordered.
                low = thresholds[k - 1] if k > 0 else -np.inf
                high = thresholds[k + 1] if k < 2 else np.inf
                candidates = candidates[(candidates >= low) & (candidates <= high)]
                if len(candidates) == 0:
                    continue
                costs = np.array(
                    [self._cost(true_band, pred, self._with(thresholds, k, c)) for c in candidates]
                )
                j = int(np.argmin(costs))
                if costs[j] < best - 1e-12:
                    best = float(costs[j])
                    thresholds = self._with(thresholds, k, float(candidates[j]))
                    improved = True
                else:
                    thresholds[k] = current
            self.history_.append((thresholds.copy(), best))
            if not improved:
                break

        self.thresholds_ = thresholds
        self.val_cost_ = best
        return self

    @staticmethod
    def _with(thresholds: np.ndarray, k: int, value: float) -> np.ndarray:
        out = thresholds.copy()
        out[k] = value
        return out

    def _cost(self, true_band: np.ndarray, pred: np.ndarray, thresholds: np.ndarray) -> float:
        action = np.searchsorted(thresholds, pred, side="right")
        return float(np.mean(self.cost_matrix[true_band, action]))

    def decide(self, pred) -> np.ndarray:
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("call fit() before decide()")
        return np.searchsorted(self.thresholds_, np.asarray(pred, dtype=float), side="right")

    def describe(self) -> str:
        t = self.thresholds_
        rows = [
            f"  {'action':<12} {'fitted cut':>11} {'physical edge':>14} {'shift':>8}",
        ]
        for k, label in enumerate(BAND_LABELS[1:]):
            rows.append(
                f"  {label:<12} {t[k]:>11.2f} {BAND_EDGES[k]:>14.2f} {t[k] - BAND_EDGES[k]:>8.2f}"
            )
        return "\n".join(rows)


@dataclass
class DistributionalPolicy(Policy):
    """Bayes-optimal action from a predicted distribution.

    Consumes a matrix of predicted quantiles (rows = coach-arrivals, columns =
    the ``taus`` the ensemble was trained at), converts it to band
    probabilities, and picks the action minimising expected cost.  This is the
    textbook-correct answer, and it needs no threshold tuning at all: the cost
    matrix alone determines the action once the probabilities are in hand.
    """

    taus: np.ndarray = field(
        default_factory=lambda: np.array([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.995])
    )
    cost_matrix: np.ndarray = field(default_factory=lambda: COST_MATRIX)
    name: str = "bayes_distributional"

    def band_probabilities(self, quantile_pred: np.ndarray) -> np.ndarray:
        """P(band = b | x) for each row, from the predicted quantile function."""
        q = np.asarray(quantile_pred, dtype=float)
        if q.ndim != 2 or q.shape[1] != len(self.taus):
            raise ValueError(f"expected (n, {len(self.taus)}) quantile matrix, got {q.shape}")
        # Quantile crossing is common when each tau is fitted independently;
        # enforce monotonicity before treating the row as a CDF.
        q = np.maximum.accumulate(q, axis=1)

        cdf_at_edge = np.empty((len(q), len(BAND_EDGES)))
        for j, edge in enumerate(BAND_EDGES):
            cdf_at_edge[:, j] = _interp_cdf(q, self.taus, edge)

        probs = np.empty((len(q), 4))
        probs[:, 0] = cdf_at_edge[:, 0]
        probs[:, 1] = cdf_at_edge[:, 1] - cdf_at_edge[:, 0]
        probs[:, 2] = cdf_at_edge[:, 2] - cdf_at_edge[:, 1]
        probs[:, 3] = 1.0 - cdf_at_edge[:, 2]
        probs = np.clip(probs, 0.0, 1.0)
        return probs / probs.sum(axis=1, keepdims=True)

    def decide(self, quantile_pred) -> np.ndarray:
        probs = self.band_probabilities(quantile_pred)
        expected = probs @ self.cost_matrix        # (n, 4): cost of each action
        return np.argmin(expected, axis=1)

    def expected_costs(self, quantile_pred) -> np.ndarray:
        return self.band_probabilities(quantile_pred) @ self.cost_matrix


def _interp_cdf(q: np.ndarray, taus: np.ndarray, edge: float) -> np.ndarray:
    """Estimate ``F(edge)`` per row by interpolating the inverse-CDF samples.

    Inside the fitted range this is plain linear interpolation of ``tau``
    against ``q``.  Outside it we have to extrapolate, and the two tails are
    extrapolated differently on purpose:

    * below ``q(tau_min)`` the CDF is squeezed linearly to 0, which is
      conservative in the harmless direction;
    * above ``q(tau_max)`` an exponential tail is used with a scale estimated
      from the *local* spacing of the top two fitted quantiles, so a row whose
      distribution is already wide at the top is credited with a heavier tail
      than one that is tight.

    With a finite tau grid the extreme tails are genuinely unresolved.  This
    is the honest limit of the approach and it is why the report compares the
    Bayes policy against directly-tuned thresholds rather than assuming the
    theoretically optimal one must win.
    """
    n, k = q.shape
    lo_t, hi_t = float(taus[0]), float(taus[-1])

    # j = how many fitted quantiles lie strictly below the edge.
    j = (q < edge).sum(axis=1)
    out = np.empty(n)

    below = j == 0
    above = j == k
    mid = ~(below | above)

    if below.any():
        q0 = q[below, 0]
        out[below] = lo_t * np.clip(np.divide(edge, q0, out=np.zeros_like(q0), where=q0 > 0), 0.0, 1.0)

    if above.any():
        qk = q[above, -1]
        # Local tail scale: how far apart the top two fitted quantiles are.
        scale = np.maximum(qk - q[above, -2], 0.25)
        out[above] = hi_t + (1.0 - hi_t) * (1.0 - np.exp(-(edge - qk) / scale))

    if mid.any():
        jm = j[mid]
        q_lo = np.take_along_axis(q[mid], (jm - 1)[:, None], axis=1).ravel()
        q_hi = np.take_along_axis(q[mid], jm[:, None], axis=1).ravel()
        t_lo, t_hi = taus[jm - 1], taus[jm]
        width = np.maximum(q_hi - q_lo, 1e-9)
        out[mid] = t_lo + (t_hi - t_lo) * (edge - q_lo) / width

    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def policy_report(y_true, action: np.ndarray, cost_matrix: np.ndarray = COST_MATRIX) -> dict:
    """Decision-level metrics for an explicit action vector."""
    true_band = density_to_band(np.asarray(y_true, dtype=float))
    action = np.asarray(action, dtype=int)
    danger = true_band == 3
    safe = true_band <= 1
    flagged = action == 3
    return {
        "exp_cost_inr": float(np.mean(cost_matrix[true_band, action])),
        "dangerous_miss": float(np.mean(action[danger] <= 1)) if danger.any() else float("nan"),
        "critical_miss": float(np.mean(action[danger] == 0)) if danger.any() else float("nan"),
        "danger_recall": float(np.mean(action[danger] == 3)) if danger.any() else float("nan"),
        "danger_precision": float(np.mean(true_band[flagged] == 3)) if flagged.any() else float("nan"),
        "false_alarm": float(np.mean(action[safe] >= 2)) if safe.any() else float("nan"),
        "intervention_rate": float(np.mean(action >= 2)),
        "relief_rake_rate": float(np.mean(action == 3)),
    }


def action_confusion(y_true, action: np.ndarray, normalize: str | None = "true") -> pd.DataFrame:
    true_band = density_to_band(np.asarray(y_true, dtype=float))
    cm = np.zeros((4, 4))
    np.add.at(cm, (true_band, np.asarray(action, dtype=int)), 1.0)
    if normalize == "true":
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    return pd.DataFrame(
        cm,
        index=[f"true_{b}" for b in BAND_LABELS],
        columns=[f"act_{b}" for b in BAND_LABELS],
    )


__all__ = [
    "DistributionalPolicy",
    "NaivePolicy",
    "Policy",
    "ThresholdPolicy",
    "action_confusion",
    "policy_report",
]
