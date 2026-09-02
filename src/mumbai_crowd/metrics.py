"""Evaluation metrics.

Two families, and the distinction matters:

*Statistical* metrics (RMSE, MAE, R^2) answer "how close is the number?".
They are reported for completeness, and a model trained under an asymmetric
loss will always look *worse* on them than an L2 model.  That is not a defect;
it is the deliberate price being paid.

*Decision* metrics answer "what does using this model cost?".  Expected cost
under :data:`mumbai_crowd.config.COST_MATRIX`, the dangerous-miss rate, and
the recall on super-dense-crush events are the numbers a railway would
actually be handed, and they are the ones the project is optimising.

Reporting only the first family is the most common way to make an
asymmetric-loss project look like a failure.  Reporting only the second lets
a model that predicts "DANGEROUS everywhere" look like a triumph.  Both are
always printed side by side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BAND_LABELS,
    COST_MATRIX,
    COST_OVER,
    COST_UNDER,
    DANGER_DENSITY,
    density_to_band,
)


# ---------------------------------------------------------------------------
# Statistical
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def bias(y_true, y_pred) -> float:
    """Mean signed error.  Positive = the model runs deliberately high."""
    return float(np.mean(np.asarray(y_pred, float) - np.asarray(y_true, float)))


def pinball(y_true, y_pred, tau: float) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = y_true - y_pred
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


# ---------------------------------------------------------------------------
# Asymmetry
# ---------------------------------------------------------------------------

def under_prediction_rate(y_true, y_pred, tol: float = 0.0) -> float:
    """Share of coaches the model claims are emptier than they really are."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(y_pred < y_true - tol))


def asymmetric_cost(y_true, y_pred, c_under: float = COST_UNDER, c_over: float = COST_OVER) -> float:
    """Mean piecewise-linear cost in the density units the loss is defined on."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = y_true - y_pred
    return float(np.mean(np.where(d > 0, c_under * d, -c_over * d)))


def tail_under_error(y_true, y_pred, threshold: float = DANGER_DENSITY) -> float:
    """Mean shortfall on coaches that really were at or past super-dense crush.

    This is the number that should keep an operator awake: on average, by how
    much does the model under-state the danger *when there is danger*?
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = y_true >= threshold
    if not m.any():
        return float("nan")
    return float(np.mean(np.maximum(y_true[m] - y_pred[m], 0.0)))


# ---------------------------------------------------------------------------
# Decision-level (band) metrics
# ---------------------------------------------------------------------------

def band_confusion(y_true, y_pred, normalize: str | None = None) -> pd.DataFrame:
    """4x4 confusion matrix over crowd bands, true rows x predicted columns."""
    t = density_to_band(y_true)
    p = density_to_band(y_pred)
    cm = np.zeros((4, 4), dtype=float)
    np.add.at(cm, (t, p), 1.0)
    if normalize == "true":
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    elif normalize == "all":
        cm = cm / max(cm.sum(), 1.0)
    return pd.DataFrame(cm, index=[f"true_{b}" for b in BAND_LABELS],
                        columns=[f"pred_{b}" for b in BAND_LABELS])


def expected_cost(y_true, y_pred, cost_matrix: np.ndarray = COST_MATRIX) -> float:
    """Mean rupee cost per coach-arrival under the operator's cost matrix."""
    t = density_to_band(y_true)
    p = density_to_band(y_pred)
    return float(np.mean(cost_matrix[t, p]))


def dangerous_miss_rate(y_true, y_pred) -> float:
    """P(model says COMFORTABLE or BUSY | the coach really is DANGEROUS).

    The headline safety metric: how often the system tells a station master
    that a super-dense-crush coach is fine to let more people onto.
    """
    t = density_to_band(y_true)
    p = density_to_band(y_pred)
    m = t == 3
    return float(np.mean(p[m] <= 1)) if m.any() else float("nan")


def critical_miss_rate(y_true, y_pred) -> float:
    """P(model says COMFORTABLE | the coach really is DANGEROUS).  Worst cell."""
    t = density_to_band(y_true)
    p = density_to_band(y_pred)
    m = t == 3
    return float(np.mean(p[m] == 0)) if m.any() else float("nan")


def danger_recall(y_true, y_pred) -> float:
    """P(model flags DANGEROUS | it really is DANGEROUS)."""
    t = density_to_band(y_true)
    p = density_to_band(y_pred)
    m = t == 3
    return float(np.mean(p[m] == 3)) if m.any() else float("nan")


def danger_precision(y_true, y_pred) -> float:
    """P(it really is DANGEROUS | model flags DANGEROUS)."""
    t = density_to_band(y_true)
    p = density_to_band(y_pred)
    m = p == 3
    return float(np.mean(t[m] == 3)) if m.any() else float("nan")


def false_alarm_rate(y_true, y_pred) -> float:
    """P(model escalates to CRUSH+ | the coach was really below CRUSH).

    The metric that stops "always shout DANGEROUS" from winning.
    """
    t = density_to_band(y_true)
    p = density_to_band(y_pred)
    m = t <= 1
    return float(np.mean(p[m] >= 2)) if m.any() else float("nan")


def intervention_rate(y_true, y_pred) -> float:
    """Share of coach-arrivals for which the model would trigger an action."""
    return float(np.mean(density_to_band(y_pred) >= 2))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def regression_report(y_true, y_pred, tau: float | None = None) -> dict[str, float]:
    """Every metric above in one dict, ready for a DataFrame row."""
    from .config import optimal_quantile

    tau = optimal_quantile() if tau is None else tau
    return {
        # statistical
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "pinball": pinball(y_true, y_pred, tau),
        # asymmetry
        "asym_cost": asymmetric_cost(y_true, y_pred),
        "under_rate": under_prediction_rate(y_true, y_pred),
        "tail_under_err": tail_under_error(y_true, y_pred),
        # decision
        "exp_cost_inr": expected_cost(y_true, y_pred),
        "dangerous_miss": dangerous_miss_rate(y_true, y_pred),
        "critical_miss": critical_miss_rate(y_true, y_pred),
        "danger_recall": danger_recall(y_true, y_pred),
        "danger_precision": danger_precision(y_true, y_pred),
        "false_alarm": false_alarm_rate(y_true, y_pred),
        "intervention_rate": intervention_rate(y_true, y_pred),
    }


#: Column order and formatting used whenever a leaderboard is printed.
REPORT_COLUMNS: list[str] = [
    "rmse", "mae", "r2", "bias", "pinball",
    "asym_cost", "under_rate", "tail_under_err",
    "exp_cost_inr", "dangerous_miss", "critical_miss",
    "danger_recall", "danger_precision", "false_alarm", "intervention_rate",
]

#: Human-readable descriptions, printed under every leaderboard so a reader
#: never has to guess which direction is good.
METRIC_GLOSSARY: dict[str, str] = {
    "rmse": "root mean squared error, standees/m^2 (lower better)",
    "mae": "mean absolute error, standees/m^2 (lower better)",
    "r2": "coefficient of determination (higher better)",
    "bias": "mean signed error; >0 means the model deliberately runs high",
    "pinball": "pinball loss at the cost-implied quantile (lower better)",
    "asym_cost": "mean piecewise-linear cost, c_under=6 / c_over=1 (lower better)",
    "under_rate": "share of coaches predicted emptier than reality (lower better)",
    "tail_under_err": "mean shortfall on genuinely DANGEROUS coaches (lower better)",
    "exp_cost_inr": "mean cost per coach-arrival under the operator cost matrix (lower better)",
    "dangerous_miss": "P(predict COMFORTABLE/BUSY | truly DANGEROUS) (lower better)",
    "critical_miss": "P(predict COMFORTABLE | truly DANGEROUS) (lower better)",
    "danger_recall": "P(flag DANGEROUS | truly DANGEROUS) (higher better)",
    "danger_precision": "P(truly DANGEROUS | flagged DANGEROUS) (higher better)",
    "false_alarm": "P(escalate to CRUSH+ | truly below CRUSH) (lower better)",
    "intervention_rate": "share of arrivals triggering an operational action",
}


def leaderboard(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Assemble per-model metric dicts into a sorted comparison table."""
    df = pd.DataFrame(results).T
    df = df[[c for c in REPORT_COLUMNS if c in df.columns]]
    return df.sort_values("exp_cost_inr")


__all__ = [
    "METRIC_GLOSSARY",
    "REPORT_COLUMNS",
    "asymmetric_cost",
    "band_confusion",
    "bias",
    "critical_miss_rate",
    "danger_precision",
    "danger_recall",
    "dangerous_miss_rate",
    "expected_cost",
    "false_alarm_rate",
    "intervention_rate",
    "leaderboard",
    "mae",
    "pinball",
    "r2",
    "regression_report",
    "rmse",
    "tail_under_error",
    "under_prediction_rate",
]
