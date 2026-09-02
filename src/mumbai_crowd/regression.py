"""CO2 -- supervised regression of coach crowd density under asymmetric loss.

The estimators here all predict the same quantity (standees per m^2 in one
coach as it leaves one station) and differ only in *what they are told a
mistake costs*.  That is the experiment: holding data, features and model
family fixed, change the loss and watch the safety metrics move.

Models
------
``MeanBaseline``               global constant.  The floor.
``HistoricalBaseline``         smoothed historical mean for
                               (station, hour, direction).  What a control
                               room already does informally, and a
                               surprisingly hard baseline to beat.
``MarginBaseline``             an L2 model plus a flat safety margin, tuned on
                               validation.  This is the strawman the project
                               exists to beat: it buys recall by inflating
                               *every* prediction, including the 3 a.m. empty
                               ones, so it pays for safety with false alarms.
``LinearAsymmetric``           linear model fitted directly on an asymmetric
                               loss by L-BFGS.  Demonstrates that the loss is
                               a property of the objective, not of LightGBM.
``BoostedRegressor``           LightGBM with any loss from
                               :mod:`mumbai_crowd.losses` as a custom
                               objective.  The workhorse.

Why a custom ``init_score`` matters
-----------------------------------
With a custom objective LightGBM starts every prediction at 0 and has to
boost its way to the right level.  Under pinball loss at tau=0.857 the
optimum is a high quantile, so the first few dozen trees are spent doing
nothing but climbing.  Starting from ``loss.optimal_constant(y_train)``
removes that waste and makes the loss comparison fair, since otherwise the
more asymmetric objectives would be handicapped by having further to climb.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import ModelConfig
from .features import CATEGORICAL_FEATURES, prepare_matrix
from .losses import AsymmetricLoss, SquaredError


class Regressor:
    """Minimal common interface so the evaluation loop can stay dumb."""

    name: str = "regressor"

    def fit(self, X, y, X_val=None, y_val=None) -> "Regressor":  # pragma: no cover
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class MeanBaseline(Regressor):
    """Predict one number for every coach in the network, forever."""

    def __init__(self, loss: AsymmetricLoss | None = None):
        self.loss = loss or SquaredError()
        self.name = f"mean[{self.loss.name}]"

    def fit(self, X, y, X_val=None, y_val=None):
        self.value_ = self.loss.optimal_constant(np.asarray(y, float))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class HistoricalBaseline(Regressor):
    """Use the smoothed historical profile column as the prediction itself.

    This is what a station master's experience amounts to: "Kurla, 09:00,
    towards town, this is what it is usually like".  Any model worth deploying
    has to beat it, and beating it on RMSE is easy while beating it on
    *cost* is not.
    """

    def __init__(self, column: str = "hist_stat_hour_dire", offset: float = 0.0):
        self.column = column
        self.offset = float(offset)
        self.name = "historical_profile"

    def fit(self, X, y, X_val=None, y_val=None):
        if self.column not in X.columns:
            raise KeyError(
                f"{self.column!r} not in the design matrix; fit the "
                "HistoricalProfileEncoder first"
            )
        return self

    def predict(self, X):
        return np.asarray(X[self.column], dtype=float) + self.offset


class MarginBaseline(Regressor):
    """Symmetric model + a flat additive safety margin chosen on validation.

    The obvious thing to do instead of changing the loss, and the honest
    comparison for it.  The margin is tuned to minimise the *same* operator
    cost the asymmetric models are optimising, so it is given every chance.
    """

    def __init__(self, base: Regressor, grid: np.ndarray | None = None):
        self.base = base
        self.grid = np.arange(0.0, 8.01, 0.1) if grid is None else np.asarray(grid, float)
        self.name = f"margin[{base.name}]"

    def fit(self, X, y, X_val=None, y_val=None):
        from .metrics import expected_cost

        if X_val is None or y_val is None:
            raise ValueError("MarginBaseline needs a validation set to tune the margin")
        base_pred = self.base.predict(X_val)
        costs = [expected_cost(y_val, base_pred + m) for m in self.grid]
        self.margin_ = float(self.grid[int(np.argmin(costs))])
        self.tuning_curve_ = pd.DataFrame({"margin": self.grid, "exp_cost_inr": costs})
        return self

    def predict(self, X):
        return self.base.predict(X) + self.margin_


# ---------------------------------------------------------------------------
# Linear model on an arbitrary asymmetric loss
# ---------------------------------------------------------------------------


class LinearAsymmetric(Regressor):
    """Linear regression fitted by L-BFGS directly on an asymmetric loss.

    Included to make a point that gets lost when everything is a boosted tree:
    asymmetric loss is a statement about the *objective*, and any
    differentiable model can be trained on it.  The gradient with respect to
    the weights is just ``X^T (dL/dyhat) / n``, so the loss classes plug
    straight in.
    """

    def __init__(self, loss: AsymmetricLoss, l2: float = 1e-3, max_iter: int = 400):
        self.loss = loss
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.name = f"linear[{loss.name}]"

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        num = X.select_dtypes(include=[np.number]).copy()
        num = num.fillna(self.medians_)
        Z = (num[self.columns_].to_numpy(dtype=float) - self.mu_) / self.sd_
        return np.hstack([np.ones((len(Z), 1)), Z])

    def fit(self, X, y, X_val=None, y_val=None):
        num = X.select_dtypes(include=[np.number])
        self.columns_ = list(num.columns)
        self.medians_ = num.median()
        num = num.fillna(self.medians_)
        A = num.to_numpy(dtype=float)
        self.mu_ = A.mean(axis=0)
        self.sd_ = np.where(A.std(axis=0) > 1e-9, A.std(axis=0), 1.0)
        Z = np.hstack([np.ones((len(A), 1)), (A - self.mu_) / self.sd_])
        y = np.asarray(y, dtype=float)
        n = len(y)

        def fun(w):
            pred = Z @ w
            val = float(np.mean(self.loss.elementwise(y, pred))) + self.l2 * float(w[1:] @ w[1:])
            g_pred, _ = self.loss.grad_hess(y, pred)
            grad = Z.T @ g_pred / n
            grad[1:] += 2.0 * self.l2 * w[1:]
            return val, grad

        w0 = np.zeros(Z.shape[1])
        w0[0] = self.loss.optimal_constant(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(fun, w0, jac=True, method="L-BFGS-B",
                           options={"maxiter": self.max_iter})
        self.w_ = res.x
        self.opt_result_ = res
        return self

    def predict(self, X):
        return self._design(X) @ self.w_

    def coefficients(self) -> pd.Series:
        return pd.Series(self.w_[1:] , index=self.columns_).sort_values(key=np.abs, ascending=False)


# ---------------------------------------------------------------------------
# Gradient boosting with a pluggable objective
# ---------------------------------------------------------------------------


@dataclass
class BoostedRegressor(Regressor):
    """LightGBM trained on an arbitrary :class:`AsymmetricLoss`."""

    loss: AsymmetricLoss = field(default_factory=SquaredError)
    cfg: ModelConfig = field(default_factory=ModelConfig)
    early_stopping_rounds: int = 60
    verbose_eval: int = 0
    label: str | None = None

    def __post_init__(self) -> None:
        self.name = self.label or f"lgbm[{self.loss.name}]"

    def _params(self) -> dict:
        return {
            "objective": self.loss.lgb_objective(),
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.cfg.num_leaves,
            "min_child_samples": self.cfg.min_child_samples,
            "bagging_fraction": self.cfg.subsample,
            "bagging_freq": 1,
            "feature_fraction": self.cfg.colsample_bytree,
            "seed": self.cfg.random_state,
            "verbosity": -1,
            "num_threads": 0,
            "max_bin": 255,
        }

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y, dtype=float)
        self.init_score_ = float(self.loss.optimal_constant(y))
        cats = [c for c in CATEGORICAL_FEATURES if c in X.columns]

        dtrain = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False)
        dtrain.set_init_score(np.full(len(y), self.init_score_))

        valid_sets, valid_names, callbacks = [], [], []
        if X_val is not None and y_val is not None:
            y_val = np.asarray(y_val, dtype=float)
            dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
            dvalid.set_init_score(np.full(len(y_val), self.init_score_))
            valid_sets, valid_names = [dvalid], ["valid"]
            callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds, verbose=bool(self.verbose_eval))
            )
        if self.verbose_eval:
            callbacks.append(lgb.log_evaluation(self.verbose_eval))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.booster_ = lgb.train(
                self._params(),
                dtrain,
                num_boost_round=self.cfg.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                feval=self.loss.lgb_eval(),
                callbacks=callbacks,
            )
        self.best_iteration_ = self.booster_.best_iteration or self.cfg.n_estimators
        return self

    def predict(self, X) -> np.ndarray:
        raw = self.booster_.predict(X, num_iteration=self.best_iteration_)
        # With a custom objective LightGBM returns the boosted residual only;
        # the init score has to be added back by hand.
        return np.asarray(raw, dtype=float) + self.init_score_

    def feature_importance(self, kind: str = "gain") -> pd.Series:
        imp = self.booster_.feature_importance(importance_type=kind)
        return pd.Series(imp, index=self.booster_.feature_name()).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Distributional model
# ---------------------------------------------------------------------------


@dataclass
class QuantileEnsemble(Regressor):
    """One LightGBM per quantile, giving a predictive distribution per coach.

    A point forecast cannot express "probably CRUSH, but with a 30% chance of
    super-dense crush", and that sentence is the entire decision problem.
    Fitting the pinball loss at a grid of ``taus`` recovers the conditional
    quantile function instead, which
    :class:`mumbai_crowd.decision.DistributionalPolicy` turns into band
    probabilities and then into the Bayes-optimal action.

    Each tau is fitted independently, so the predicted quantiles can cross;
    the consuming policy re-sorts them rather than pretending it never
    happens.  ``predict`` returns the median, so the class still satisfies the
    plain :class:`Regressor` interface and can be scored alongside the others.
    """

    taus: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99)
    cfg: ModelConfig = field(default_factory=ModelConfig)
    early_stopping_rounds: int = 60
    label: str | None = None

    def __post_init__(self) -> None:
        self.name = self.label or f"quantile_ensemble[{len(self.taus)} taus]"

    def fit(self, X, y, X_val=None, y_val=None):
        from .losses import PinballLoss

        self.models_: dict[float, BoostedRegressor] = {}
        for tau in self.taus:
            m = BoostedRegressor(
                loss=PinballLoss(tau),
                cfg=self.cfg,
                early_stopping_rounds=self.early_stopping_rounds,
            )
            m.fit(X, y, X_val, y_val)
            self.models_[tau] = m
        return self

    def predict_quantiles(self, X) -> np.ndarray:
        """``(n_rows, n_taus)`` matrix of predicted quantiles."""
        return np.column_stack([self.models_[t].predict(X) for t in self.taus])

    def predict(self, X) -> np.ndarray:
        """Median forecast, so the ensemble slots into the standard leaderboard."""
        q = self.predict_quantiles(X)
        j = int(np.argmin(np.abs(np.asarray(self.taus) - 0.5)))
        return q[:, j]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def fit_predict(
    model: Regressor,
    split,
    columns: list[str],
    target: str = "density_depart",
) -> tuple[Regressor, dict[str, np.ndarray]]:
    """Fit on train (early-stopping on val) and predict every slice."""
    Xtr = prepare_matrix(split.train, columns)
    Xva = prepare_matrix(split.val, columns)
    Xte = prepare_matrix(split.test, columns)
    ytr = split.train[target].to_numpy(dtype=float)
    yva = split.val[target].to_numpy(dtype=float)

    model.fit(Xtr, ytr, Xva, yva)
    return model, {
        "train": model.predict(Xtr),
        "val": model.predict(Xva),
        "test": model.predict(Xte),
    }


__all__ = [
    "BoostedRegressor",
    "QuantileEnsemble",
    "HistoricalBaseline",
    "LinearAsymmetric",
    "MarginBaseline",
    "MeanBaseline",
    "Regressor",
    "fit_predict",
]
