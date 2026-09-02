"""A calibrated probability model for the crowd bands.

Why this module exists
----------------------
The regression side of the project (CO2) produces a point forecast, and
:class:`mumbai_crowd.decision.DistributionalPolicy` turns a *quantile ensemble*
into band probabilities so that the Bayes-optimal action can be taken.  On this
problem that route measurably fails, and the reliability diagram in the report
says exactly why: the quantile ensemble is well calibrated *marginally* -- its
empirical coverage tracks nominal tau to within 0.03 -- but it under-states
``P(DANGEROUS | x)`` by up to 2.6x in the range where the decision actually
turns, because a nine-point quantile grid cannot resolve a 1.1% conditional
tail.  The Bayes rule then declines to act, and a crudely tuned threshold on a
point forecast beats the theoretically optimal policy.

The fix implied by that diagnosis is not more quantiles.  It is to model the
thing the decision needs -- the band probabilities -- *directly*, and then to
recalibrate them before handing them to the cost matrix.  That is what this
module does, and whether it actually closes the gap is an empirical question
the pipeline answers rather than an assumption.

Design notes
------------
**Class weighting and calibration are deliberately separated.**  Up-weighting
the 1.1% DANGEROUS class helps a boosted ensemble spend capacity on the tail,
but it also destroys calibration by construction: the model is now fitting a
re-weighted distribution, not the real one.  Under a cost matrix that is fine
*provided* you put the calibration back, so the two steps are separate objects
here and the pipeline reports both with and without.

**The calibration set is not the early-stopping set.**  Fitting isotonic
regression on the same rows that chose the iteration count would calibrate
against a slice the model has already been tuned on.  The validation window is
therefore split in half *by date*, the earlier half choosing the number of
trees and the later half fitting the calibrator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import BAND_LABELS, ModelConfig, density_to_band
from .features import CATEGORICAL_FEATURES

N_BANDS: int = len(BAND_LABELS)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class IsotonicCalibrator:
    """Per-class isotonic recalibration of a probability vector.

    One isotonic regression per band maps the raw score for that band onto the
    frequency actually observed, after which the four are renormalised to sum
    to one.  Isotonic is the right family here because the failure being
    corrected is *monotone* -- the model's ordering of rows by danger is
    informative, its absolute scale is not -- and because with tens of
    thousands of calibration rows a non-parametric fit is affordable, where
    Platt scaling would impose a sigmoid the data has no reason to follow.

    Renormalising after per-class fitting is a mild approximation: four
    independent isotonic maps need not produce a coherent joint distribution.
    The alternative (a full multinomial recalibration) needs far more data than
    a 1.1% class provides, and the renormalised version is what practitioners
    use for exactly this reason.
    """

    clip: float = 1e-6

    def fit(self, probs: np.ndarray, y_band: np.ndarray) -> "IsotonicCalibrator":
        probs = np.asarray(probs, dtype=float)
        y_band = np.asarray(y_band, dtype=int)
        if probs.ndim != 2 or probs.shape[1] != N_BANDS:
            raise ValueError(f"expected an (n, {N_BANDS}) probability matrix, got {probs.shape}")

        self.models_: list[IsotonicRegression | None] = []
        for k in range(N_BANDS):
            target = (y_band == k).astype(float)
            if target.sum() < 5 or target.sum() == len(target):
                # Too few (or too many) positives to calibrate against; leave
                # this class alone rather than fitting noise.
                self.models_.append(None)
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(probs[:, k], target)
            self.models_.append(iso)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise RuntimeError("call fit() before transform()")
        probs = np.asarray(probs, dtype=float)
        out = np.empty_like(probs)
        for k, iso in enumerate(self.models_):
            out[:, k] = probs[:, k] if iso is None else iso.predict(probs[:, k])
        out = np.clip(out, self.clip, None)
        return out / out.sum(axis=1, keepdims=True)


@dataclass
class TemperatureCalibrator:
    """Single-parameter (temperature) rescaling of the probability vector.

    ``softmax(log p / T)``: one degree of freedom for the whole model.  It can
    make a model uniformly more or less confident and nothing else.

    That rigidity is the point.  :class:`IsotonicCalibrator` has enough freedom
    to reproduce whatever probability-to-frequency relationship held on the
    calibration window, *including the part of it that was just that window's
    base rate*.  When the deployment period has a different base rate -- which
    on this problem it does, monsoon to post-monsoon -- the flexible calibrator
    faithfully transfers a correction that no longer applies.  A one-parameter
    map has far less to transfer wrongly, so comparing the two is a direct
    measurement of how much of a calibrator's fit is signal and how much is the
    calibration window's prior.
    """

    max_temperature: float = 10.0

    def fit(self, probs: np.ndarray, y_band: np.ndarray) -> "TemperatureCalibrator":
        from scipy.optimize import minimize_scalar

        probs = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
        y_band = np.asarray(y_band, dtype=int)
        logp = np.log(probs)

        def nll(log_t: float) -> float:
            t = float(np.exp(log_t))
            z = logp / t
            z -= z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(p[np.arange(len(y_band)), y_band] + 1e-12)))

        bound = float(np.log(self.max_temperature))
        res = minimize_scalar(nll, bounds=(-bound, bound), method="bounded",
                              options={"xatol": 1e-4})
        self.temperature_ = float(np.exp(res.x))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not hasattr(self, "temperature_"):
            raise RuntimeError("call fit() before transform()")
        logp = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, None))
        z = logp / self.temperature_
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class IdentityCalibrator:
    """No recalibration at all -- the control condition.

    Worth having as an explicit object rather than an ``if`` branch, because on
    this problem it is the one that wins, and a comparison that cannot express
    "do nothing" cannot discover that.
    """

    def fit(self, probs: np.ndarray, y_band: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return np.asarray(probs, dtype=float)


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


@dataclass
class BandClassifier:
    """LightGBM multiclass model over the four crowd bands.

    ``class_weight="balanced"`` gives every band equal total weight, which on
    an 86/10/3/1 split means each DANGEROUS coach counts for roughly 77
    comfortable ones.  That buys the tail a lot of model capacity and wrecks
    the probability scale; :class:`IsotonicCalibrator` is what puts the scale
    back, and :class:`CalibratedBandClassifier` wires the two together.
    """

    cfg: ModelConfig = field(default_factory=ModelConfig)
    class_weight: str | None = None       # None | "balanced"
    early_stopping_rounds: int = 60
    label: str | None = None

    def __post_init__(self) -> None:
        suffix = "balanced" if self.class_weight == "balanced" else "unweighted"
        self.name = self.label or f"lgbm_classifier[{suffix}]"

    def _sample_weight(self, y_band: np.ndarray) -> np.ndarray | None:
        if self.class_weight is None:
            return None
        if self.class_weight != "balanced":
            raise ValueError(f"unsupported class_weight {self.class_weight!r}")
        counts = np.bincount(y_band, minlength=N_BANDS).astype(float)
        counts[counts == 0] = 1.0
        w = len(y_band) / (N_BANDS * counts)
        return w[y_band]

    def fit(self, X, y_density, X_val=None, y_val_density=None) -> "BandClassifier":
        y = density_to_band(np.asarray(y_density, dtype=float))
        cats = [c for c in CATEGORICAL_FEATURES if c in X.columns]
        dtrain = lgb.Dataset(
            X, label=y, weight=self._sample_weight(y),
            categorical_feature=cats, free_raw_data=False,
        )

        valid_sets, valid_names, callbacks = [], [], []
        if X_val is not None and y_val_density is not None:
            y_v = density_to_band(np.asarray(y_val_density, dtype=float))
            dvalid = lgb.Dataset(
                X_val, label=y_v, weight=self._sample_weight(y_v),
                reference=dtrain, free_raw_data=False,
            )
            valid_sets, valid_names = [dvalid], ["valid"]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        params = {
            "objective": "multiclass",
            "num_class": N_BANDS,
            "metric": "multi_logloss",
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.cfg.num_leaves,
            "min_child_samples": self.cfg.min_child_samples,
            "bagging_fraction": self.cfg.subsample,
            "bagging_freq": 1,
            "feature_fraction": self.cfg.colsample_bytree,
            "seed": self.cfg.random_state,
            "verbosity": -1,
            "num_threads": 0,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.booster_ = lgb.train(
                params, dtrain,
                num_boost_round=self.cfg.n_estimators,
                valid_sets=valid_sets, valid_names=valid_names,
                callbacks=callbacks,
            )
        self.best_iteration_ = self.booster_.best_iteration or self.cfg.n_estimators
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.asarray(
            self.booster_.predict(X, num_iteration=self.best_iteration_), dtype=float
        )
        return p.reshape(-1, N_BANDS)

    def feature_importance(self, kind: str = "gain") -> pd.Series:
        imp = self.booster_.feature_importance(importance_type=kind)
        return pd.Series(imp, index=self.booster_.feature_name()).sort_values(ascending=False)


@dataclass
class CalibratedBandClassifier:
    """A :class:`BandClassifier` plus an :class:`IsotonicCalibrator`.

    ``fit`` expects the validation window already split in two: ``*_es`` chooses
    the iteration count, ``*_cal`` fits the calibrator.  Keeping the split an
    explicit argument rather than doing it internally means the caller can see
    -- and a reviewer can check -- that the two sets are disjoint and ordered
    in time.
    """

    classifier: BandClassifier
    calibrator: object = field(default_factory=IsotonicCalibrator)
    label: str | None = None

    def __post_init__(self) -> None:
        kind = type(self.calibrator).__name__.replace("Calibrator", "").lower()
        self.name = self.label or f"{self.classifier.name} + {kind} calibration"

    def fit(self, X, y_density, X_es, y_es_density, X_cal, y_cal_density) -> "CalibratedBandClassifier":
        self.classifier.fit(X, y_density, X_es, y_es_density)
        raw = self.classifier.predict_proba(X_cal)
        self.calibrator.fit(raw, density_to_band(np.asarray(y_cal_density, dtype=float)))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.calibrator.transform(self.classifier.predict_proba(X))

    def predict_proba_uncalibrated(self, X) -> np.ndarray:
        return self.classifier.predict_proba(X)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_validation_by_date(val: pd.DataFrame, date_column: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a validation frame in half by calendar date.

    Earlier half -> early stopping, later half -> calibration.  Splitting by
    date rather than by row keeps whole operating days on one side, which
    matters because services on the same day share weather, festival state and
    disruption history.
    """
    dates = np.sort(val[date_column].unique())
    if len(dates) < 2:
        raise ValueError("need at least two distinct dates to split the validation window")
    cut = dates[len(dates) // 2]
    return val[val[date_column] < cut].copy(), val[val[date_column] >= cut].copy()


def expected_class_probabilities(probs: np.ndarray) -> pd.Series:
    """Mean predicted probability per band -- a quick check against the base rate."""
    probs = np.asarray(probs, dtype=float)
    return pd.Series(probs.mean(axis=0), index=BAND_LABELS)


__all__ = [
    "N_BANDS",
    "BandClassifier",
    "CalibratedBandClassifier",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "TemperatureCalibrator",
    "expected_class_probabilities",
    "split_validation_by_date",
]
