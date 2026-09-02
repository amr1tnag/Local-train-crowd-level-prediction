"""Feature engineering, with leakage handled explicitly rather than implicitly.

The single easiest way to produce a beautiful, worthless R^2 on this problem is
to feed the model the boardings and alightings recorded *at the same stop of
the same train* whose density you are predicting.  Those quantities are
measured at the moment the doors close -- the same moment as the target -- so a
model using them is not forecasting anything, it is reading the answer.

This module therefore defines two explicit feature sets and refuses to mix
them:

``schedule``  What a control room knows the evening before: the timetable, the
              calendar, the weather forecast, the station, the coach, and
              historical averages learned from past data.  This is the set
              that supports *planning* -- deciding tonight where to position
              tomorrow evening's relief rakes.
``realtime``  Everything above, plus what the network itself reports minutes
              before the event: how late this service is running, the actual
              gap since the previous train, and the density of this same coach
              one stop back.  This is the set that supports *intervention* --
              telling the station master at Kurla what is about to pull in.

Both are legitimate; they answer different operational questions and are
reported separately.  Anything measured simultaneously with the target lives
in :data:`LEAKY_COLUMNS` and never enters either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SimConfig
from .demand import day_type_of
from .network import ROUTE_SHARE

#: The target.
TARGET: str = "density_depart"

#: Columns that are contemporaneous with (or downstream of) the target.  Using
#: any of them as a predictor is leakage, full stop.
LEAKY_COLUMNS: list[str] = [
    "density_true", "density_depart", "density_arrive",
    "load_arrive", "load_depart",
    "boardings", "alightings", "left_behind",
    "crowd_band", "is_dangerous",
]

#: Identifier / bookkeeping columns: carried through evaluation, never fitted.
ID_COLUMNS: list[str] = ["date", "timestamp", "service_id", "station_code", "coach_position"]

CATEGORICAL_FEATURES: list[str] = [
    "station_code", "coach_class", "direction", "route", "fob_position",
]


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def _planned_headway(df: pd.DataFrame, cfg: SimConfig) -> np.ndarray:
    """Timetabled headway, reconstructed from the published service pattern.

    Unlike the *observed* gap since the previous train, this is knowable the
    night before, so it belongs in the schedule-only feature set.
    """
    from .simulate import _headway_minutes

    day_type = np.where(
        (df["is_holiday"].to_numpy() == 1) | (df["dow"].to_numpy() == 6), "sunday",
        np.where(df["dow"].to_numpy() == 5, "saturday", "weekday"),
    )
    hours = df["sched_min"].to_numpy() / 60.0
    out = np.empty(len(df))
    routes = df["route"].to_numpy()
    for i in range(len(df)):
        out[i] = _headway_minutes(hours[i] % 24, day_type[i], routes[i], cfg)
    return out


def add_derived_columns(df: pd.DataFrame, cfg: SimConfig | None = None) -> pd.DataFrame:
    """Add cyclical time encodings, peak flags and coach descriptors."""
    cfg = cfg or SimConfig()
    df = df.copy()

    minute_of_day = df["sched_min"] % 1440.0
    hour_f = minute_of_day / 60.0
    df["minute_of_day"] = minute_of_day
    df["hour_frac"] = hour_f

    # Cyclical encodings: 23:50 and 00:10 are twenty minutes apart, and a tree
    # split on raw "hour" cannot see that.
    df["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    df["is_am_peak"] = ((hour_f >= 7.25) & (hour_f < 11.0)).astype(int)
    df["is_pm_peak"] = ((hour_f >= 16.75) & (hour_f < 21.5)).astype(int)
    df["is_peak"] = (df["is_am_peak"] | df["is_pm_peak"]).astype(int)

    # The tidal interaction that defines Mumbai commuting: an AM train heading
    # for CSMT and a PM train heading away from it are the same phenomenon
    # mirrored, and a model that cannot form this product has to spend depth
    # rediscovering it.
    df["peak_toward_cbd"] = (df["is_am_peak"] & (df["direction"] == "UP")).astype(int)
    df["peak_away_cbd"] = (df["is_pm_peak"] & (df["direction"] == "DN")).astype(int)
    df["tidal_alignment"] = df["peak_toward_cbd"] + df["peak_away_cbd"]

    df["is_ladies_coach"] = df["coach_class"].isin(["ladies", "ladies_first"]).astype(int)
    df["is_first_class"] = df["coach_class"].isin(["first", "ladies_first"]).astype(int)
    df["coach_frac"] = df["coach_position"] / df["n_cars"]
    df["dist_from_coach_mid"] = (df["coach_frac"] - 0.5).abs()

    df["route_share"] = df["route"].map(ROUTE_SHARE).astype(float)
    df["planned_headway_min"] = _planned_headway(df, cfg)

    # Rain interactions.  Rain matters most in the peak, when there is no
    # spare capacity to absorb the extra riders it pushes onto the network.
    df["rain_x_peak"] = df["rain_mm_hr"] * df["is_peak"]
    df["rain_x_tidal"] = df["rain_mm_hr"] * df["tidal_alignment"]
    df["log_rain"] = np.log1p(df["rain_mm_hr"])
    df["log_rain_3h"] = np.log1p(df["rain_3h_mm"])
    df["festival_x_evening"] = df["festival_intensity"] * ((hour_f >= 18) & (hour_f < 24)).astype(int)
    return df


# ---------------------------------------------------------------------------
# Historical profile encoding (fit on train only)
# ---------------------------------------------------------------------------

@dataclass
class HistoricalProfileEncoder:
    """Smoothed historical mean density for coarse context keys.

    This is target encoding, so it is the second-easiest way to leak: fit it on
    the full dataset and every model looks superb.  It is therefore fitted
    **only on the training slice** and merely applied to validation and test,
    with an empirical-Bayes shrink towards the global mean so that thin cells
    (Manasarovar at 05:00 on a Sunday) do not contribute noise.
    """

    smoothing: float = 60.0
    keys: tuple[tuple[str, ...], ...] = (
        ("station_code", "hour", "direction"),
        ("station_code", "hour", "direction", "is_weekend"),
        ("station_code", "coach_position", "direction"),
        ("route", "hour", "direction"),
    )
    #: Populated by fit(): encoded-column name -> the key columns it groups on.
    key_of_: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)

    def fit(self, df: pd.DataFrame, target: str = TARGET) -> "HistoricalProfileEncoder":
        self.global_mean_ = float(df[target].mean())
        self.tables_: dict[str, pd.DataFrame] = {}
        for key in self.keys:
            name = "hist_" + "_".join(k[:4] for k in key)
            g = df.groupby(list(key), observed=True)[target].agg(["mean", "count"])
            shrunk = (g["mean"] * g["count"] + self.global_mean_ * self.smoothing) / (
                g["count"] + self.smoothing
            )
            self.tables_[name] = shrunk.rename(name).reset_index()
            self.key_of_[name] = list(key)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "tables_"):
            raise RuntimeError("call fit() before transform()")
        out = df.copy()
        for name, table in self.tables_.items():
            out = out.merge(table, on=self.key_of_[name], how="left")
            out[name] = out[name].fillna(self.global_mean_)
        return out

    @property
    def feature_names(self) -> list[str]:
        return list(self.tables_.keys())


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

_SCHEDULE_NUMERIC: list[str] = [
    # time
    "minute_of_day", "hour_frac", "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    "doy_sin", "doy_cos", "dow", "is_weekend", "is_holiday",
    "is_festival", "festival_intensity", "festival_x_evening", "month",
    "is_am_peak", "is_pm_peak", "is_peak", "tidal_alignment",
    "peak_toward_cbd", "peak_away_cbd",
    # network position
    "station_seq", "km_from_csmt", "route_share",
    # rake / coach
    "n_cars", "coach_position", "coach_frac", "dist_from_coach_mid",
    "coach_pos_bias", "is_ladies_coach", "is_first_class",
    # planned service
    "planned_headway_min",
    # weather (a forecast is available the night before)
    "temp_c", "humidity_pct", "rain_mm_hr", "rain_3h_mm", "log_rain",
    "log_rain_3h", "is_raining", "heavy_rain", "visibility_km", "is_monsoon",
    "rain_x_peak", "rain_x_tidal",
]

_REALTIME_EXTRA: list[str] = [
    "service_delay_min",
    "headway_min",
    "prev_station_density",
    "prev_station_load",
    "prev_station_left_behind",
]


def feature_columns(feature_set: str, encoder: HistoricalProfileEncoder | None = None) -> list[str]:
    """Ordered predictor list for ``"schedule"`` or ``"realtime"``."""
    cols = list(_SCHEDULE_NUMERIC) + list(CATEGORICAL_FEATURES)
    if feature_set == "realtime":
        cols += _REALTIME_EXTRA
    elif feature_set != "schedule":
        raise ValueError(f"feature_set must be 'schedule' or 'realtime', got {feature_set!r}")
    if encoder is not None and hasattr(encoder, "tables_"):
        cols += encoder.feature_names
    return cols


def prepare_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select predictors and give categoricals the dtype LightGBM wants."""
    X = df[columns].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def assert_no_leakage(columns: list[str]) -> None:
    """Fail loudly if a contemporaneous column ever reaches the model."""
    bad = sorted(set(columns) & set(LEAKY_COLUMNS))
    if bad:
        raise AssertionError(
            f"leaky columns in the feature matrix: {bad}. These are measured at "
            "the same instant as the target; see features.LEAKY_COLUMNS."
        )


# ---------------------------------------------------------------------------
# Temporal splitting
# ---------------------------------------------------------------------------

@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        def _rng(d: pd.DataFrame) -> str:
            return f"{d['date'].min().date()} .. {d['date'].max().date()}  n={len(d):,}"
        return (
            f"  train : {_rng(self.train)}\n"
            f"  val   : {_rng(self.val)}\n"
            f"  test  : {_rng(self.test)}"
        )


def temporal_split(df: pd.DataFrame, val_days: int, test_days: int) -> Split:
    """Split by calendar date, never at random.

    A random split would put 09:14 and 09:19 of the same Tuesday on opposite
    sides of the fence, and neighbouring services on the same track share
    almost all of their state.  The reported score would then be an optimistic
    fiction.  Forecasting is a temporal problem and gets a temporal split.
    """
    dates = np.sort(df["date"].unique())
    if val_days + test_days >= len(dates):
        raise ValueError(
            f"val_days + test_days = {val_days + test_days} but only {len(dates)} days available"
        )
    test_start = dates[-test_days]
    val_start = dates[-(test_days + val_days)]
    return Split(
        train=df[df["date"] < val_start].copy(),
        val=df[(df["date"] >= val_start) & (df["date"] < test_start)].copy(),
        test=df[df["date"] >= test_start].copy(),
    )


def build_design(
    df: pd.DataFrame,
    feature_set: str,
    val_days: int,
    test_days: int,
    cfg: SimConfig | None = None,
    use_history: bool = True,
) -> tuple[Split, list[str], HistoricalProfileEncoder | None]:
    """End-to-end: derive columns, split by date, fit the encoder on train only."""
    df = add_derived_columns(df, cfg)
    split = temporal_split(df, val_days=val_days, test_days=test_days)

    encoder = None
    if use_history:
        encoder = HistoricalProfileEncoder().fit(split.train)
        split = Split(
            train=encoder.transform(split.train),
            val=encoder.transform(split.val),
            test=encoder.transform(split.test),
        )

    cols = feature_columns(feature_set, encoder)
    assert_no_leakage(cols)
    return split, cols, encoder


__all__ = [
    "CATEGORICAL_FEATURES",
    "ID_COLUMNS",
    "LEAKY_COLUMNS",
    "TARGET",
    "HistoricalProfileEncoder",
    "Split",
    "add_derived_columns",
    "assert_no_leakage",
    "build_design",
    "feature_columns",
    "prepare_matrix",
    "temporal_split",
]
