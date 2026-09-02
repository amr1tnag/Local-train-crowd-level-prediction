"""Leakage and split discipline.  These are the tests that stop the project
reporting a beautiful, meaningless R^2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mumbai_crowd.config import SimConfig
from mumbai_crowd.features import (
    CATEGORICAL_FEATURES,
    LEAKY_COLUMNS,
    TARGET,
    HistoricalProfileEncoder,
    add_derived_columns,
    assert_no_leakage,
    build_design,
    feature_columns,
    prepare_matrix,
    temporal_split,
)
from mumbai_crowd.simulate import simulate


@pytest.fixture(scope="module")
def obs():
    return simulate(SimConfig(n_days=16, monitored_service_fraction=0.30), verbose=False).coach_observations


def test_no_feature_set_contains_a_leaky_column():
    for fs in ("schedule", "realtime"):
        cols = feature_columns(fs)
        assert not set(cols) & set(LEAKY_COLUMNS)
        assert_no_leakage(cols)


def test_assert_no_leakage_actually_fires():
    with pytest.raises(AssertionError):
        assert_no_leakage(["hour", "boardings"])
    with pytest.raises(AssertionError):
        assert_no_leakage(["hour", TARGET])


def test_schedule_set_excludes_realtime_signals():
    """A plan made the night before cannot know today's delay."""
    sched = set(feature_columns("schedule"))
    realtime_only = {"service_delay_min", "headway_min", "prev_station_density",
                     "prev_station_load", "prev_station_left_behind"}
    assert not sched & realtime_only
    assert realtime_only <= set(feature_columns("realtime"))


def test_unknown_feature_set_is_rejected():
    with pytest.raises(ValueError):
        feature_columns("psychic")


def test_temporal_split_is_ordered_and_disjoint(obs):
    split = temporal_split(add_derived_columns(obs), val_days=3, test_days=4)
    assert split.train["date"].max() < split.val["date"].min()
    assert split.val["date"].max() < split.test["date"].min()
    assert split.val["date"].nunique() == 3
    assert split.test["date"].nunique() == 4
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == len(obs)


def test_temporal_split_refuses_to_consume_the_whole_history(obs):
    with pytest.raises(ValueError):
        temporal_split(add_derived_columns(obs), val_days=100, test_days=100)


def test_derived_columns_are_sane(obs):
    d = add_derived_columns(obs, SimConfig())
    assert np.allclose(d["tod_sin"] ** 2 + d["tod_cos"] ** 2, 1.0)
    assert d["minute_of_day"].between(0, 1440).all()
    assert d["planned_headway_min"].gt(0).all()
    assert set(d["tidal_alignment"].unique()) <= {0, 1}
    # AM peak towards CSMT and PM peak away from it cannot both be true.
    assert not (d["peak_toward_cbd"].astype(bool) & d["peak_away_cbd"].astype(bool)).any()


def test_historical_encoder_is_fitted_on_train_only(obs):
    """The encoder must not see validation or test targets."""
    d = add_derived_columns(obs)
    split = temporal_split(d, val_days=3, test_days=4)
    enc = HistoricalProfileEncoder().fit(split.train)
    assert enc.global_mean_ == pytest.approx(split.train[TARGET].mean())
    assert enc.global_mean_ != pytest.approx(d[TARGET].mean())

    out = enc.transform(split.test)
    for name in enc.feature_names:
        assert out[name].notna().all()
    # Encoded values must depend only on the key, never on the row's own target.
    key_cols = enc.key_of_[enc.feature_names[0]]
    grouped = out.groupby(key_cols, observed=True)[enc.feature_names[0]].nunique()
    assert (grouped == 1).all()


def test_encoder_shrinks_thin_cells_towards_the_global_mean(obs):
    d = add_derived_columns(obs)
    split = temporal_split(d, val_days=3, test_days=4)
    strong = HistoricalProfileEncoder(smoothing=1e-6).fit(split.train)
    heavy = HistoricalProfileEncoder(smoothing=1e9).fit(split.train)
    name = strong.feature_names[0]
    assert strong.transform(split.val)[name].std() > heavy.transform(split.val)[name].std()
    assert heavy.transform(split.val)[name].std() == pytest.approx(0.0, abs=1e-6)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        HistoricalProfileEncoder().transform(pd.DataFrame({"a": [1]}))


def test_build_design_produces_a_usable_matrix(obs):
    split, cols, enc = build_design(obs, "realtime", val_days=3, test_days=4)
    X = prepare_matrix(split.train, cols)
    assert list(X.columns) == cols
    for c in CATEGORICAL_FEATURES:
        assert str(X[c].dtype) == "category"
    # Only the genuinely unavailable lag columns may be missing.
    missing = {c for c in X.columns if X[c].isna().any()}
    assert missing <= {"prev_station_density", "prev_station_load"}


def test_history_can_be_disabled(obs):
    _, cols, enc = build_design(obs, "schedule", val_days=3, test_days=4, use_history=False)
    assert enc is None
    assert not any(c.startswith("hist_") for c in cols)
