"""
Sanity tests for the synthetic data generator. These check structural
invariants and the directional/behavioral assumptions documented in
DATA_GENERATION.md — not "correctness" against real data, since there is
no real data (this is a synthetic dataset by design).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation.generate_dataset import generate
from src.data_generation.stations import STATIONS, MAX_DISTANCE_KM


def _small_dataset():
    # One week spanning a weekday and a Sunday, cheap to regenerate.
    return generate(start="2023-01-02", end="2023-01-08", seed=42)


def test_row_count_matches_grid_size():
    df = _small_dataset()
    n_days = 7
    n_stations = len(STATIONS)
    n_hours = 19
    n_directions = 2
    n_coach_types = 3
    assert len(df) == n_days * n_stations * n_hours * n_directions * n_coach_types


def test_occupancy_bounds():
    df = _small_dataset()
    assert df["occupancy_pct"].min() >= 0.0
    assert df["occupancy_pct"].max() <= 300.0
    assert df["occupancy_pct"].isnull().sum() == 0


def test_stations_are_the_verified_harbour_line_list():
    names = {s.name for s in STATIONS}
    assert "Vidyavihar" not in names  # confirmed Central Line, not Harbour Line
    assert "King's Circle" not in names  # confirmed Goregaon branch, not CSMT-Panvel trunk
    assert "CSMT" in names and "Panvel" in names
    assert len(STATIONS) == 25


def test_am_peak_load_builds_towards_cst_for_up_direction():
    """UP trains (CBD-bound) should be more crowded near CSMT than near Panvel
    during the AM peak — the train has accumulated boarders along the way."""
    df = _small_dataset()
    sub = df[
        (df.hour == 8) & (df.direction == "UP") & (df.coach_type == "General")
        & (df.day_of_week == "Monday")
    ]
    near_cst = sub[sub.station_order <= 2]["occupancy_pct"].mean()
    near_panvel = sub[sub.station_order >= 22]["occupancy_pct"].mean()
    assert near_cst > near_panvel


def test_pm_peak_load_builds_away_from_cst_for_down_direction():
    df = _small_dataset()
    sub = df[
        (df.hour == 19) & (df.direction == "DOWN") & (df.coach_type == "General")
        & (df.day_of_week == "Monday")
    ]
    near_cst = sub[sub.station_order <= 2]["occupancy_pct"].mean()
    near_panvel = sub[sub.station_order >= 22]["occupancy_pct"].mean()
    assert near_panvel > near_cst


def test_weekday_more_crowded_than_holiday_at_peak():
    # 2023-01-23..27 (Mon-Fri) includes Republic Day (Thu 26th) as a weekday
    # holiday, isolating the holiday effect from the weekend effect.
    df = generate(start="2023-01-23", end="2023-01-27", seed=42)
    sub = df[
        (df.hour == 8) & (df.direction == "UP") & (df.coach_type == "General")
        & (df.station_order <= 2)
    ]
    weekday = sub[~sub.is_weekend & ~sub.is_holiday]["occupancy_pct"].mean()
    holiday = sub[sub.is_holiday]["occupancy_pct"].mean()
    assert weekday > holiday


def test_general_coach_more_crowded_than_first_class_at_peak():
    df = _small_dataset()
    sub = df[
        (df.hour == 8) & (df.direction == "UP") & (df.station_order <= 2)
        & (df.day_of_week == "Monday")
    ]
    general = sub[sub.coach_type == "General"]["occupancy_pct"].mean()
    first_class = sub[sub.coach_type == "First Class"]["occupancy_pct"].mean()
    assert general > first_class


def test_deterministic_given_seed():
    df1 = generate(start="2023-01-02", end="2023-01-03", seed=99)
    df2 = generate(start="2023-01-02", end="2023-01-03", seed=99)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))
