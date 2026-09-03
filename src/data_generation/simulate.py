"""
Core ridership simulation: turns (date, station, hour, direction, coach_type)
+ weather into a simulated occupancy_pct.

occupancy_pct is expressed as % of a coach's rated carrying capacity, and is
allowed to exceed 100%: Mumbai's suburban trains are famous for running at
"super-dense crush loads" well above rated capacity during peak hours - a
2.5x-3x overshoot is a widely reported, real figure (see DATA_GENERATION.md
for the citation), so this simulation lets occupancy_pct range up to 300%
rather than artificially capping at 100%. A value of 100% is "at rated
capacity", not "the maximum possible".

Modeling approach (see DATA_GENERATION.md for the full narrative)
-------------------------------------------------------------------
occupancy_pct is built as a sum of an off-peak "floor" and a peak "swing",
each scaled independently by several factors, then adjusted for coach type
and given noise:

  floor      : baseline ridership present at all hours (turns
               station-type-dependent: interchange hubs and CBD-type
               stations keep a much higher floor than sleepy residential
               or dockland stations).
  peak_swing : the AM/PM commute bulge. Its size is a function of how far
               the train has "filled up" along the route by the time it
               reaches this station (the direction-dependent accumulation
               model below) - this is what makes occupancy at a station
               far from CSMT lower in the AM (train hasn't filled up yet)
               and higher close to CSMT (train has been accumulating
               boarders the whole way).

Both terms are then scaled by day-type (weekday/weekend/holiday), a
monsoon-rainfall regime (see `compute_monsoon_regime`), and mega-block
service reduction, then by a per-station "shape" multiplier that gives
each station_type its own defensible daily silhouette (this is what
Phase 2's clustering is meant to discover), then by a coach-type
multiplier, then noise is added and the result is clipped to [0, 300].

This is a deliberately-designed simulator, not a physical model of actual
train operations - every design choice here is a documented, defensible
assumption for a course project, not a claim of ground-truth accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stations import STATIONS, MAX_DISTANCE_KM

HOURS = list(range(5, 24))  # service hours 05:00-23:00
DIRECTIONS = ["UP", "DOWN"]  # UP = towards CSMT (CBD-bound); DOWN = towards Panvel
COACH_TYPES = ["General", "Ladies", "First Class"]

AM_PEAK_MU, AM_PEAK_SIGMA = 8.5, 1.3
PM_PEAK_MU, PM_PEAK_SIGMA = 19.0, 1.5
PEAK_HOURS = set(range(7, 11)) | set(range(18, 22))

MEGA_BLOCK_HOURS = set(range(11, 16))  # 11:00-16:00, simplification (see calendar_features.py)


def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def compute_monsoon_regime(weather_df: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """
    One regime draw per date, so a day's disruption (or lack of it) is
    coherent across all stations/hours that day rather than i.i.d. noise.

    Regimes, keyed off IMD's rainfall intensity band for that date:
      none/very_light/light : normal service, multiplier 1.0
      moderate               : mild crowding increase (rushing to beat
                                disruption, fewer road alternatives) - a
                                deterministic mild multiplier, no regime
                                draw needed.
      heavy                  : 70% "strained" (fewer trains -> much more
                                crowded), 30% "suspended" (line paused,
                                stations empty out) for peak hours.
      very_heavy/extremely_heavy : 85% "suspended", 15% "last-train-out
                                packed" strained regime.
    Suspension multipliers only bite during peak hours; off-peak hours on
    a disrupted day are treated as only mildly reduced, since disruption
    is a peak-service phenomenon in the real network.
    """
    rng = np.random.default_rng(seed)
    band = weather_df["rain_intensity_band"].to_numpy()
    mm = weather_df["rainfall_mm"].to_numpy()

    n = len(weather_df)
    regime = np.array(["normal"] * n, dtype=object)
    peak_mult = np.ones(n)
    offpeak_mult = np.ones(n)

    moderate_mask = band == "moderate"
    moderate_intensity = np.clip((mm - 15.6) / (64.5 - 15.6), 0, 1)
    peak_mult[moderate_mask] = 1.0 + 0.15 * moderate_intensity[moderate_mask]
    offpeak_mult[moderate_mask] = 1.0 + 0.08 * moderate_intensity[moderate_mask]
    regime[moderate_mask] = "moderate_rain"

    heavy_mask = band == "heavy"
    heavy_idx = np.where(heavy_mask)[0]
    strained = rng.random(len(heavy_idx)) < 0.70
    for j, idx in enumerate(heavy_idx):
        if strained[j]:
            peak_mult[idx] = rng.uniform(1.3, 1.6)
            offpeak_mult[idx] = rng.uniform(1.05, 1.2)
            regime[idx] = "heavy_rain_strained"
        else:
            peak_mult[idx] = rng.uniform(0.3, 0.5)
            offpeak_mult[idx] = rng.uniform(0.6, 0.8)
            regime[idx] = "heavy_rain_suspended"

    very_heavy_mask = np.isin(band, ["very_heavy", "extremely_heavy"])
    vh_idx = np.where(very_heavy_mask)[0]
    still_running = rng.random(len(vh_idx)) < 0.15
    for j, idx in enumerate(vh_idx):
        if still_running[j]:
            peak_mult[idx] = rng.uniform(1.4, 1.8)
            offpeak_mult[idx] = rng.uniform(1.0, 1.15)
            regime[idx] = "extreme_rain_strained"
        else:
            peak_mult[idx] = rng.uniform(0.15, 0.4)
            offpeak_mult[idx] = rng.uniform(0.35, 0.55)
            regime[idx] = "extreme_rain_suspended"

    out = weather_df[["date"]].copy()
    out["rain_regime"] = regime
    out["rain_peak_multiplier"] = peak_mult
    out["rain_offpeak_multiplier"] = offpeak_mult
    return out


def _station_shape_multiplier(hour: np.ndarray, station_type: np.ndarray) -> np.ndarray:
    is_peak = np.isin(hour, list(PEAK_HOURS))
    mult = np.ones(len(hour))

    dock = station_type == "industrial_dock"
    mult = np.where(dock & is_peak, 0.55, mult)
    mult = np.where(dock & ~is_peak, 0.35, mult)

    hub = station_type == "interchange_hub"
    mult = np.where(hub & is_peak, 1.0, mult)
    mult = np.where(hub & ~is_peak, 1.35, mult)

    sec_cbd = station_type == "secondary_cbd"
    mult = np.where(sec_cbd & is_peak, 1.0, mult)
    mult = np.where(sec_cbd & ~is_peak, 1.1, mult)

    term = np.isin(station_type, ["cbd_terminal", "terminal_hub"])
    mult = np.where(term & is_peak, 1.0, mult)
    mult = np.where(term & ~is_peak, 1.15, mult)

    # residential: leave at 1.0 (the reference sharp-peak / quiet-trough shape)
    return mult


def build_grid(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Cartesian product of calendar x station x hour x direction x coach_type."""
    station_df = pd.DataFrame([{
        "station_id": s.station_id,
        "station_name": s.name,
        "station_order": s.order,
        "distance_km": s.distance_km,
        "station_type": s.station_type,
        "is_interchange": s.is_interchange,
    } for s in STATIONS])

    combos = pd.MultiIndex.from_product(
        [station_df["station_id"], HOURS, DIRECTIONS, COACH_TYPES],
        names=["station_id", "hour", "direction", "coach_type"],
    ).to_frame(index=False)
    combos = combos.merge(station_df, on="station_id", how="left")

    calendar_df = calendar_df.copy()
    calendar_df["_key"] = 1
    combos["_key"] = 1
    grid = calendar_df.merge(combos, on="_key").drop(columns="_key")
    return grid


def simulate_occupancy(grid: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = grid.copy()

    hour = df["hour"].to_numpy(dtype=float)
    distance_km = df["distance_km"].to_numpy(dtype=float)
    direction = df["direction"].to_numpy()
    station_type = df["station_type"].to_numpy()

    prox_to_cst = 1.0 - distance_km / MAX_DISTANCE_KM     # 1 near CSMT, 0 near Panvel
    prox_from_cst = distance_km / MAX_DISTANCE_KM          # 0 near CSMT, 1 near Panvel

    am = _gaussian(hour, AM_PEAK_MU, AM_PEAK_SIGMA)
    pm = _gaussian(hour, PM_PEAK_MU, PM_PEAK_SIGMA)

    is_up = direction == "UP"
    peak_swing = np.where(
        is_up,
        am * (0.35 + 1.65 * prox_to_cst) + pm * (0.25 + 0.35 * prox_to_cst),
        pm * (0.35 + 1.65 * prox_from_cst) + am * (0.25 + 0.35 * prox_from_cst),
    )

    floor = np.full(len(df), 0.20)

    # Belapur CBD reverse-commute bump: local workers arrive DOWN in the AM,
    # leave UP in the PM - the opposite of the CSMT-bound pattern.
    is_sec_cbd = station_type == "secondary_cbd"
    reverse_peak = np.where(is_sec_cbd, np.where(is_up, pm, am) * 0.30, 0.0)

    is_weekend = df["is_weekend"].to_numpy()
    is_holiday = df["is_holiday"].to_numpy()
    day_of_week = df["day_of_week"].to_numpy()

    peak_day_mult = np.ones(len(df))
    peak_day_mult = np.where(day_of_week == "Saturday", 0.80, peak_day_mult)
    peak_day_mult = np.where(day_of_week == "Sunday", 0.55, peak_day_mult)
    peak_day_mult = np.where(is_holiday, 0.45, peak_day_mult)

    floor_day_mult = np.ones(len(df))
    floor_day_mult = np.where(day_of_week == "Saturday", 0.90, floor_day_mult)
    floor_day_mult = np.where(day_of_week == "Sunday", 0.75, floor_day_mult)
    floor_day_mult = np.where(is_holiday, 0.70, floor_day_mult)

    is_mega_block = df["is_mega_block_day"].to_numpy()
    hour_int = df["hour"].to_numpy()
    in_block_window = np.isin(hour_int, list(MEGA_BLOCK_HOURS))
    block_mult = np.where(is_mega_block & in_block_window, 1.6, 1.0)

    is_peak_hour = np.isin(hour_int, list(PEAK_HOURS))
    rain_mult = np.where(is_peak_hour, df["rain_peak_multiplier"].to_numpy(),
                          df["rain_offpeak_multiplier"].to_numpy())

    shape_mult = _station_shape_multiplier(hour_int, station_type)

    commute_component = (floor * floor_day_mult + (peak_swing + reverse_peak) * peak_day_mult * block_mult) \
        * rain_mult * shape_mult

    peak_intensity = np.clip(am + pm, 0, 1)  # ~0 off-peak, ~1 at peak center
    coach_type = df["coach_type"].to_numpy()
    coach_mult = np.ones(len(df))
    coach_mult = np.where(coach_type == "Ladies", 1.05 - 0.20 * peak_intensity, coach_mult)
    coach_mult = np.where(coach_type == "First Class", 0.35 + 0.30 * peak_intensity, coach_mult)

    station_day_key = df["station_id"].astype(str) + "_" + df["date"].astype(str)
    unique_keys, inverse = np.unique(station_day_key.to_numpy(), return_inverse=True)
    station_day_noise_vals = rng.normal(0, 8, size=len(unique_keys))
    station_day_noise = station_day_noise_vals[inverse]
    row_noise = rng.normal(0, 4, size=len(df))

    occupancy_pct = commute_component * coach_mult * 100 + station_day_noise + row_noise
    occupancy_pct = np.clip(occupancy_pct, 0, 300)

    df["occupancy_pct"] = np.round(occupancy_pct, 1)
    return df
