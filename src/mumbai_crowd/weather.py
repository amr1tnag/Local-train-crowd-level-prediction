"""Hourly weather generator for Mumbai.

Weather is not decoration in this project -- it is one of the two feature
families the CO2 regression is built on (the other being time).  In Mumbai the
mechanism is concrete: heavy monsoon rain simultaneously (a) suppresses road
and two-wheeler travel, pushing riders onto the rail network, (b) slows the
locals down, stretching headways, and (c) makes people bunch under the
foot-over-bridge rather than walk the platform.  All three push coach density
up, so rainfall is a genuinely causal predictor, not a spurious one.

The generator is a seasonal climatology for Mumbai (Santacruz observatory
normals) plus a two-state wet/dry Markov chain for hourly rain, which
reproduces the burstiness that a plain i.i.d. draw would miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Monthly climatological normals for Mumbai: (mean daily max C, mean daily min C,
# mean relative humidity %, probability an hour is wet, mean wet-hour rain mm).
# Calibrated so that a simulated year lands near Mumbai's ~2,300 mm normal,
# with the June-September window carrying ~95% of it.
_MONTHLY_NORMALS: dict[int, tuple[float, float, float, float, float]] = {
    1:  (31.0, 17.5, 62, 0.002, 0.4),
    2:  (31.5, 18.5, 64, 0.002, 0.4),
    3:  (33.0, 21.5, 68, 0.004, 0.6),
    4:  (33.5, 24.5, 72, 0.008, 0.9),
    5:  (33.5, 27.0, 74, 0.020, 1.6),
    6:  (32.0, 26.5, 84, 0.230, 4.2),
    7:  (30.0, 25.5, 90, 0.340, 4.3),
    8:  (29.8, 25.0, 90, 0.300, 2.7),
    9:  (30.5, 24.5, 87, 0.190, 3.6),
    10: (33.0, 24.0, 76, 0.055, 2.0),
    11: (33.5, 22.0, 68, 0.012, 0.8),
    12: (32.0, 19.5, 64, 0.004, 0.5),
}

# Two-state Markov persistence: rain begets rain.
_P_WET_GIVEN_WET = 0.72


def generate_weather(
    dates: pd.DatetimeIndex,
    hours: range | list[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return one row per (date, hour) with temperature, humidity and rain.

    Columns
    -------
    date, hour, temp_c, humidity_pct, rain_mm_hr, rain_3h_mm, is_raining,
    heavy_rain, visibility_km, is_monsoon
    """
    hours = list(hours)
    records = []
    wet = False

    for date in dates:
        month = date.month
        t_max, t_min, rh, p_wet, rain_mu = _MONTHLY_NORMALS[month]
        # Day-to-day synoptic wobble, shared by every hour of the day.
        day_temp_shift = rng.normal(0.0, 1.4)
        day_wet_shift = float(np.clip(rng.normal(1.0, 0.55), 0.15, 2.6))

        for hour in hours:
            # Diurnal temperature curve: minimum ~05:00, maximum ~15:00.
            phase = 2 * np.pi * (hour - 5) / 24.0
            temp = (t_min + t_max) / 2 - (t_max - t_min) / 2 * np.cos(phase)
            temp += day_temp_shift + rng.normal(0.0, 0.5)

            # Calibrate the dry->wet transition so the chain's *stationary*
            # wet fraction still equals the climatological p_wet despite the
            # persistence term (pi = p_dw / (1 - p_ww + p_dw)).
            pi = min(0.90, p_wet * day_wet_shift)
            p_dry_to_wet = pi * (1.0 - _P_WET_GIVEN_WET) / max(1.0 - pi, 1e-6)
            p_transition = _P_WET_GIVEN_WET if wet else min(p_dry_to_wet, 0.95)
            wet = rng.random() < p_transition
            if wet:
                # Gamma-distributed intensity: mostly drizzle, occasional
                # 40 mm/hr cloudburst of the 26-July-2005 family.
                rain = float(rng.gamma(shape=0.75, scale=rain_mu / 0.75))
                rain = min(rain, 80.0)
            else:
                rain = 0.0

            humidity = float(np.clip(rh + 9.0 * (rain > 0) + rng.normal(0, 4.0), 30, 100))
            visibility = float(np.clip(10.0 - 0.28 * rain + rng.normal(0, 0.6), 0.3, 12.0))

            records.append(
                {
                    "date": date.normalize(),
                    "hour": hour,
                    "temp_c": round(float(temp), 2),
                    "humidity_pct": round(humidity, 1),
                    "rain_mm_hr": round(rain, 2),
                    "visibility_km": round(visibility, 2),
                }
            )

    wx = pd.DataFrame.from_records(records)
    wx = wx.sort_values(["date", "hour"], kind="stable").reset_index(drop=True)

    # Rolling 3-hour accumulation: waterlogging is a *cumulative* phenomenon,
    # and it is the accumulation -- not the instantaneous rate -- that floods
    # the Sandhurst Road / Kurla tracks and collapses the timetable.
    wx["rain_3h_mm"] = (
        wx.groupby("date", sort=False)["rain_mm_hr"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
        .round(2)
    )
    wx["is_raining"] = (wx["rain_mm_hr"] > 0.1).astype(int)
    wx["heavy_rain"] = (wx["rain_mm_hr"] >= 7.5).astype(int)
    wx["is_monsoon"] = wx["date"].dt.month.isin([6, 7, 8, 9]).astype(int)
    return wx


def rain_demand_multiplier(rain_mm_hr: np.ndarray, rain_3h_mm: np.ndarray) -> np.ndarray:
    """Extra rail demand caused by rain (mode shift off the roads).

    Saturating rather than linear: the first few mm push a lot of scooter and
    bus riders onto the train, after which there is nobody left to shift, and
    at extreme accumulations people simply abandon the trip.
    """
    rain_mm_hr = np.asarray(rain_mm_hr, dtype=float)
    rain_3h_mm = np.asarray(rain_3h_mm, dtype=float)
    shift = 0.20 * (1.0 - np.exp(-rain_mm_hr / 4.0))
    # Extreme waterlogging: trips get cancelled outright.
    suppression = 0.30 * (1.0 - np.exp(-np.maximum(rain_3h_mm - 45.0, 0.0) / 30.0))
    return 1.0 + shift - suppression


def rain_headway_multiplier(rain_mm_hr: np.ndarray, rain_3h_mm: np.ndarray) -> np.ndarray:
    """How much rain stretches the effective gap between trains.

    A stretched headway is the mechanism that turns a wet evening into a
    dangerous one: the *same* demand accumulates on the platform for longer
    before a train arrives to absorb it.
    """
    rain_mm_hr = np.asarray(rain_mm_hr, dtype=float)
    rain_3h_mm = np.asarray(rain_3h_mm, dtype=float)
    return 1.0 + 0.030 * rain_mm_hr + 0.008 * rain_3h_mm


__all__ = [
    "generate_weather",
    "rain_demand_multiplier",
    "rain_headway_multiplier",
]
