"""
Daily weather generation for Mumbai, calibrated against real climate
normals (not raw historical time series — see provenance note below).

Provenance
----------
This project's brief asked to pull real historical Mumbai weather (IMD
public data or a weather API) where feasible, and fall back to
statistically plausible synthetic weather otherwise. A live pull was
attempted against the Open-Meteo Historical Weather Archive API
(https://archive-api.open-meteo.com), which needs no API key and covers
Mumbai back to 1940 - but this sandbox's network egress policy blocks
that host (and en.wikipedia.org, used to try to fetch the same normals
tabulated). That failure is recorded, not silently swallowed:
DATA_GENERATION.md documents the attempted call and the policy denial.

Falling back per the brief: daily rainfall and temperature are generated
synthetically, but *calibrated* to real published IMD Santacruz-station
climate normals (Mumbai's official long-period-average weather station),
obtained via live web search rather than invented outright:
  - Monthly rainfall normals and the ~2,502 mm annual total: corroborated
    directly from search results citing IMD Santacruz data.
  - Monthly mean max/min temperatures: corroborated from search results
    ("mean maximum ~32C summer / 30C winter, mean minimum ~26C summer /
    18C winter") and interpolated month-to-month.
  - IMD's daily rainfall intensity bands (light/moderate/heavy/very heavy/
    extremely heavy, in mm/24hr) are IMD's standard published
    classification, used here to decide monsoon-disruption regimes in
    simulate.py.

Generation method: for each date, a wet/dry state is drawn from a
2-state Markov chain (persistence tuned per month, so monsoon rain
arrives in realistic multi-day bursts rather than i.i.d. days), and wet
days draw a rainfall amount from a Gamma distribution whose mean is set
so the expected monthly total matches the calibrated normal for that
month. Temperature follows a smooth seasonal curve through the monthly
normals with small AR(1) day-to-day noise, further reduced on wet days
(rain measurably cools Mumbai - a real, well-documented effect).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Monthly normals, 1-indexed by month. Rainfall in mm (monthly total).
# Temperature in degrees C (monthly mean max / mean min).
MONTHLY_RAINFALL_MM = {
    1: 0.2, 2: 0.2, 3: 0.3, 4: 1.1, 5: 16.2, 6: 526.3,
    7: 919.9, 8: 560.8, 9: 383.5, 10: 85.0, 11: 16.5, 12: 1.6,
}
MONTHLY_TEMP_MAX_C = {
    1: 31.0, 2: 31.5, 3: 32.5, 4: 33.0, 5: 33.5, 6: 32.0,
    7: 30.0, 8: 29.5, 9: 30.5, 10: 33.0, 11: 33.0, 12: 32.0,
}
MONTHLY_TEMP_MIN_C = {
    1: 18.5, 2: 19.0, 3: 22.0, 4: 25.0, 5: 27.0, 6: 26.0,
    7: 25.0, 8: 24.5, 9: 24.5, 10: 24.0, 11: 21.5, 12: 19.0,
}

# P(wet day) by month, and Markov persistence (P(wet tomorrow | wet today))
# tuned so monsoon rain clusters into multi-day spells like the real thing.
MONTHLY_P_WET = {
    1: 0.01, 2: 0.01, 3: 0.02, 4: 0.05, 5: 0.15, 6: 0.55,
    7: 0.75, 8: 0.70, 9: 0.55, 10: 0.20, 11: 0.06, 12: 0.02,
}
MONTHLY_WET_PERSISTENCE = {
    1: 0.10, 2: 0.10, 3: 0.10, 4: 0.15, 5: 0.25, 6: 0.55,
    7: 0.70, 8: 0.65, 9: 0.55, 10: 0.30, 11: 0.15, 12: 0.10,
}

DAYS_IN_MONTH_2023 = {m: pd.Period(f"2023-{m:02d}").days_in_month for m in range(1, 13)}


def _rainfall_intensity_band(mm: float) -> str:
    """IMD's standard 24hr rainfall intensity classification."""
    if mm <= 0.0:
        return "none"
    if mm < 2.5:
        return "very_light"
    if mm < 15.6:
        return "light"
    if mm < 64.5:
        return "moderate"
    if mm < 115.6:
        return "heavy"
    if mm < 204.5:
        return "very_heavy"
    return "extremely_heavy"


def generate_weather(dates: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(dates)
    months = dates.month.to_numpy()

    # --- wet/dry Markov chain ---
    wet = np.zeros(n, dtype=bool)
    wet[0] = rng.random() < MONTHLY_P_WET[int(months[0])]
    for i in range(1, n):
        m = int(months[i])
        p_wet_base = MONTHLY_P_WET[m]
        persistence = MONTHLY_WET_PERSISTENCE[m]
        if wet[i - 1]:
            p = persistence + (1 - persistence) * p_wet_base
        else:
            p = p_wet_base * (1 - persistence)
        wet[i] = rng.random() < p

    # --- rainfall amount on wet days, Gamma-distributed, mean tuned per month ---
    rainfall = np.zeros(n)
    for m in range(1, 13):
        mask = (months == m) & wet
        n_wet_expected = max(DAYS_IN_MONTH_2023[m] * MONTHLY_P_WET[m], 1e-6)
        mean_wet_day_mm = MONTHLY_RAINFALL_MM[m] / n_wet_expected
        mean_wet_day_mm = max(mean_wet_day_mm, 0.1)
        shape = 1.3  # heavier right tail than exponential - occasional deluge days
        scale = mean_wet_day_mm / shape
        n_days_m = int(mask.sum())
        if n_days_m > 0:
            rainfall[mask] = rng.gamma(shape=shape, scale=scale, size=n_days_m)

    # --- temperature: seasonal curve through monthly normals + AR(1) noise ---
    doy = dates.dayofyear.to_numpy()
    month_mid_doy = np.array([pd.Timestamp(f"2023-{m:02d}-15").dayofyear for m in range(1, 13)])
    tmax_normals = np.array([MONTHLY_TEMP_MAX_C[m] for m in range(1, 13)])
    tmin_normals = np.array([MONTHLY_TEMP_MIN_C[m] for m in range(1, 13)])
    tmax_smooth = np.interp(doy, month_mid_doy, tmax_normals, period=365)
    tmin_smooth = np.interp(doy, month_mid_doy, tmin_normals, period=365)

    ar_noise = np.zeros(n)
    for i in range(1, n):
        ar_noise[i] = 0.6 * ar_noise[i - 1] + rng.normal(0, 0.7)
    temp_max = tmax_smooth + ar_noise
    temp_min = tmin_smooth + ar_noise * 0.7

    # Rain measurably cools Mumbai - documented real effect.
    rain_cooling = np.clip(rainfall / 40.0, 0, 3.0)
    temp_max -= rain_cooling
    temp_min -= rain_cooling * 0.5
    temp_min = np.minimum(temp_min, temp_max - 2.0)

    temperature_c = (temp_max + temp_min) / 2.0

    df = pd.DataFrame({
        "date": dates,
        "rainfall_mm": np.round(rainfall, 1),
        "temp_max_c": np.round(temp_max, 1),
        "temp_min_c": np.round(temp_min, 1),
        "temperature_c": np.round(temperature_c, 1),
    })
    df["rain_intensity_band"] = df["rainfall_mm"].apply(_rainfall_intensity_band)
    return df
