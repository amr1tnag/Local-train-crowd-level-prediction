"""
Calendar features: day-of-week, holidays, monsoon window, mega-block days.

Holiday list
------------
Maharashtra/India public holidays observed in Mumbai for calendar year
2023 (the simulated year), from general public knowledge of the Indian
national + Maharashtra state holiday calendar. This is a fixed list of
named dates rather than a rule-based calculator, since several (Holi,
Ganesh Chaturthi, Diwali, Eid) follow lunar/regional calendars that are
impractical to derive algorithmically here. Documented as a simplification
in DATA_GENERATION.md.

Mega-block days
----------------
Indian Railways runs planned maintenance "mega blocks" on Sundays, which
in reality suspend/reduce service on one line for a several-hour window
(commonly late morning to early afternoon), not the whole day. This
project simplifies a mega block to a whole-day station-level flag with
~40% of Sundays affected (rest are unaffected Sundays), and applies its
crowding effect only during the block's simulated hours (11:00-16:00) in
simulate.py — approximating the real practice while keeping the calendar
feature itself a simple boolean per date. This is a deliberate
simplification, documented here and in DATA_GENERATION.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAYS_2023 = {
    "2023-01-01": "New Year's Day",
    "2023-01-26": "Republic Day",
    "2023-03-08": "Holi",
    "2023-03-30": "Ram Navami",
    "2023-04-07": "Good Friday",
    "2023-04-14": "Dr. Ambedkar Jayanti",
    "2023-04-22": "Eid-ul-Fitr",
    "2023-05-01": "Maharashtra Din",
    "2023-06-29": "Bakri Eid",
    "2023-08-15": "Independence Day",
    "2023-08-29": "Raksha Bandhan",
    "2023-09-19": "Ganesh Chaturthi",
    "2023-10-02": "Gandhi Jayanti",
    "2023-10-24": "Dussehra",
    "2023-11-12": "Diwali (Laxmi Pujan)",
    "2023-11-13": "Diwali (Balipratipada)",
    "2023-11-27": "Guru Nanak Jayanti",
    "2023-12-25": "Christmas",
}


def build_calendar(start: str = "2023-01-01", end: str = "2023-12-31", seed: int = 42) -> pd.DataFrame:
    """One row per date with all date-level (station-independent) features."""
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)

    holiday_dates = set(HOLIDAYS_2023.keys())
    df = pd.DataFrame({"date": dates})
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5  # Sat=5, Sun=6
    df["month"] = df["date"].dt.month
    df["is_holiday"] = df["date_str"].isin(holiday_dates)
    df["holiday_name"] = df["date_str"].map(HOLIDAYS_2023).fillna("")
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9])

    is_sunday = df["date"].dt.dayofweek == 6
    sunday_block_roll = rng.random(len(df)) < 0.40
    df["is_mega_block_day"] = is_sunday & sunday_block_roll

    df = df.drop(columns=["date_str"])
    return df
