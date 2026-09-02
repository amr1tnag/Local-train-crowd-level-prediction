"""Indian public holidays and Mumbai festival windows.

Ganeshotsav is not a footnote for this problem.  For eleven days the evening
and late-night loading on the Harbour Line goes far outside anything the rest
of the year contains, and Anant Chaturdashi (visarjan day) is the single worst
crowd-safety night on the Mumbai suburban calendar.  A model that has never
seen it will under-predict exactly when under-prediction is most expensive,
which is the whole point of this project -- so the flag is a first-class
feature, not a nuisance variable.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

#: Gazetted / widely observed public holidays that visibly change Mumbai
#: suburban ridership.  Dates are the observed Maharashtra dates.
PUBLIC_HOLIDAYS: dict[str, str] = {
    # 2024
    "2024-06-17": "Bakri Id",
    "2024-07-17": "Muharram",
    "2024-08-15": "Independence Day",
    "2024-08-19": "Raksha Bandhan",
    "2024-08-26": "Janmashtami",
    "2024-09-07": "Ganesh Chaturthi",
    "2024-09-16": "Id-e-Milad",
    "2024-10-02": "Gandhi Jayanti",
    "2024-10-12": "Dussehra",
    "2024-11-01": "Diwali (Laxmi Pujan)",
    "2024-12-25": "Christmas",
    # 2025
    "2025-01-26": "Republic Day",
    "2025-03-14": "Holi / Dhulivandan",
    "2025-03-31": "Id-ul-Fitr",
    "2025-04-14": "Ambedkar Jayanti",
    "2025-05-01": "Maharashtra Day",
    "2025-08-15": "Independence Day",
    "2025-08-27": "Ganesh Chaturthi",
    "2025-10-02": "Gandhi Jayanti / Dussehra",
    "2025-10-21": "Diwali (Laxmi Pujan)",
    "2025-12-25": "Christmas",
    # 2026
    "2026-01-26": "Republic Day",
    "2026-05-01": "Maharashtra Day",
    "2026-08-15": "Independence Day",
    "2026-09-14": "Ganesh Chaturthi",
    "2026-10-02": "Gandhi Jayanti",
}

#: (start, end) inclusive windows of the ten-day Ganeshotsav festival, whose
#: last day (Anant Chaturdashi / visarjan) is the peak.
GANESHOTSAV: list[tuple[str, str]] = [
    ("2024-09-07", "2024-09-17"),
    ("2025-08-27", "2025-09-06"),
    ("2026-09-14", "2026-09-24"),
]


def _in_window(date: _dt.date, windows: list[tuple[str, str]]) -> tuple[bool, int]:
    for start, end in windows:
        s = _dt.date.fromisoformat(start)
        e = _dt.date.fromisoformat(end)
        if s <= date <= e:
            return True, (date - s).days + 1
    return False, 0


def build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per date with holiday / festival flags and demand multipliers."""
    rows = []
    for ts in dates:
        d = ts.date()
        key = d.isoformat()
        is_holiday = key in PUBLIC_HOLIDAYS
        in_fest, fest_day = _in_window(d, GANESHOTSAV)

        # Festival loading ramps through the ten days and spikes on visarjan.
        if in_fest:
            festival_intensity = 0.10 + 0.045 * (fest_day - 1)
            if fest_day == 11:
                festival_intensity = 0.85          # Anant Chaturdashi
            elif fest_day in (1, 2):
                festival_intensity = 0.35          # installation days
        else:
            festival_intensity = 0.0

        rows.append(
            {
                "date": ts.normalize(),
                "dow": int(ts.weekday()),
                "is_weekend": int(ts.weekday() >= 5),
                "is_holiday": int(is_holiday),
                "holiday_name": PUBLIC_HOLIDAYS.get(key, ""),
                "is_festival": int(in_fest),
                "festival_day": fest_day,
                "festival_intensity": round(float(festival_intensity), 3),
                "month": int(ts.month),
                "day_of_year": int(ts.dayofyear),
            }
        )
    return pd.DataFrame.from_records(rows)


__all__ = ["GANESHOTSAV", "PUBLIC_HOLIDAYS", "build_calendar"]
