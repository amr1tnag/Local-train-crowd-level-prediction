"""Origin-destination demand model for the Harbour Line.

Why bother with a full OD model instead of drawing "crowd level" straight from
a distribution?  Because the thing we are trying to predict -- how many people
are inside *one coach* of *one train* at *one station* -- is a downstream
consequence of a queueing process:

    trips generated -> passengers accumulate on the platform ->
    a train arrives with finite spare capacity -> some board, some are
    left behind -> load accumulates along the run and drains at the sinks.

Only a mechanism like that produces the features a model must actually learn:
the tidal AM/PM reversal, load peaking mid-route rather than at the terminus,
left-behind passengers compounding when headways stretch in the rain, and the
long right tail where danger lives.  Sampling a marginal distribution would
give a dataset whose "hard" cases are pure noise, and any claimed skill on it
would be an artefact.

Trip taxonomy
-------------
Every trip belongs to one of three purposes, each with its own time-of-day
profile and its own destination-choice rule:

``to_work``   generated in proportion to a station's *residential* catchment,
              attracted to *employment* mass.  Drives the AM peak.
``to_home``   generated in proportion to *employment*, attracted to
              *residential* mass.  Drives the (fatter, later) PM peak.
``other``     education, shopping, hospital, reverse-commute.  Nearly flat,
              attracted to a blend of both masses; it is what keeps the
              off-peak floor above zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .network import ROUTES, load_stations

# ---------------------------------------------------------------------------
# Time-of-day profiles
# ---------------------------------------------------------------------------


def _gauss(t: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


def hourly_profiles(day_type: str) -> dict[str, np.ndarray]:
    """Trip-generation profile per purpose over hours 0..23 (each sums to 1).

    ``day_type`` is ``"weekday"``, ``"saturday"`` or ``"sunday"``.
    """
    t = np.arange(24, dtype=float)

    if day_type == "weekday":
        to_work = (
            1.00 * _gauss(t, 9.0, 0.95)      # the office peak, and it is sharp
            + 0.40 * _gauss(t, 7.5, 1.00)    # early shift / factory / school
            + 0.12 * _gauss(t, 11.5, 1.60)   # late starters
            + 0.09 * _gauss(t, 17.6, 2.20)   # night-shift departures
            + 0.018
        )
        to_home = (
            1.00 * _gauss(t, 18.9, 1.25)     # the evening peak: later and
            + 0.48 * _gauss(t, 20.6, 1.45)   # broader than the morning one
            + 0.22 * _gauss(t, 16.4, 1.40)
            + 0.09 * _gauss(t, 13.0, 1.80)
            + 0.018
        )
        other = 0.55 * _gauss(t, 12.5, 3.6) + 0.45 * _gauss(t, 19.0, 3.0) + 0.10
    elif day_type == "saturday":
        to_work = 0.72 * _gauss(t, 9.6, 1.55) + 0.28 * _gauss(t, 11.5, 2.2) + 0.06
        to_home = 0.62 * _gauss(t, 15.5, 2.2) + 0.68 * _gauss(t, 19.6, 2.0) + 0.07
        other = 0.60 * _gauss(t, 13.5, 3.9) + 0.55 * _gauss(t, 19.5, 2.8) + 0.12
    elif day_type == "sunday":
        to_work = 0.45 * _gauss(t, 10.2, 2.4) + 0.06
        to_home = 0.55 * _gauss(t, 20.0, 2.5) + 0.30 * _gauss(t, 16.0, 3.0) + 0.07
        other = 0.85 * _gauss(t, 13.0, 4.2) + 0.75 * _gauss(t, 19.5, 2.9) + 0.15
    else:
        raise ValueError(f"unknown day_type {day_type!r}")

    # Nothing runs 00:00-04:00; zero it out before normalising.
    for arr in (to_work, to_home, other):
        arr[:4] = 0.0

    return {
        "to_work": to_work / to_work.sum(),
        "to_home": to_home / to_home.sum(),
        "other": other / other.sum(),
    }


#: Share of a day's trips by purpose, per day type.
PURPOSE_MIX: dict[str, dict[str, float]] = {
    "weekday": {"to_work": 0.415, "to_home": 0.415, "other": 0.170},
    "saturday": {"to_work": 0.300, "to_home": 0.300, "other": 0.400},
    "sunday": {"to_work": 0.150, "to_home": 0.170, "other": 0.680},
}

#: Total trips relative to a normal weekday.
DAY_TYPE_VOLUME: dict[str, float] = {"weekday": 1.00, "saturday": 0.78, "sunday": 0.46}

#: Day-of-week shading on top of that (Mon 0 ... Sun 6).
DOW_MULTIPLIER: np.ndarray = np.array([0.97, 1.01, 1.02, 1.02, 1.04, 1.00, 1.00])


# ---------------------------------------------------------------------------
# Destination choice
# ---------------------------------------------------------------------------

#: Distance-decay scale (km) in the gravity model.  Mumbai commutes are long:
#: a Panvel-to-CSMT trip is 49 km and entirely routine, so the decay is gentle.
DECAY_KM: float = 30.0


def destination_weights(route: str) -> dict[str, np.ndarray]:
    """Row-normalised destination-choice matrices for one route.

    Returns one ``(n_stops, n_stops)`` matrix per trip purpose; entry
    ``[i, j]`` is P(alight at j | board at i, purpose).  Diagonals are zero.
    """
    stops = ROUTES[route]
    st = load_stations().loc[stops]
    km = st["km"].to_numpy(dtype=float)
    pop = st["population_index"].to_numpy(dtype=float)
    emp = st["employment_index"].to_numpy(dtype=float)
    interchange = st["interchange"].to_numpy(dtype=bool)

    dist = np.abs(km[:, None] - km[None, :])
    decay = np.exp(-dist / DECAY_KM)
    np.fill_diagonal(decay, 0.0)
    # Nobody buys a ticket for one stop; suppress very short hops.
    decay *= 1.0 - np.exp(-dist / 2.2)

    # Interchanges soak up extra trips because they are also a gateway to the
    # Central/Western lines and the metro, not just a destination in themselves.
    gateway = 1.0 + 0.55 * interchange.astype(float)

    def _norm(w: np.ndarray) -> np.ndarray:
        w = np.maximum(w, 0.0)
        row = w.sum(axis=1, keepdims=True)
        return np.divide(w, row, out=np.zeros_like(w), where=row > 0)

    to_work = _norm(decay * (emp[None, :] ** 1.35) * gateway[None, :])
    to_home = _norm(decay * (pop[None, :] ** 1.25) * gateway[None, :])
    other = _norm(decay * ((0.5 * emp + 0.5 * pop)[None, :]) * gateway[None, :])
    return {"to_work": to_work, "to_home": to_home, "other": other}


def production_mass(route: str) -> dict[str, np.ndarray]:
    """Relative number of trips *starting* at each stop, per purpose."""
    stops = ROUTES[route]
    st = load_stations().loc[stops]
    pop = st["population_index"].to_numpy(dtype=float)
    emp = st["employment_index"].to_numpy(dtype=float)
    return {
        "to_work": pop / pop.sum(),
        "to_home": emp / emp.sum(),
        "other": (0.55 * pop + 0.45 * emp) / (0.55 * pop + 0.45 * emp).sum(),
    }


def day_type_of(ts: pd.Timestamp, is_holiday: bool) -> str:
    if is_holiday or ts.weekday() == 6:
        return "sunday"
    if ts.weekday() == 5:
        return "saturday"
    return "weekday"


__all__ = [
    "DAY_TYPE_VOLUME",
    "DECAY_KM",
    "DOW_MULTIPLIER",
    "PURPOSE_MIX",
    "day_type_of",
    "destination_weights",
    "hourly_profiles",
    "production_mass",
]
