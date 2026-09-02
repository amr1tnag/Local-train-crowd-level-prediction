"""Physical description of the Mumbai Suburban Railway *Harbour Line*.

Everything in this module is static reference data: the ordered station list,
approximate chainage (distance from CSMT in route-km), the catchment character
of each station, and the rake/coach layout of a Mumbai local.

Sources & caveats
-----------------
Station order, interchanges and approximate distances follow the public
Central Railway Harbour Line timetable.  The ``population_index`` and
``employment_index`` columns are *modelling assumptions*, not measured data:
they are ordinal scores (0-1) that encode how residential vs. how
job-dense a station's catchment is, and they are the knobs the demand
simulator in :mod:`mumbai_crowd.simulate` turns.  They are documented here so
that a reviewer can see -- and challenge -- every assumption in one place.

If you later obtain real UTS/ATVM ticketing counts or AFC gate data, you only
need to replace :func:`load_stations` and the simulator; every downstream
module consumes the resulting DataFrames, not this file.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import pandas as pd

Role = Literal["cbd", "commercial", "mixed", "dock", "residential", "suburb"]

# ---------------------------------------------------------------------------
# Station master
# ---------------------------------------------------------------------------
# columns: code, name, km (from CSMT), branch, role, interchange,
#          population_index (residential catchment, 0-1),
#          employment_index (jobs in catchment, 0-1),
#          platforms, fob_position (where the foot-over-bridge sits, which
#          decides which coaches get hammered -- see coach_position_bias)

_STATION_ROWS: list[tuple] = [
    # ---- Trunk: CSMT -> Vadala Road (shared by both Harbour services) ------
    ("CSMT", "Chhatrapati Shivaji Maharaj Terminus", 0.0, "trunk", "cbd", True, 0.10, 1.00, 18, "centre"),
    ("MSD", "Masjid", 1.4, "trunk", "commercial", False, 0.22, 0.72, 4, "south"),
    ("SNRD", "Sandhurst Road", 2.4, "trunk", "commercial", True, 0.28, 0.55, 4, "centre"),
    ("DKRD", "Dockyard Road", 3.5, "trunk", "dock", False, 0.24, 0.30, 2, "north"),
    ("RRD", "Reay Road", 4.5, "trunk", "dock", False, 0.20, 0.26, 2, "north"),
    ("CTGN", "Cotton Green", 5.5, "trunk", "dock", False, 0.18, 0.28, 2, "centre"),
    ("SVE", "Sewri", 7.0, "trunk", "mixed", False, 0.34, 0.38, 2, "south"),
    ("VDLR", "Vadala Road", 9.3, "trunk", "mixed", True, 0.46, 0.44, 6, "centre"),
    # ---- Panvel branch ----------------------------------------------------
    ("GTBN", "Guru Tegh Bahadur Nagar", 10.9, "panvel", "mixed", False, 0.44, 0.30, 2, "south"),
    ("CLA", "Chunabhatti", 12.5, "panvel", "mixed", False, 0.40, 0.26, 2, "north"),
    ("CLA_KRL", "Kurla", 15.5, "panvel", "mixed", True, 0.78, 0.62, 8, "centre"),
    ("TNA_TLK", "Tilak Nagar", 17.0, "panvel", "residential", False, 0.56, 0.20, 2, "south"),
    ("CMBR", "Chembur", 18.3, "panvel", "mixed", True, 0.66, 0.42, 4, "centre"),
    ("GVN", "Govandi", 20.0, "panvel", "residential", False, 0.62, 0.22, 2, "north"),
    ("MNKD", "Mankhurd", 21.8, "panvel", "residential", False, 0.58, 0.24, 4, "centre"),
    ("VSH", "Vashi", 26.0, "panvel", "mixed", True, 0.70, 0.66, 6, "centre"),
    ("SNPD", "Sanpada", 28.5, "panvel", "residential", False, 0.48, 0.30, 2, "south"),
    ("JNJ", "Juinagar", 30.4, "panvel", "residential", False, 0.50, 0.22, 2, "north"),
    ("NRL", "Nerul", 32.6, "panvel", "mixed", True, 0.72, 0.48, 6, "centre"),
    ("SWDV", "Seawoods-Darave", 34.3, "panvel", "residential", False, 0.64, 0.34, 4, "centre"),
    ("BEPR", "Belapur CBD", 37.0, "panvel", "commercial", False, 0.58, 0.60, 4, "centre"),
    ("KHAG", "Kharghar", 41.0, "panvel", "residential", False, 0.76, 0.24, 4, "south"),
    ("MSVR", "Manasarovar", 43.5, "panvel", "residential", False, 0.42, 0.14, 2, "north"),
    ("KNDS", "Khandeshwar", 46.0, "panvel", "residential", False, 0.46, 0.16, 2, "centre"),
    ("PNVL", "Panvel", 49.5, "panvel", "suburb", True, 0.88, 0.40, 7, "centre"),
    # ---- Goregaon branch (Vadala Road -> Goregaon) -------------------------
    ("KCE", "King's Circle", 11.2, "goregaon", "mixed", False, 0.50, 0.40, 4, "south"),
    ("MM", "Mahim Junction", 13.5, "goregaon", "mixed", True, 0.54, 0.44, 6, "centre"),
    ("BA", "Bandra", 15.6, "goregaon", "commercial", True, 0.62, 0.82, 6, "centre"),
    ("KHAR", "Khar Road", 17.5, "goregaon", "mixed", False, 0.52, 0.54, 4, "north"),
    ("STC", "Santacruz", 19.3, "goregaon", "mixed", False, 0.58, 0.58, 4, "centre"),
    ("VLP", "Vile Parle", 21.2, "goregaon", "mixed", False, 0.56, 0.62, 4, "south"),
    ("ADH", "Andheri", 23.4, "goregaon", "commercial", True, 0.74, 0.94, 9, "centre"),
    ("JOS", "Jogeshwari", 25.6, "goregaon", "residential", False, 0.62, 0.36, 4, "north"),
    ("RMAR", "Ram Mandir", 27.0, "goregaon", "residential", False, 0.44, 0.30, 2, "centre"),
    ("GMN", "Goregaon", 28.6, "goregaon", "mixed", True, 0.70, 0.56, 6, "centre"),
]

_COLUMNS = [
    "code", "name", "km", "branch", "role", "interchange",
    "population_index", "employment_index", "platforms", "fob_position",
]

# Ordered station codes for each timetabled service pattern.
ROUTES: dict[str, list[str]] = {
    # CSMT <-> Panvel ("slow" Harbour service, all stops)
    "CSMT_PNVL": [
        "CSMT", "MSD", "SNRD", "DKRD", "RRD", "CTGN", "SVE", "VDLR",
        "GTBN", "CLA", "CLA_KRL", "TNA_TLK", "CMBR", "GVN", "MNKD",
        "VSH", "SNPD", "JNJ", "NRL", "SWDV", "BEPR", "KHAG", "MSVR",
        "KNDS", "PNVL",
    ],
    # CSMT <-> Goregaon
    "CSMT_GMN": [
        "CSMT", "MSD", "SNRD", "DKRD", "RRD", "CTGN", "SVE", "VDLR",
        "KCE", "MM", "BA", "KHAR", "STC", "VLP", "ADH", "JOS", "RMAR", "GMN",
    ],
}

# Share of daily services run on each pattern (Central Rly runs far more
# CSMT-Panvel services than CSMT-Goregaon ones).
ROUTE_SHARE: dict[str, float] = {"CSMT_PNVL": 0.68, "CSMT_GMN": 0.32}


def load_stations() -> pd.DataFrame:
    """Return the station master table indexed by station ``code``."""
    df = pd.DataFrame(_STATION_ROWS, columns=_COLUMNS)
    df = df.set_index("code", drop=False)
    return df


def route_stations(route: str, direction: str) -> list[str]:
    """Ordered stop list for ``route``.

    ``direction`` is ``"DN"`` for CSMT -> suburbs (the classic evening peak
    direction) and ``"UP"`` for suburbs -> CSMT (morning peak).
    """
    stops = list(ROUTES[route])
    return stops if direction == "DN" else stops[::-1]


# ---------------------------------------------------------------------------
# Rake / coach layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoachSpec:
    """Physical + service description of one coach of a local train."""

    position: int          # 1 = leading coach in the DN direction
    coach_class: str       # second | first | ladies | ladies_first | luggage
    seats: int
    standing_area_m2: float   # usable standing floor area
    crush_capacity: int       # standees at 16/m^2 ("super-dense crush load")
    boarding_weight: float    # relative attractiveness to a boarding passenger

    @property
    def total_capacity(self) -> int:
        return self.seats + self.crush_capacity


# Capacity, grounded in Indian Railways' own load taxonomy
# ------------------------------------------------------------------
# IR rates suburban EMU loading in standees per m^2 of standing floor:
#   ~6/m^2  "normal crush load"
#   ~10/m^2 "dense crush load"
#   ~14-16/m^2 "super-dense crush load"
# A 12-car Mumbai rake is routinely quoted as carrying between 4,500 and
# 6,000 people at super-dense crush, which is ~450-500 per coach -- far
# outside the Fruin/TCQSM scale used for Western metros, and the reason this
# project uses Mumbai-specific band edges (see config.CROWD_BANDS).
#
# ``crush_capacity`` below is the number of *standees* at 16/m^2, i.e. the
# physical ceiling.  ``boarding_weight`` is how attractive the coach is to a
# boarding passenger relative to a plain second-class coach: the luggage /
# divyang compartment is partly occupied by goods, vendors and reserved space,
# so it fills more slowly.
_SECOND = dict(seats=100, standing_area_m2=25.0, crush_capacity=400, boarding_weight=1.00)
_FIRST = dict(seats=76, standing_area_m2=18.0, crush_capacity=288, boarding_weight=1.00)
_LADIES = dict(seats=100, standing_area_m2=25.0, crush_capacity=400, boarding_weight=1.00)
_LADIES_FIRST = dict(seats=40, standing_area_m2=10.0, crush_capacity=160, boarding_weight=1.00)
_LUGGAGE = dict(seats=20, standing_area_m2=26.0, crush_capacity=416, boarding_weight=0.55)


def _spec(position: int, coach_class: str) -> CoachSpec:
    base = {
        "second": _SECOND,
        "first": _FIRST,
        "ladies": _LADIES,
        "ladies_first": _LADIES_FIRST,
        "luggage": _LUGGAGE,
    }[coach_class]
    return CoachSpec(position=position, coach_class=coach_class, **base)


# Approximate composition of Central Railway EMU rakes.  Positions are counted
# from the CSMT end of the rake.
_RAKE_LAYOUT: dict[int, list[str]] = {
    12: [
        "luggage", "ladies", "second", "first", "second", "second",
        "ladies", "second", "second", "ladies_first", "second", "luggage",
    ],
    15: [
        "luggage", "ladies", "second", "second", "first", "second", "second",
        "ladies", "second", "second", "first", "second", "ladies", "second",
        "luggage",
    ],
}


#: Coach classes that share a queue on the platform.  A woman heading for the
#: ladies compartment is not competing for space in a general coach, and a
#: first-class ticket is a different market again, so each pool has its own
#: demand share and its own capacity constraint.
CLASS_TO_POOL: dict[str, str] = {
    "second": "general",
    "luggage": "general",
    "ladies": "ladies",
    "ladies_first": "ladies_first",
    "first": "first",
}

#: Ordered demand pools and their share of total ridership.
POOL_NAMES: tuple[str, ...] = ("general", "ladies", "first", "ladies_first")


def rake_layout(n_cars: int) -> list[CoachSpec]:
    """Coach-by-coach description of a ``n_cars``-car rake."""
    if n_cars not in _RAKE_LAYOUT:
        raise ValueError(f"unsupported rake length {n_cars}; expected one of {sorted(_RAKE_LAYOUT)}")
    return [_spec(i + 1, cls) for i, cls in enumerate(_RAKE_LAYOUT[n_cars])]


def coach_position_bias(n_cars: int, fob_position: str) -> list[float]:
    """Relative attractiveness of each coach given where the FOB/exit is.

    Commuters cluster near the foot-over-bridge so they can get out fast, so
    coaches under the bridge routinely run 30-50% denser than coaches at the
    far end of the same train.  This is one of the strongest real-world
    drivers of *coach-level* crowding and is why a train-level average hides
    exactly the danger we care about.
    """
    centres = {"south": 0.20, "centre": 0.5, "north": 0.80}
    if fob_position not in centres:
        raise ValueError(f"unknown fob_position {fob_position!r}")
    mu = centres[fob_position]
    bias = []
    for i in range(n_cars):
        x = (i + 0.5) / n_cars
        # Gaussian pull towards the bridge, floored so no coach is ever empty.
        bias.append(0.65 + 0.75 * pow(2.718281828, -0.5 * ((x - mu) / 0.28) ** 2))
    total = sum(bias)
    return [b * n_cars / total for b in bias]


def station_table_as_records() -> list[dict]:
    """Plain-python view of the station master (handy for tests/notebooks)."""
    return load_stations().to_dict(orient="records")


__all__ = [
    "CLASS_TO_POOL",
    "POOL_NAMES",
    "CoachSpec",
    "ROUTES",
    "ROUTE_SHARE",
    "coach_position_bias",
    "load_stations",
    "rake_layout",
    "route_stations",
    "station_table_as_records",
]
