"""
Station metadata for the Mumbai Harbour Line (CSMT <-> Panvel corridor).

Source / provenance
--------------------
Station names, order, and interchange status were verified by web search
against Wikipedia's "Harbour line (Mumbai Suburban Railway)" article and
corroborating sources (NAVITIME transit stop lists, MumbaiLocal.Info route
guide) on 2026-09-03. Direct fetch of the Wikipedia page itself was blocked
by this environment's network egress policy (see DATA_GENERATION.md), so
the list below is reconstructed from search-result summaries of that page
and cross-checked against two independent secondary sources.

Scope / simplification
-----------------------
The real Harbour Line network has ~35 stations across THREE branches
(CSMT-Panvel, CSMT-Goregaon via Mahim/Andheri, and the Trans-Harbour
Vashi-Thane line). This project models only the classic CSMT <-> Panvel
trunk (the original, highest-ridership Harbour Line corridor) — 25
stations. Stations that only exist on the other branches (e.g. King's
Circle, Mahim, Andheri) are intentionally excluded even though they are
colloquially considered part of "the Harbour Line", because they are not
reachable from CSMT without a direction change at Wadala Road.

Note on the project brief: the brief's example station list included
"Vidyavihar" and "King's Circle". Verification found Vidyavihar is on the
Central Line (between Kurla and Ghatkopar), not the Harbour Line, so it is
excluded. King's Circle is on the CSMT-Goregaon branch, not the CSMT-Panvel
trunk modeled here, so it is also excluded. Both corrections are called
out here explicitly since the brief asked for defensibility to an examiner.

`distance_km` is an approximate cumulative distance from CSMT, assembled
from general public route-distance figures (the ~54 km CSMT-Panvel trunk
length is well-documented; individual inter-station spacing is
interpolated proportionally along the route, not survey-grade GPS data).
It is used only for *relative* modeling (e.g. "how far has the train
filled up by the time it reaches this station"), never presented as an
authoritative distance.

`station_type` drives baseline ridership shape in simulate.py:
  - cbd_terminal      : CSMT itself - the demand sink/source for AM/PM peaks
  - industrial_dock    : old Bombay Port Trust dockland - low residential
                          density, low midday footfall, minimal peak sharpness
  - interchange_hub    : junction with another line/mode - elevated,
                          flatter all-day ridership (people passing through,
                          not just commuting home)
  - secondary_cbd      : Belapur CBD, Navi Mumbai's planned business
                          district - its own smaller AM-in/PM-out pattern
  - residential        : default suburban catchment - sharp AM/PM peaks,
                          quiet midday and late night
  - terminal_hub       : Panvel - line terminus AND interchange with
                          Central Line / long-distance services
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    station_id: str
    name: str
    order: int          # 0 = CSMT, increasing towards Panvel
    distance_km: float  # approx. cumulative distance from CSMT
    station_type: str
    is_interchange: bool


# order, name, distance_km (approx, cumulative), station_type, is_interchange
_RAW = [
    (0, "CSMT", 0.0, "cbd_terminal", True),
    (1, "Masjid", 1.2, "industrial_dock", False),
    (2, "Sandhurst Road", 2.3, "industrial_dock", False),
    (3, "Dockyard Road", 3.2, "industrial_dock", False),
    (4, "Reay Road", 4.0, "industrial_dock", False),
    (5, "Cotton Green", 5.0, "industrial_dock", False),
    (6, "Sewri", 6.4, "industrial_dock", False),
    (7, "Wadala Road", 8.4, "interchange_hub", True),   # + Monorail
    (8, "GTB Nagar", 9.6, "residential", False),
    (9, "Chunabhatti", 10.8, "residential", False),
    (10, "Kurla", 12.6, "interchange_hub", True),        # + Central Line
    (11, "Tilak Nagar", 14.0, "residential", False),
    (12, "Chembur", 16.2, "residential", False),
    (13, "Govandi", 18.4, "residential", False),
    (14, "Mankhurd", 20.6, "residential", False),
    (15, "Vashi", 24.8, "interchange_hub", True),        # Navi Mumbai hub
    (16, "Sanpada", 26.6, "residential", False),
    (17, "Juinagar", 28.6, "residential", False),
    (18, "Nerul", 31.0, "residential", False),
    (19, "Seawoods-Darave", 33.4, "residential", False),
    (20, "Belapur CBD", 36.4, "secondary_cbd", False),
    (21, "Kharghar", 40.0, "residential", False),
    (22, "Mansarovar", 42.4, "residential", False),
    (23, "Khandeshwar", 44.6, "residential", False),
    (24, "Panvel", 47.6, "terminal_hub", True),          # + Central Line
]

STATIONS: list[Station] = [
    Station(
        station_id=f"S{order:02d}",
        name=name,
        order=order,
        distance_km=dist,
        station_type=stype,
        is_interchange=interchange,
    )
    for order, name, dist, stype, interchange in _RAW
]

STATION_BY_ID = {s.station_id: s for s in STATIONS}
MAX_DISTANCE_KM = max(s.distance_km for s in STATIONS)

STATION_TYPES = sorted({s.station_type for s in STATIONS})
