"""Mesoscopic simulation of Harbour-line operations.

This module turns the static network (:mod:`mumbai_crowd.network`), the demand
model (:mod:`mumbai_crowd.demand`), the weather generator
(:mod:`mumbai_crowd.weather`) and the calendar (:mod:`mumbai_crowd.calendar_in`)
into the three tables the rest of the project consumes:

``coach_observations``   one row per (monitored service, station, instrumented
                         coach).  The CO2 supervised-learning table.  Target is
                         ``density_depart`` in passengers per m^2 of standing
                         floor.
``station_hour_flows``   one row per (date, station, hour, direction) with
                         boardings, alightings and left-behind counts summed
                         over *every* service.  The CO5 clustering table.
``service_records``      one row per (monitored service, station) at train
                         level; diagnostics and plots.

Honest labelling
----------------
There is no public coach-level occupancy dataset for the Mumbai suburban
network, so this data is **synthetic**.  It is not a random draw dressed up as
data: it is the output of a queueing process whose parameters are set from
published network facts and stated assumptions, and every assumption lives in
:mod:`mumbai_crowd.network`, :mod:`mumbai_crowd.demand` or
:class:`mumbai_crowd.config.SimConfig`.  Conclusions about *model behaviour
under asymmetric loss* transfer to real data; specific numbers do not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar_in import build_calendar
from .config import SimConfig, density_to_band
from .demand import (
    DAY_TYPE_VOLUME,
    DOW_MULTIPLIER,
    PURPOSE_MIX,
    day_type_of,
    destination_weights,
    hourly_profiles,
    production_mass,
)
from .network import (
    CLASS_TO_POOL,
    POOL_NAMES,
    ROUTE_SHARE,
    ROUTES,
    coach_position_bias,
    load_stations,
    rake_layout,
    route_stations,
)
from .weather import (
    generate_weather,
    rain_demand_multiplier,
    rain_headway_multiplier,
)

#: Modelled daily trips across the two Harbour-line service patterns.  The
#: Harbour Line is usually quoted at 1.1-1.5 million journeys a day.
TOTAL_DAILY_TRIPS: float = 1_450_000.0

#: Demand pools (see :data:`mumbai_crowd.network.POOL_NAMES`).
POOLS: tuple[str, ...] = POOL_NAMES

#: Mean tolerance for loading past the 16 standees/m^2 physical ceiling.  It is
#: drawn per service, because how far a rake is allowed to overload really does
#: vary with the motorman, the guard and how much the RPF is watching.
OVERLOAD_MEAN: float = 1.02
OVERLOAD_SD: float = 0.06


@dataclass
class SimulationOutput:
    coach_observations: pd.DataFrame
    station_hour_flows: pd.DataFrame
    service_records: pd.DataFrame
    weather: pd.DataFrame
    calendar: pd.DataFrame

    def summary(self) -> str:
        c = self.coach_observations
        return (
            f"coach_observations : {len(c):,} rows x {c.shape[1]} cols\n"
            f"station_hour_flows : {len(self.station_hour_flows):,} rows\n"
            f"service_records    : {len(self.service_records):,} rows\n"
            f"date range         : {c['date'].min().date()} .. {c['date'].max().date()}\n"
            f"mean density       : {c['density_depart'].mean():.2f} pax/m2\n"
            f"P(DANGEROUS)       : {(c['crowd_band'] == 3).mean():.3%}"
        )


# ---------------------------------------------------------------------------
# Timetable
# ---------------------------------------------------------------------------


def _headway_minutes(hour: float, day_type: str, route: str, cfg: SimConfig) -> float:
    am_peak = 7.25 <= hour < 11.0
    pm_peak = 16.75 <= hour < 21.5
    base = cfg.peak_headway_min if (am_peak or pm_peak) else cfg.offpeak_headway_min
    if hour < 6.0 or hour >= 22.5:
        base *= 1.7
    if day_type == "saturday":
        base *= 1.20
    elif day_type == "sunday":
        base *= 1.45
    # Split the line's total frequency between the two service patterns.
    return base / ROUTE_SHARE[route]


def _build_timetable(
    date: pd.Timestamp,
    route: str,
    direction: str,
    day_type: str,
    cfg: SimConfig,
    wx_day: pd.DataFrame,
    megablock: tuple[float, float] | None,
    disruption: tuple[float, float, float] | None,
    rng: np.random.Generator,
) -> list[dict]:
    """Departure times (minutes past midnight) from the origin terminus."""
    rain_by_hour = wx_day.set_index("hour")["rain_mm_hr"].to_dict()
    rain3_by_hour = wx_day.set_index("hour")["rain_3h_mm"].to_dict()

    t = cfg.service_start_hour * 60.0
    end = cfg.service_end_hour * 60.0
    services: list[dict] = []
    while t < end:
        hour = int(t // 60) % 24
        headway = _headway_minutes(t / 60.0, day_type, route, cfg)
        headway *= float(
            rain_headway_multiplier(rain_by_hour.get(hour, 0.0), rain3_by_hour.get(hour, 0.0))
        )
        if megablock is not None and megablock[0] <= t / 60.0 < megablock[1]:
            headway *= 2.4
        if disruption is not None and disruption[0] <= t / 60.0 < disruption[1]:
            headway *= disruption[2]
        headway *= float(np.clip(rng.normal(1.0, 0.10), 0.65, 1.6))

        n_cars = int(rng.choice(cfg.rake_lengths, p=cfg.rake_length_weights))
        rain_now = rain_by_hour.get(hour, 0.0)
        rain3_now = rain3_by_hour.get(hour, 0.0)
        # Delay: exponential tail, fattened by rain and by peak-hour congestion.
        delay_scale = 1.1 + 0.28 * rain_now + 0.035 * rain3_now
        if 7.25 <= t / 60.0 < 11.0 or 16.75 <= t / 60.0 < 21.5:
            delay_scale *= 1.7
        delay = float(rng.exponential(delay_scale))
        if disruption is not None and disruption[0] <= t / 60.0 < disruption[1]:
            delay += float(rng.exponential(9.0))

        services.append(
            {
                "dep_min": t,
                "n_cars": n_cars,
                "delay_min": round(delay, 2),
                "monitored": bool(rng.random() < cfg.monitored_service_fraction),
            }
        )
        t += headway

    # Date-qualified so that grouping by service_id across a multi-day dataset
    # cannot silently merge the 09:12 Panvel local of one Tuesday with the next.
    stamp = date.strftime("%Y%m%d")
    for i, s in enumerate(services):
        s["service_id"] = f"{stamp}_{route}_{direction}_{i:04d}"
    return services


# ---------------------------------------------------------------------------
# Route pre-computation
# ---------------------------------------------------------------------------


class _RouteGeometry:
    """Everything about a (route, direction) that does not change day to day."""

    def __init__(self, route: str, direction: str, cfg: SimConfig):
        self.route = route
        self.direction = direction
        self.stops = route_stations(route, direction)
        self.n = len(self.stops)

        st = load_stations()
        base_order = ROUTES[route]
        perm = [base_order.index(code) for code in self.stops]
        self.perm = np.asarray(perm)

        sub = st.loc[self.stops]
        self.km = sub["km"].to_numpy(dtype=float)
        self.fob = sub["fob_position"].tolist()
        self.station_codes = self.stops

        # Cumulative running time from the origin terminus, in minutes.
        seg = np.abs(np.diff(self.km, prepend=self.km[0]))
        run = np.cumsum(seg) / cfg.run_speed_kmph * 60.0
        dwell = np.arange(self.n) * cfg.dwell_seconds / 60.0
        self.travel_min = run + dwell

        # Destination-choice matrices, permuted into travel order and made
        # upper-triangular (you can only alight ahead of where you boarded).
        # FOB bias is a function of (rake length, station) only, so cache it:
        # it is otherwise recomputed a million times in the inner loop.
        self.coach_bias = {
            n_cars: [np.asarray(coach_position_bias(n_cars, f)) for f in self.fob]
            for n_cars in cfg.rake_lengths
        }

        W = destination_weights(route)
        M = production_mass(route)
        tri = np.triu(np.ones((self.n, self.n)), k=1)
        self.W = {p: W[p][np.ix_(perm, perm)] * tri for p in W}
        self.mass = {p: M[p][perm] for p in M}


def _pool_structure(n_cars: int) -> dict:
    """Coach -> pool index map, seats, standing area and per-pool crush capacity."""
    layout = rake_layout(n_cars)
    pool_idx = np.array([POOLS.index(CLASS_TO_POOL[c.coach_class]) for c in layout])
    seats = np.array([c.seats for c in layout], dtype=float)
    area = np.array([c.standing_area_m2 for c in layout], dtype=float)
    cap = np.array([c.crush_capacity + c.seats for c in layout], dtype=float)
    bw = np.array([c.boarding_weight for c in layout], dtype=float)
    pool_cap = np.array([cap[pool_idx == p].sum() for p in range(len(POOLS))])
    return {
        "layout": layout, "pool_idx": pool_idx, "seats": seats, "area": area,
        "cap": cap, "boarding_weight": bw, "pool_cap": pool_cap,
    }


_POOL_CACHE: dict[int, dict] = {}


def pool_structure(n_cars: int) -> dict:
    if n_cars not in _POOL_CACHE:
        _POOL_CACHE[n_cars] = _pool_structure(n_cars)
    return _POOL_CACHE[n_cars]


def _allocate(total: float, weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Split ``total`` across coaches by ``weights``, respecting ``caps``."""
    if total <= 0 or weights.sum() <= 0:
        return np.zeros_like(weights)
    alloc = total * weights / weights.sum()
    for _ in range(3):
        over = np.maximum(alloc - caps, 0.0)
        spill = over.sum()
        if spill <= 1e-9:
            break
        alloc = np.minimum(alloc, caps)
        room = caps - alloc
        if room.sum() <= 1e-9:
            break
        alloc = alloc + spill * room / room.sum()
    return np.minimum(alloc, caps)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def simulate(cfg: SimConfig | None = None, verbose: bool = True) -> SimulationOutput:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")
    # Weather covers every hour, and one extra day, because the last locals of
    # the night arrive at their far terminus after midnight and must be joined
    # to *that* calendar day's weather, not the operating day's.
    wx = generate_weather(dates.union(dates[-1:] + pd.Timedelta(days=1)), range(24), rng)
    cal = build_calendar(dates)

    geometries = {
        (r, d): _RouteGeometry(r, d, cfg)
        for r in ROUTES
        for d in ("UP", "DN")
    }
    pool_share = np.array(
        [
            1.0 - cfg.ladies_share - cfg.first_class_share - cfg.ladies_first_share,
            cfg.ladies_share,
            cfg.first_class_share,
            cfg.ladies_first_share,
        ]
    )
    n_pools = len(POOLS)

    coach_rows: list[tuple] = []
    service_rows: list[tuple] = []
    flow_acc: dict[tuple, np.ndarray] = {}

    wx_by_date = {d: g for d, g in wx.groupby("date", sort=False)}
    cal_by_date = {r.date: r for r in cal.itertuples()}
    t0 = time.time()

    one_day = pd.Timedelta(days=1)
    for day_i, date in enumerate(dates):
        next_date = date + one_day
        cal_row = cal_by_date[date]
        wx_day = wx_by_date[date]
        day_type = day_type_of(date, bool(cal_row.is_holiday))
        profiles = hourly_profiles(day_type)
        mix = PURPOSE_MIX[day_type]

        day_volume = (
            TOTAL_DAILY_TRIPS
            * DAY_TYPE_VOLUME[day_type]
            * DOW_MULTIPLIER[cal_row.dow]
            * cfg.daily_riders_scale
            * float(np.clip(rng.normal(1.0, 0.045), 0.8, 1.25))
        )

        rain_h = wx_day.set_index("hour")["rain_mm_hr"]
        rain3_h = wx_day.set_index("hour")["rain_3h_mm"]
        wx_mult = pd.Series(
            rain_demand_multiplier(rain_h.to_numpy(), rain3_h.to_numpy()), index=rain_h.index
        )

        # Festival travel is concentrated in the evening and late night.
        fest = float(cal_row.festival_intensity)
        fest_mult = np.ones(24)
        if fest > 0:
            h = np.arange(24, dtype=float)
            fest_mult = 1.0 + fest * (
                0.55 * np.exp(-0.5 * ((h - 20.0) / 2.6) ** 2)
                + 0.35 * np.exp(-0.5 * ((h - 22.5) / 2.0) ** 2)
                + 0.12
            )

        megablock = None
        if day_type == "sunday" and rng.random() < cfg.p_megablock_sunday:
            megablock = (10.5, 15.5)
        disruption = None
        if rng.random() < cfg.p_service_disruption:
            start = float(rng.uniform(6.0, 21.0))
            disruption = (start, start + float(rng.uniform(0.75, 3.0)), float(rng.uniform(1.4, 2.6)))

        for route in ROUTES:
            route_volume = day_volume * ROUTE_SHARE[route]
            for direction in ("UP", "DN"):
                geo = geometries[(route, direction)]
                n = geo.n

                # Hourly OD arrival-rate tensor (trips/hour), travel order.
                rate = np.zeros((24, n, n))
                for purpose in ("to_work", "to_home", "other"):
                    prof = profiles[purpose]
                    base = route_volume * mix[purpose] * geo.mass[purpose][:, None] * geo.W[purpose]
                    rate += prof[:, None, None] * base[None, :, :]
                for hh in range(24):
                    m = float(wx_mult.get(hh, 1.0)) * float(fest_mult[hh])
                    rate[hh] *= m

                services = _build_timetable(
                    date, route, direction, day_type, cfg, wx_day, megablock, disruption, rng
                )

                waiting = np.zeros((n_pools, n, n))
                last_seen = np.full(n, cfg.service_start_hour * 60.0)

                for svc in services:
                    n_cars = svc["n_cars"]
                    ps = pool_structure(n_cars)
                    pool_idx, seats, area = ps["pool_idx"], ps["seats"], ps["area"]
                    pool_cap, layout = ps["pool_cap"], ps["layout"]
                    overload = float(np.clip(rng.normal(OVERLOAD_MEAN, OVERLOAD_SD), 0.86, 1.16))
                    onboard = np.zeros((n_pools, n))
                    monitored = svc["monitored"]
                    if monitored:
                        coach_load = np.zeros(n_cars)
                        coach_effect = np.exp(rng.normal(0.0, 0.13, size=n_cars))
                        sensor_mask = rng.random(n_cars) < 0.30
                        if not sensor_mask.any():
                            sensor_mask[rng.integers(n_cars)] = True
                        coach_caps = ps["cap"] * overload
                        prev_density = np.full(n_cars, np.nan)
                        prev_load = np.full(n_cars, np.nan)
                    prev_left_behind = 0.0

                    for a in range(n):
                        sched_min = svc["dep_min"] + geo.travel_min[a]
                        t_arr = sched_min + svc["delay_min"] * (0.35 + 0.65 * a / max(n - 1, 1))
                        # The *timetabled* hour is what a planner knows in
                        # advance, so that is the one recorded as a feature;
                        # the actual arrival hour only drives demand accrual.
                        hour = int(sched_min // 60) % 24
                        wx_date = date if sched_min < 1440.0 else next_date
                        rate_hour = int(t_arr // 60) % 24
                        gap = max(t_arr - last_seen[a], 0.0)
                        last_seen[a] = t_arr

                        # --- platform accumulation since the previous train ---
                        lam = rate[rate_hour, a, :] * gap / 60.0
                        if lam.sum() > 0:
                            arrivals = rng.poisson(lam[None, :] * pool_share[:, None])
                            waiting[:, a, :] += arrivals

                        # --- alighting ---
                        alight_pool = onboard[:, a].copy()
                        onboard[:, a] = 0.0
                        load_before_board = onboard.sum(axis=1)

                        # --- boarding, capacity limited ---
                        w = waiting[:, a, :]
                        want = w.sum(axis=1)
                        cap_left = np.maximum(pool_cap * overload - load_before_board, 0.0)
                        frac = np.where(want > 0, np.minimum(1.0, cap_left / np.maximum(want, 1e-9)), 0.0)
                        onboard += w * frac[:, None]
                        waiting[:, a, :] = w * (1.0 - frac)[:, None]
                        board_pool = want * frac
                        left_pool = want - board_pool

                        boardings = float(board_pool.sum())
                        alightings = float(alight_pool.sum())
                        left_behind = float(left_pool.sum())
                        load_depart = float(onboard.sum())

                        key = (date, wx_date, geo.station_codes[a], hour, direction)
                        acc = flow_acc.get(key)
                        if acc is None:
                            acc = np.zeros(6)
                            flow_acc[key] = acc
                        acc += (boardings, alightings, left_behind, load_depart, 1.0, gap)

                        if not monitored:
                            continue

                        # --- coach-level bookkeeping (instrumented rakes) -----
                        bias = geo.coach_bias[n_cars][a]
                        arriving_load = coach_load.copy()
                        for p in range(n_pools):
                            sel = pool_idx == p
                            if not sel.any():
                                continue
                            pool_now = coach_load[sel]
                            if alight_pool[p] > 0 and pool_now.sum() > 0:
                                out = alight_pool[p] * pool_now / pool_now.sum()
                                coach_load[sel] = np.maximum(pool_now - out, 0.0)
                            if board_pool[p] > 0:
                                wts = bias[sel] * coach_effect[sel] * ps["boarding_weight"][sel]
                                room = np.maximum(coach_caps[sel] - coach_load[sel], 0.0)
                                coach_load[sel] += _allocate(board_pool[p], wts, room)

                        standees = np.maximum(coach_load - seats, 0.0)
                        density_true = standees / area
                        # Sensor error: multiplicative and clipped, because a
                        # load cell or a CCTV head-count is a ratio estimator
                        # with a bounded failure mode, not an additive gaussian.
                        noise = np.exp(
                            np.clip(
                                rng.normal(0.0, cfg.density_measurement_noise, size=n_cars),
                                -2.5 * cfg.density_measurement_noise,
                                2.5 * cfg.density_measurement_noise,
                            )
                        )
                        density_obs = density_true * noise
                        arr_standees = np.maximum(arriving_load - seats, 0.0)
                        density_arrive = arr_standees / area

                        service_rows.append(
                            (
                                date, svc["service_id"], route, direction, n_cars,
                                geo.station_codes[a], a, float(geo.km[a]),
                                round(sched_min, 2), round(t_arr, 2), hour,
                                round(svc["delay_min"], 2), round(gap, 2),
                                round(boardings, 1), round(alightings, 1),
                                round(left_behind, 1), round(load_depart, 1),
                            )
                        )

                        for c in np.flatnonzero(sensor_mask):
                            spec = layout[c]
                            coach_rows.append(
                                (
                                    date, wx_date, svc["service_id"], route, direction, n_cars,
                                    geo.station_codes[a], a, float(geo.km[a]), geo.fob[a],
                                    round(sched_min, 2), hour,
                                    round(svc["delay_min"], 2), round(gap, 2),
                                    int(spec.position), spec.coach_class,
                                    round(float(bias[c]), 4),
                                    round(float(arriving_load[c]), 1),
                                    round(float(coach_load[c]), 1),
                                    round(float(density_arrive[c]), 4),
                                    round(float(density_true[c]), 4),
                                    round(float(density_obs[c]), 4),
                                    round(float(prev_density[c]), 4),
                                    round(float(prev_load[c]), 1),
                                    round(prev_left_behind, 1),
                                    round(boardings, 1), round(alightings, 1),
                                    round(left_behind, 1),
                                )
                            )
                        prev_density = density_obs
                        prev_load = coach_load.copy()
                        prev_left_behind = left_behind

        if verbose and (day_i + 1) % 15 == 0:
            el = time.time() - t0
            print(
                f"  simulated {day_i + 1:>4}/{cfg.n_days} days  "
                f"({el:6.1f}s, {len(coach_rows):,} coach rows)",
                flush=True,
            )

    coach_df = pd.DataFrame(
        coach_rows,
        columns=[
            "date", "wx_date", "service_id", "route", "direction", "n_cars",
            "station_code", "station_seq", "km_from_csmt", "fob_position",
            "sched_min", "hour", "service_delay_min", "headway_min",
            "coach_position", "coach_class", "coach_pos_bias",
            "load_arrive", "load_depart", "density_arrive",
            "density_true", "density_depart",
            "prev_station_density", "prev_station_load", "prev_station_left_behind",
            "boardings", "alightings", "left_behind",
        ],
    )
    service_df = pd.DataFrame(
        service_rows,
        columns=[
            "date", "service_id", "route", "direction", "n_cars",
            "station_code", "station_seq", "km_from_csmt",
            "sched_min", "actual_min", "hour",
            "service_delay_min", "headway_min",
            "boardings", "alightings", "left_behind", "load_depart",
        ],
    )

    flow_df = pd.DataFrame(
        [(k[0], k[1], k[2], k[3], k[4], *v) for k, v in flow_acc.items()],
        columns=[
            "date", "wx_date", "station_code", "hour", "direction",
            "boardings", "alightings", "left_behind", "load_depart_sum",
            "n_trains", "headway_sum",
        ],
    )
    flow_df["mean_load"] = flow_df["load_depart_sum"] / flow_df["n_trains"]
    flow_df["mean_headway_min"] = flow_df["headway_sum"] / flow_df["n_trains"]
    flow_df = flow_df.drop(columns=["load_depart_sum", "headway_sum"])

    # Attach calendar (keyed on the operating day) and weather (keyed on the
    # true wall-clock day, which differs for post-midnight arrivals).
    for df in (coach_df, service_df, flow_df):
        df["date"] = pd.to_datetime(df["date"])
    for df in (coach_df, flow_df):
        df["wx_date"] = pd.to_datetime(df["wx_date"])

    wx_join = wx.rename(columns={"date": "wx_date"})
    coach_df = coach_df.merge(cal, on="date", how="left").merge(
        wx_join, on=["wx_date", "hour"], how="left"
    )
    flow_df = flow_df.merge(cal, on="date", how="left").merge(
        wx_join, on=["wx_date", "hour"], how="left"
    )

    coach_df["crowd_band"] = density_to_band(coach_df["density_depart"].to_numpy())
    coach_df["is_dangerous"] = (coach_df["crowd_band"] == 3).astype(int)
    coach_df["timestamp"] = coach_df["date"] + pd.to_timedelta(coach_df["sched_min"], unit="m")
    coach_df = coach_df.sort_values(["timestamp", "service_id", "coach_position"], kind="stable")
    coach_df = coach_df.reset_index(drop=True)

    if verbose:
        print(f"  simulation finished in {time.time() - t0:.1f}s", flush=True)

    return SimulationOutput(
        coach_observations=coach_df,
        station_hour_flows=flow_df.sort_values(["date", "station_code", "hour"]).reset_index(drop=True),
        service_records=service_df,
        weather=wx,
        calendar=cal,
    )


__all__ = [
    "OVERLOAD_MEAN",
    "OVERLOAD_SD",
    "POOLS",
    "TOTAL_DAILY_TRIPS",
    "SimulationOutput",
    "pool_structure",
    "simulate",
]
