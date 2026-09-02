"""Physical sanity of the simulator.

If these fail, every downstream number in the project is meaningless, so they
check conservation and capacity rather than just "did it run".
"""
from __future__ import annotations

import numpy as np
import pytest

from mumbai_crowd.config import SimConfig, density_to_band
from mumbai_crowd.network import (
    CLASS_TO_POOL,
    ROUTES,
    coach_position_bias,
    load_stations,
    rake_layout,
    route_stations,
)
from mumbai_crowd.simulate import pool_structure, simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(SimConfig(n_days=4, monitored_service_fraction=0.35), verbose=False)


def test_tables_are_populated(sim):
    assert len(sim.coach_observations) > 1000
    assert len(sim.station_hour_flows) > 100
    assert len(sim.service_records) > 100


def test_no_missing_weather_after_the_post_midnight_fix(sim):
    """Late services arrive after midnight; their weather join must still hit."""
    for col in ("temp_c", "rain_mm_hr", "rain_3h_mm", "humidity_pct"):
        assert sim.coach_observations[col].isna().sum() == 0


def test_density_is_non_negative_and_physically_bounded(sim):
    d = sim.coach_observations["density_true"]
    assert d.min() >= 0
    # 16 standees/m^2 is the physical ceiling; the per-service overload factor
    # tops out at 1.16, so nothing should exceed ~19 before sensor noise.
    assert d.max() <= 19.0


def test_coach_load_never_exceeds_capacity(sim):
    obs = sim.coach_observations
    caps = {}
    for n_cars in obs["n_cars"].unique():
        ps = pool_structure(int(n_cars))
        for spec in ps["layout"]:
            caps[(int(n_cars), spec.position)] = spec.seats + spec.crush_capacity
    limit = np.array([caps[(int(n), int(p))] for n, p in
                      zip(obs["n_cars"], obs["coach_position"])], dtype=float)
    # 1.16 is the overload clip in simulate.py, plus a rounding tolerance.
    assert (obs["load_depart"] <= limit * 1.16 + 1.0).all()


def test_passengers_are_conserved_along_a_run(sim):
    """Onboard load must equal cumulative boardings minus cumulative alightings."""
    sr = sim.service_records.sort_values(["date", "service_id", "station_seq"])
    g = sr.groupby(["date", "service_id"])
    net = g["boardings"].cumsum() - g["alightings"].cumsum()
    assert np.allclose(net.to_numpy(), sr["load_depart"].to_numpy(), atol=1.0)


def test_service_ids_are_unique_per_day(sim):
    """Regression test: service_id used to repeat across days."""
    sr = sim.service_records
    per_day = sr.groupby(["date", "service_id"]).size()
    assert (per_day <= sr["station_seq"].max() + 1).all()
    assert sr["service_id"].str[:8].nunique() == sr["date"].nunique()


def test_every_run_ends_empty(sim):
    sr = sim.service_records.sort_values(["date", "service_id", "station_seq"])
    last = sr.groupby(["date", "service_id"]).tail(1)
    assert last["load_depart"].max() < 1.0


def test_am_peak_flows_towards_the_cbd(sim):
    """The tidal reversal is the central structure; assert it exists."""
    obs = sim.coach_observations
    wd = obs[(obs["is_weekend"] == 0) & (obs["is_holiday"] == 0)]
    am = wd[wd["hour"].isin([8, 9, 10])]
    pm = wd[wd["hour"].isin([18, 19, 20])]
    assert am[am["direction"] == "UP"]["density_true"].mean() > \
           am[am["direction"] == "DN"]["density_true"].mean()
    assert pm[pm["direction"] == "DN"]["density_true"].mean() > \
           pm[pm["direction"] == "UP"]["density_true"].mean()


def test_load_peaks_mid_route_not_at_the_terminus(sim):
    """A cumulative-load model must peak before the sink, not at the origin."""
    sr = sim.service_records
    up = sr[(sr["direction"] == "UP") & (sr["hour"].isin([8, 9, 10]))]
    profile = up.groupby("station_seq")["load_depart"].mean()
    peak = int(profile.idxmax())
    assert 0 < peak < int(profile.index.max())


def test_dangerous_band_is_rare_but_present(sim):
    band = density_to_band(sim.coach_observations["density_depart"].to_numpy())
    share = float((band == 3).mean())
    assert 0.0005 < share < 0.10, f"dangerous share {share:.4%} is implausible"


def test_rain_increases_peak_crowding(sim):
    obs = sim.coach_observations
    peak = obs[obs["hour"].isin([8, 9, 10, 18, 19, 20])]
    wet = peak[peak["rain_mm_hr"] > 2.0]["density_true"].mean()
    dry = peak[peak["rain_mm_hr"] == 0.0]["density_true"].mean()
    assert wet > dry


def test_first_class_is_less_crowded_than_second(sim):
    obs = sim.coach_observations
    means = obs.groupby("coach_class")["density_true"].mean()
    assert means["first"] < means["second"]


def test_simulation_is_reproducible():
    a = simulate(SimConfig(n_days=2, seed=99), verbose=False).coach_observations
    b = simulate(SimConfig(n_days=2, seed=99), verbose=False).coach_observations
    assert a.shape == b.shape
    assert np.allclose(a["density_true"].to_numpy(), b["density_true"].to_numpy())


def test_different_seeds_give_different_data():
    a = simulate(SimConfig(n_days=2, seed=1), verbose=False).coach_observations
    b = simulate(SimConfig(n_days=2, seed=2), verbose=False).coach_observations
    assert not np.allclose(
        a["density_true"].to_numpy()[:100], b["density_true"].to_numpy()[:100]
    )


# --- network -------------------------------------------------------------

def test_station_master_is_consistent():
    st = load_stations()
    assert len(st) == 35
    assert st.index.is_unique
    for col in ("population_index", "employment_index"):
        assert st[col].between(0, 1).all()
    for route, stops in ROUTES.items():
        assert set(stops) <= set(st.index), f"{route} references unknown stations"
        km = st.loc[stops, "km"].to_numpy()
        assert (np.diff(km) > 0).all(), f"{route} is not monotone in chainage"


def test_route_direction_reverses_the_stop_order():
    dn = route_stations("CSMT_PNVL", "DN")
    up = route_stations("CSMT_PNVL", "UP")
    assert dn[0] == "CSMT" and dn[-1] == "PNVL"
    assert up == dn[::-1]


@pytest.mark.parametrize("n_cars", [12, 15])
def test_rake_layout_and_pools(n_cars):
    layout = rake_layout(n_cars)
    assert len(layout) == n_cars
    assert [c.position for c in layout] == list(range(1, n_cars + 1))
    assert all(c.coach_class in CLASS_TO_POOL for c in layout)
    assert all(c.crush_capacity > 0 and c.standing_area_m2 > 0 for c in layout)


def test_rake_layout_rejects_unknown_lengths():
    with pytest.raises(ValueError):
        rake_layout(9)


@pytest.mark.parametrize("fob", ["north", "centre", "south"])
def test_coach_position_bias_is_normalised_and_peaks_at_the_bridge(fob):
    bias = np.asarray(coach_position_bias(12, fob))
    assert bias.mean() == pytest.approx(1.0)
    assert (bias > 0).all()
    peak = int(np.argmax(bias))
    expected = {"south": 2, "centre": 5, "north": 9}[fob]
    assert abs(peak - expected) <= 1


def test_coach_position_bias_rejects_unknown_positions():
    with pytest.raises(ValueError):
        coach_position_bias(12, "east")
