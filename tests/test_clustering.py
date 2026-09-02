"""CO5: profiles, partitions and the claims made about them."""
from __future__ import annotations

import numpy as np
import pytest

from mumbai_crowd.clustering import (
    META_COLUMNS,
    PROFILE_HOURS,
    build_station_profiles,
    cluster_summary,
    cut_dendrogram,
    dbscan_scan,
    dendrogram_linkage,
    fit_clusters,
    gmm_selection,
    method_agreement,
    profile_matrix,
    select_k,
    station_assignment_table,
)
from mumbai_crowd.config import ClusterConfig, SimConfig
from mumbai_crowd.network import load_stations
from mumbai_crowd.simulate import simulate


@pytest.fixture(scope="module")
def flows():
    return simulate(SimConfig(n_days=21, monitored_service_fraction=0.05), verbose=False).station_hour_flows


@pytest.fixture(scope="module")
def profiles(flows):
    return build_station_profiles(flows)


def test_every_station_gets_a_profile(profiles):
    assert len(profiles) == len(load_stations())
    assert profiles.index.is_unique


def test_profile_curves_are_normalised_shapes(profiles):
    board = [f"board_h{h:02d}" for h in PROFILE_HOURS]
    alight = [f"alight_h{h:02d}" for h in PROFILE_HOURS]
    assert np.allclose(profiles[board].sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(profiles[alight].sum(axis=1), 1.0, atol=1e-6)


def test_profiles_have_no_missing_values(profiles):
    assert profiles.drop(columns=META_COLUMNS).isna().sum().sum() == 0


def test_net_source_captures_the_tidal_role(profiles):
    """Panvel should be a morning source; CSMT should be a morning sink."""
    assert profiles.loc["PNVL", "am_net_source"] > 0.2
    assert profiles.loc["CSMT", "am_net_source"] < -0.2
    assert profiles.loc["CSMT", "pm_net_source"] > 0.2


def test_profile_matrix_is_standardised(profiles):
    X, cols, scaler = profile_matrix(profiles)
    assert X.shape == (len(profiles), len(cols))
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-9)
    assert not set(cols) & set(META_COLUMNS)


def test_select_k_returns_all_indices(profiles):
    X, _, _ = profile_matrix(profiles)
    out = select_k(X, ClusterConfig(k_range=tuple(range(2, 8))))
    assert list(out.index) == list(range(2, 8))
    for col in ("inertia", "silhouette", "davies_bouldin", "calinski_harabasz", "elbow_distance"):
        assert col in out.columns
    # Inertia must fall monotonically with k.
    assert (np.diff(out["inertia"].to_numpy()) < 0).all()


def test_gmm_selection_runs(profiles):
    X, _, _ = profile_matrix(profiles)
    out = gmm_selection(X, ClusterConfig(k_range=(2, 3, 4, 5)))
    assert set(out.columns) == {"bic", "aic", "loglik"}


@pytest.mark.parametrize("method", ["kmeans", "ward", "gmm"])
def test_fit_clusters_produces_a_complete_partition(profiles, method):
    res = fit_clusters(profiles, 5, method=method)
    assert len(res.labels) == len(profiles)
    assert res.labels.nunique() == 5
    assert -1.0 <= res.silhouette <= 1.0
    assert res.coords.shape == (len(profiles), 2)
    assert len(res.names) == 5


def test_unknown_method_is_rejected(profiles):
    with pytest.raises(ValueError):
        fit_clusters(profiles, 3, method="kohonen")


def test_cluster_names_are_unique_and_descriptive(profiles):
    res = fit_clusters(profiles, 5)
    names = list(res.names.values())
    assert len(set(names)) == len(names)
    assert all(isinstance(n, str) and len(n) > 5 for n in names)


def test_cluster_summary_accounts_for_every_station(profiles):
    res = fit_clusters(profiles, 5)
    s = cluster_summary(res)
    assert s["n_stations"].sum() == len(profiles)


def test_cbd_and_far_suburbs_never_share_a_cluster(profiles):
    """A partition that puts CSMT with Panvel has learned nothing."""
    res = fit_clusters(profiles, 5)
    assert res.labels["CSMT"] != res.labels["PNVL"]
    assert res.labels["CSMT"] != res.labels["KHAG"]


def test_dendrogram_cut_matches_the_requested_size(profiles):
    Z = dendrogram_linkage(profiles)
    assert Z.shape == (len(profiles) - 1, 4)
    labels = cut_dendrogram(Z, 5)
    assert len(set(labels)) == 5


def test_methods_broadly_agree(profiles):
    """If three algorithms disagree completely, the structure is not real."""
    M = method_agreement(profiles, 5)
    assert np.allclose(np.diag(M.to_numpy(dtype=float)), 1.0)
    assert float(M.loc["kmeans", "ward"]) > 0.3


def test_dbscan_scan_shape(profiles):
    X, _, _ = profile_matrix(profiles)
    out = dbscan_scan(X, np.linspace(3.0, 8.0, 6))
    assert len(out) == 6
    assert (out["n_noise"] <= len(X)).all()


def test_assignment_table_is_readable(profiles):
    res = fit_clusters(profiles, 5)
    t = station_assignment_table(res)
    assert len(t) == len(profiles)
    assert {"name", "cluster", "cluster_name", "silhouette"} <= set(t.columns)
