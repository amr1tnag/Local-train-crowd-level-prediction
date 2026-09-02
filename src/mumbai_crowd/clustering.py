"""CO5 -- unsupervised clustering of Harbour-line station profiles.

The question
------------
Thirty-five stations, each with a 24-hour signature of boardings and
alightings.  Are there really thirty-five different kinds of station, or a
handful of *roles* that stations play?  If the latter, the roles are useful:
they compress the network for planners, they explain why crowding peaks where
it does, and -- tested at the end of this module -- they can be fed back into
the CO2 regression as a feature.

What is clustered
-----------------
Each station becomes one row of a profile vector combining

* its **shape**: normalised hourly boarding and alighting curves, which say
  *when* the station is a source and *when* it is a sink, independent of size;
* its **scale and asymmetry**: log volume, AM/PM peak shares, the
  boarding-to-alighting ratio in each peak, directional imbalance, weekend
  ratio, rain sensitivity, and how often passengers get left behind.

Normalising the curves before clustering is the important design decision.
Raw hourly counts would cluster stations by *size* -- Panvel and CSMT in one
group because both are big -- which is a fact we already knew from the
timetable.  Normalising makes the algorithm answer the interesting question:
which stations *behave* alike?

Honest limits
-------------
With n = 35 there is no test set and no bootstrap that creates information
that is not there.  Silhouette and Davies-Bouldin on 35 points are noisy, and
a two-cluster solution will often win on internal indices simply because it is
the easiest structure to separate.  The module therefore reports several
indices, a bootstrap stability score over resampled *days* (which is real
resampling -- the day-to-day variation is genuine), and a dendrogram, and
leaves the final k to a judgement call that is stated out loud rather than
laundered through a single number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import ClusterConfig
from .network import load_stations

#: Hours retained in the profile curves (the network is shut outside these).
PROFILE_HOURS: list[int] = list(range(4, 24))


# ---------------------------------------------------------------------------
# Profile construction
# ---------------------------------------------------------------------------


def build_station_profiles(flows: pd.DataFrame, weekdays_only: bool = True) -> pd.DataFrame:
    """One row per station describing its 24-hour behavioural signature."""
    df = flows.copy()
    if weekdays_only:
        df = df[(df["is_weekend"] == 0) & (df["is_holiday"] == 0)]
    df = df[df["hour"].isin(PROFILE_HOURS)]

    n_days = df["date"].nunique()
    per_hour = (
        df.groupby(["station_code", "hour"], observed=True)[["boardings", "alightings", "left_behind"]]
        .sum()
        .div(n_days)
        .reset_index()
    )

    board = per_hour.pivot(index="station_code", columns="hour", values="boardings").fillna(0.0)
    alight = per_hour.pivot(index="station_code", columns="hour", values="alightings").fillna(0.0)
    board = board.reindex(columns=PROFILE_HOURS, fill_value=0.0)
    alight = alight.reindex(columns=PROFILE_HOURS, fill_value=0.0)

    total_board = board.sum(axis=1)
    total_alight = alight.sum(axis=1)

    # Shape: what fraction of the day's boardings happen in each hour.
    board_shape = board.div(total_board.replace(0, np.nan), axis=0).fillna(0.0)
    alight_shape = alight.div(total_alight.replace(0, np.nan), axis=0).fillna(0.0)
    board_shape.columns = [f"board_h{h:02d}" for h in PROFILE_HOURS]
    alight_shape.columns = [f"alight_h{h:02d}" for h in PROFILE_HOURS]

    am = [h for h in PROFILE_HOURS if 7 <= h < 11]
    pm = [h for h in PROFILE_HOURS if 17 <= h < 22]

    def _share(frame: pd.DataFrame, hours: list[int], total: pd.Series) -> pd.Series:
        cols = [c for c in frame.columns if int(c) in hours]
        return frame[cols].sum(axis=1) / total.replace(0, np.nan)

    scalars = pd.DataFrame(index=board.index)
    scalars["log_daily_boardings"] = np.log1p(total_board)
    scalars["log_daily_alightings"] = np.log1p(total_alight)
    scalars["am_board_share"] = _share(board, am, total_board).fillna(0.0)
    scalars["pm_board_share"] = _share(board, pm, total_board).fillna(0.0)
    scalars["am_alight_share"] = _share(alight, am, total_alight).fillna(0.0)
    scalars["pm_alight_share"] = _share(alight, pm, total_alight).fillna(0.0)

    am_b, am_a = board[am].sum(axis=1), alight[am].sum(axis=1)
    pm_b, pm_a = board[pm].sum(axis=1), alight[pm].sum(axis=1)
    # >0 means the station is a net SOURCE in that peak (people leaving home);
    # <0 means a net SINK (people arriving at work).  This single number is the
    # clearest expression of a station's role in the tidal flow.
    scalars["am_net_source"] = (am_b - am_a) / (am_b + am_a).replace(0, np.nan)
    scalars["pm_net_source"] = (pm_b - pm_a) / (pm_b + pm_a).replace(0, np.nan)
    scalars["tidal_reversal"] = scalars["am_net_source"] - scalars["pm_net_source"]
    scalars = scalars.fillna(0.0)

    # Directional imbalance: does this station feed one direction only?
    dirn = (
        df.groupby(["station_code", "direction"], observed=True)["boardings"].sum().unstack(fill_value=0.0)
    )
    tot = dirn.sum(axis=1).replace(0, np.nan)
    scalars["up_board_share"] = (dirn.get("UP", 0.0) / tot).fillna(0.5)

    # Weekend and rain elasticity: measured, not assumed.
    all_days = flows[flows["hour"].isin(PROFILE_HOURS)]
    wk = all_days.groupby(["station_code", "is_weekend"], observed=True)["boardings"].mean().unstack()
    scalars["weekend_ratio"] = (wk.get(1) / wk.get(0)).reindex(scalars.index).fillna(1.0)

    wet = all_days.assign(wet=(all_days["rain_mm_hr"] > 1.0).astype(int))
    rn = wet.groupby(["station_code", "wet"], observed=True)["boardings"].mean().unstack()
    scalars["rain_ratio"] = (rn.get(1) / rn.get(0)).reindex(scalars.index).fillna(1.0)

    lb = df.groupby("station_code", observed=True)[["left_behind", "boardings"]].sum()
    scalars["left_behind_rate"] = (lb["left_behind"] / lb["boardings"].replace(0, np.nan)).fillna(0.0)

    profiles = pd.concat([board_shape, alight_shape, scalars], axis=1)

    st = load_stations()
    meta = st.loc[profiles.index, ["name", "km", "branch", "role", "interchange"]]
    profiles = pd.concat([meta, profiles], axis=1)
    profiles.index.name = "station_code"
    return profiles


#: Columns that are metadata rather than clustering inputs.
META_COLUMNS: list[str] = ["name", "km", "branch", "role", "interchange"]


def profile_matrix(profiles: pd.DataFrame, standardize: bool = True) -> tuple[np.ndarray, list[str], StandardScaler | None]:
    """Numeric matrix ready for clustering, with the metadata stripped off."""
    cols = [c for c in profiles.columns if c not in META_COLUMNS]
    X = profiles[cols].to_numpy(dtype=float)
    scaler = None
    if standardize:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    return X, cols, scaler


# ---------------------------------------------------------------------------
# Choosing k
# ---------------------------------------------------------------------------


def select_k(X: np.ndarray, cfg: ClusterConfig | None = None) -> pd.DataFrame:
    """Internal validation indices across a range of k.

    Four indices disagreeing is the normal outcome, not a problem to hide:
    inertia always falls, silhouette usually favours small k, Calinski-Harabasz
    favours compact well-separated blobs and Davies-Bouldin is the only one
    where lower is better.  They are reported together so the choice of k can
    be argued rather than asserted.
    """
    cfg = cfg or ClusterConfig()
    rows = []
    for k in cfg.k_range:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, n_init=cfg.n_init, random_state=cfg.random_state).fit(X)
        labels = km.labels_
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X, labels)) if k > 1 else np.nan,
                "davies_bouldin": float(davies_bouldin_score(X, labels)) if k > 1 else np.nan,
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)) if k > 1 else np.nan,
            }
        )
    out = pd.DataFrame(rows).set_index("k")
    # Elbow via the "distance to the chord" rule, stated explicitly so the
    # elbow is a computation rather than a squint at a chart.
    ks = out.index.to_numpy(dtype=float)
    inertia = out["inertia"].to_numpy()
    p1, p2 = np.array([ks[0], inertia[0]]), np.array([ks[-1], inertia[-1]])
    d = p2 - p1
    d = d / np.linalg.norm(d)
    pts = np.column_stack([ks, inertia]) - p1
    out["elbow_distance"] = np.linalg.norm(pts - np.outer(pts @ d, d), axis=1)
    return out


def gmm_selection(X: np.ndarray, cfg: ClusterConfig | None = None) -> pd.DataFrame:
    """BIC / AIC for a Gaussian mixture, as a model-based second opinion on k."""
    cfg = cfg or ClusterConfig()
    rows = []
    for k in cfg.k_range:
        if k >= len(X):
            continue
        gm = GaussianMixture(
            n_components=k,
            covariance_type="diag",       # n=35 cannot support full covariances
            random_state=cfg.random_state,
            n_init=5,
            reg_covar=1e-4,
        ).fit(X)
        rows.append({"k": k, "bic": float(gm.bic(X)), "aic": float(gm.aic(X)),
                     "loglik": float(gm.score(X) * len(X))})
    return pd.DataFrame(rows).set_index("k")


def bootstrap_stability(
    flows: pd.DataFrame, k: int, cfg: ClusterConfig | None = None
) -> dict[str, float]:
    """How stable is the partition if we had observed different days?

    Resamples *days* with replacement, rebuilds the profiles from scratch, and
    measures the adjusted Rand index against the full-data partition.  This is
    the only honest uncertainty statement available here: the 35 stations are
    the population, but the days are a sample.
    """
    cfg = cfg or ClusterConfig()
    rng = np.random.default_rng(cfg.random_state)

    base_profiles = build_station_profiles(flows)
    Xb, _, _ = profile_matrix(base_profiles)
    base = KMeans(n_clusters=k, n_init=cfg.n_init, random_state=cfg.random_state).fit_predict(Xb)

    dates = flows["date"].unique()
    scores = []
    for _ in range(cfg.bootstrap_runs):
        picked = rng.choice(dates, size=len(dates), replace=True)
        counts = pd.Series(picked).value_counts()
        sub = flows[flows["date"].isin(counts.index)].copy()
        sub["_w"] = sub["date"].map(counts).astype(float)
        for c in ("boardings", "alightings", "left_behind"):
            sub[c] = sub[c] * sub["_w"]
        prof = build_station_profiles(sub)
        Xs, _, _ = profile_matrix(prof.reindex(base_profiles.index))
        lab = KMeans(n_clusters=k, n_init=cfg.n_init, random_state=cfg.random_state).fit_predict(Xs)
        scores.append(adjusted_rand_score(base, lab))
    scores = np.asarray(scores)
    return {
        "k": k,
        "ari_mean": float(scores.mean()),
        "ari_std": float(scores.std()),
        "ari_p05": float(np.quantile(scores, 0.05)),
        "n_runs": int(len(scores)),
    }


# ---------------------------------------------------------------------------
# Fitting and interpreting
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    labels: pd.Series
    model: object
    X: np.ndarray
    columns: list[str]
    pca: PCA
    coords: np.ndarray
    silhouette: float
    silhouette_per_station: pd.Series
    profiles: pd.DataFrame
    names: dict[int, str] = field(default_factory=dict)

    @property
    def named_labels(self) -> pd.Series:
        return self.labels.map(lambda i: self.names.get(int(i), f"cluster_{i}"))


def fit_clusters(
    profiles: pd.DataFrame, k: int, cfg: ClusterConfig | None = None, method: str = "kmeans"
) -> ClusterResult:
    """Fit the chosen partition and attach everything needed to interpret it."""
    cfg = cfg or ClusterConfig()
    X, cols, _ = profile_matrix(profiles)

    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=cfg.n_init, random_state=cfg.random_state).fit(X)
        labels = model.labels_
    elif method == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        labels = model.labels_
    elif method == "gmm":
        model = GaussianMixture(
            n_components=k, covariance_type="diag", random_state=cfg.random_state,
            n_init=5, reg_covar=1e-4,
        ).fit(X)
        labels = model.predict(X)
    else:
        raise ValueError(f"unknown method {method!r}")

    pca = PCA(n_components=min(cfg.pca_components, X.shape[1])).fit(X)
    coords = pca.transform(X)
    sil = float(silhouette_score(X, labels))
    sil_i = pd.Series(silhouette_samples(X, labels), index=profiles.index, name="silhouette")

    result = ClusterResult(
        labels=pd.Series(labels, index=profiles.index, name="cluster"),
        model=model, X=X, columns=cols, pca=pca, coords=coords,
        silhouette=sil, silhouette_per_station=sil_i, profiles=profiles,
    )
    result.names = name_clusters(result)
    return result


def dendrogram_linkage(profiles: pd.DataFrame) -> np.ndarray:
    X, _, _ = profile_matrix(profiles)
    return linkage(X, method="ward")


def cut_dendrogram(Z: np.ndarray, k: int) -> np.ndarray:
    return fcluster(Z, t=k, criterion="maxclust") - 1


def dbscan_scan(X: np.ndarray, eps_grid: np.ndarray | None = None, min_samples: int = 3) -> pd.DataFrame:
    """DBSCAN across a range of eps, as a density-based sanity check.

    Included because it is the one method here that can say "these three
    stations belong to no cluster at all", and on a network with genuine
    oddities (a dock station with almost no traffic, a terminus that is pure
    sink) that answer deserves a hearing.
    """
    if eps_grid is None:
        eps_grid = np.linspace(2.0, 9.0, 29)
    rows = []
    for eps in eps_grid:
        lab = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(X)
        n_clusters = len({int(l) for l in lab if l >= 0})
        n_noise = int((lab < 0).sum())
        sil = np.nan
        if n_clusters > 1 and n_noise < len(X):
            m = lab >= 0
            if len(set(lab[m])) > 1:
                sil = float(silhouette_score(X[m], lab[m]))
        rows.append({"eps": float(eps), "n_clusters": n_clusters, "n_noise": n_noise, "silhouette": sil})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Naming and summarising
# ---------------------------------------------------------------------------


def cluster_summary(result: ClusterResult) -> pd.DataFrame:
    """Per-cluster means of the interpretable scalar features."""
    keep = [
        "log_daily_boardings", "am_board_share", "pm_board_share",
        "am_net_source", "pm_net_source", "tidal_reversal",
        "up_board_share", "weekend_ratio", "rain_ratio", "left_behind_rate",
    ]
    df = result.profiles.copy()
    df["cluster"] = result.labels
    agg = df.groupby("cluster")[keep].mean()
    agg["n_stations"] = df.groupby("cluster").size()
    agg["mean_km"] = df.groupby("cluster")["km"].mean()
    agg["interchange_share"] = df.groupby("cluster")["interchange"].mean()
    agg["stations"] = df.groupby("cluster").apply(
        lambda g: ", ".join(g["name"].tolist()), include_groups=False
    )
    return agg


def name_clusters(result: ClusterResult) -> dict[int, str]:
    """Attach a human-readable role to each cluster from its centroid.

    Rule-based rather than hand-written, so that re-running with a different k
    or a different seed does not silently leave the labels lying about what
    the clusters contain.
    """
    s = cluster_summary(result)
    names: dict[int, str] = {}

    # A volume qualifier is attached only when the cluster is genuinely at one
    # end of the range.  A median split would force every cluster into "high"
    # or "low" and end up calling a group containing Andheri, Bandra and Kurla
    # low-volume simply because two clusters have to be on that side -- and a
    # fifteen-station cluster spanning Andheri to Chunabhatti does not have a
    # meaningful single volume anyway.  When in doubt, say nothing: the
    # behavioural half of the label is the informative half.
    vol = s["log_daily_boardings"]
    spread = float(vol.std(ddof=0))
    z = (vol - vol.mean()) / (spread if spread > 1e-9 else 1.0)

    for cid, row in s.iterrows():
        am_source = row["am_net_source"]
        pm_source = row["pm_net_source"]
        interchange = row["interchange_share"] >= 0.4

        if am_source > 0.25 and pm_source < -0.1:
            base = "morning-source dormitory"
        elif am_source < -0.25 and pm_source > 0.1:
            base = "employment sink"
        elif interchange:
            base = "interchange churn"
        elif abs(am_source) <= 0.25 and abs(pm_source) <= 0.25:
            base = "balanced mixed-use"
        elif am_source > 0:
            base = "net residential"
        else:
            base = "net commercial"

        zi = float(z.loc[cid])
        if zi > 0.8:
            names[int(cid)] = f"high-volume {base}"
        elif zi < -0.8:
            names[int(cid)] = f"low-volume {base}"
        else:
            names[int(cid)] = base

    # Disambiguate collisions so two clusters never share a label.
    seen: dict[str, int] = {}
    for cid in sorted(names):
        label = names[cid]
        if label in seen:
            seen[label] += 1
            names[cid] = f"{label} ({seen[label]})"
        else:
            seen[label] = 1
    return names


def station_assignment_table(result: ClusterResult) -> pd.DataFrame:
    out = result.profiles[META_COLUMNS].copy()
    out["cluster"] = result.labels
    out["cluster_name"] = result.named_labels
    out["silhouette"] = result.silhouette_per_station
    for c in ("log_daily_boardings", "am_net_source", "pm_net_source", "left_behind_rate"):
        out[c] = result.profiles[c]
    return out.sort_values(["cluster", "km"])


def method_agreement(profiles: pd.DataFrame, k: int, cfg: ClusterConfig | None = None) -> pd.DataFrame:
    """Adjusted Rand index between k-means, Ward and a Gaussian mixture.

    If three quite different algorithms recover the same partition, the
    structure is in the data rather than in the algorithm -- which is the only
    external validation available when there are no ground-truth labels.
    """
    cfg = cfg or ClusterConfig()
    labs = {m: fit_clusters(profiles, k, cfg, method=m).labels.to_numpy() for m in ("kmeans", "ward", "gmm")}
    names = list(labs)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            M.loc[a, b] = adjusted_rand_score(labs[a], labs[b])
    return M


__all__ = [
    "META_COLUMNS",
    "PROFILE_HOURS",
    "ClusterResult",
    "bootstrap_stability",
    "build_station_profiles",
    "cluster_summary",
    "cut_dendrogram",
    "dbscan_scan",
    "dendrogram_linkage",
    "fit_clusters",
    "gmm_selection",
    "method_agreement",
    "name_clusters",
    "profile_matrix",
    "select_k",
    "station_assignment_table",
]
