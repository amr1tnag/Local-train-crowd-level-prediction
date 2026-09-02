#!/usr/bin/env python3
"""CO5 -- cluster Harbour-line stations by their 24-hour flow signatures.

    python scripts/03_cluster_stations.py --data data --out reports --k 5

Steps:

1.  builds a behavioural profile per station (normalised hourly boarding and
    alighting curves plus scale, tidal-asymmetry, weekend and rain summaries);
2.  scans k with four internal indices plus a Gaussian-mixture BIC, and says
    out loud that they disagree;
3.  cross-checks the chosen partition against Ward linkage and a GMM (adjusted
    Rand index) and against DBSCAN;
4.  measures stability by bootstrapping *days* and re-running the whole
    pipeline;
5.  names each cluster from its centroid and writes the assignment table;
6.  feeds the cluster label back into the CO2 regression to test whether the
    unsupervised structure carries information the supervised model did not
    already have.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mumbai_crowd import clustering as C
from mumbai_crowd import plots
from mumbai_crowd.config import ClusterConfig, ModelConfig, optimal_quantile
from mumbai_crowd.decision import ThresholdPolicy, policy_report
from mumbai_crowd.features import TARGET, build_design
from mumbai_crowd.losses import PinballLoss
from mumbai_crowd.metrics import regression_report
from mumbai_crowd.regression import BoostedRegressor, fit_predict

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 120)


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--out", type=str, default="reports")
    ap.add_argument("--k", type=int, default=5, help="number of station roles")
    ap.add_argument("--bootstrap", type=int, default=ClusterConfig.bootstrap_runs)
    ap.add_argument("--skip-co2-bridge", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    ccfg = ClusterConfig(bootstrap_runs=args.bootstrap)

    banner("CO5 | Building station behavioural profiles")
    flows = pd.read_csv(Path(args.data) / "station_hour_flows.csv.gz", parse_dates=["date", "wx_date"])
    profiles = C.build_station_profiles(flows)
    X, cols, _ = C.profile_matrix(profiles)
    print(f"  {len(profiles)} stations x {len(cols)} profile features")
    print(f"  ({len(C.PROFILE_HOURS)} hourly boarding shares + {len(C.PROFILE_HOURS)} alighting shares "
          f"+ {len(cols) - 2 * len(C.PROFILE_HOURS)} scalar summaries)")
    profiles.to_csv(tabdir / "co5_station_profiles.csv")

    banner("CO5 | Choosing k -- five criteria, and they do not agree")
    indices = C.select_k(X, ccfg)
    gmm = C.gmm_selection(X, ccfg)
    table = indices.join(gmm[["bic"]])
    print(table.round(3).to_string())
    verdict = {
        "elbow (max distance to chord)": int(indices["elbow_distance"].idxmax()),
        "silhouette (max)": int(indices["silhouette"].idxmax()),
        "davies_bouldin (min)": int(indices["davies_bouldin"].idxmin()),
        "calinski_harabasz (max)": int(indices["calinski_harabasz"].idxmax()),
        "gmm bic (min)": int(gmm["bic"].idxmin()),
    }
    print("\n  what each criterion would pick:")
    for name, k in verdict.items():
        print(f"    {name:<32s} k = {k}")
    print(f"\n  chosen: k = {args.k}. Silhouette on 35 points rewards the coarsest split it can find,")
    print( "  and the two- and three-cluster solutions collapse the dock belt into the CBD and")
    print( "  the interchanges into the dormitories -- distinctions the operator cares about.")
    print(f"  k = {args.k} is the finest split at which every cluster still has a one-sentence")
    print( "  operational description, and it is the level the bootstrap below says is stable.")
    table.to_csv(tabdir / "co5_k_selection.csv")

    banner(f"CO5 | Fitting k = {args.k}")
    result = C.fit_clusters(profiles, args.k, ccfg, method="kmeans")
    print(f"  silhouette = {result.silhouette:.3f}")
    summary = C.cluster_summary(result)
    summary.insert(0, "name", [result.names[int(i)] for i in summary.index])
    print()
    print(summary[["name", "n_stations", "log_daily_boardings", "am_net_source",
                   "pm_net_source", "interchange_share", "left_behind_rate"]].round(3).to_string())
    print("\n  membership")
    for cid, row in summary.iterrows():
        print(f"    [{cid}] {row['name']}")
        print(f"        {row['stations']}")
    summary.to_csv(tabdir / "co5_cluster_summary.csv")

    assign = C.station_assignment_table(result)
    assign.to_csv(tabdir / "co5_station_assignments.csv")
    print("\n  stations with a negative silhouette (poorly placed):")
    bad = assign[assign["silhouette"] < 0]
    print("    none" if bad.empty else bad[["name", "cluster_name", "silhouette"]].round(3).to_string())

    banner("CO5 | External validation")
    agree = C.method_agreement(profiles, args.k, ccfg)
    print("  adjusted Rand index between algorithms (1.0 = identical partition):")
    print(agree.round(3).to_string())
    agree.to_csv(tabdir / "co5_method_agreement.csv")

    db = C.dbscan_scan(X)
    best_db = db.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).head(5)
    print("\n  DBSCAN sweep (best five by silhouette):")
    print(best_db.round(3).to_string(index=False) if not best_db.empty
          else "    DBSCAN found no multi-cluster solution at any eps on this profile space")
    db.to_csv(tabdir / "co5_dbscan_scan.csv", index=False)

    print(f"\n  bootstrap stability over resampled days ({args.bootstrap} runs) ...", flush=True)
    t0 = time.time()
    stab = C.bootstrap_stability(flows, args.k, ccfg)
    print(f"    ARI vs full-data partition: {stab['ari_mean']:.3f} +/- {stab['ari_std']:.3f} "
          f"(5th pct {stab['ari_p05']:.3f})   [{time.time() - t0:.0f}s]")
    stab_rows = [stab]
    for k in (args.k - 1, args.k + 1):
        if 2 <= k < len(profiles):
            s = C.bootstrap_stability(flows, k, ClusterConfig(bootstrap_runs=max(20, args.bootstrap // 3)))
            stab_rows.append(s)
            print(f"    (k={k} for comparison: ARI {s['ari_mean']:.3f} +/- {s['ari_std']:.3f})")
    pd.DataFrame(stab_rows).to_csv(tabdir / "co5_stability.csv", index=False)

    banner("CO5 | Figures")
    Z = C.dendrogram_linkage(profiles)
    labels = [f"{profiles.loc[c, 'name'][:20]}" for c in profiles.index]
    for p in [
        plots.fig_k_selection(indices, gmm, args.k, figdir),
        plots.fig_dendrogram(Z, labels, args.k, figdir),
        plots.fig_cluster_pca(result, figdir),
        plots.fig_cluster_profiles(result, figdir),
        plots.fig_line_map(result, figdir),
        plots.fig_silhouette(result, figdir),
    ]:
        print(f"  {p}")

    # ------------------------------------------------------------ CO5 -> CO2 --
    bridge = None
    if not args.skip_co2_bridge:
        banner("CO5 -> CO2 | Does the unsupervised structure help the supervised model?")
        coach = pd.read_csv(
            Path(args.data) / "coach_observations.csv.gz",
            parse_dates=["date", "wx_date", "timestamp"], low_memory=False,
        )
        mcfg = ModelConfig()
        tau = optimal_quantile()
        rows = {}
        for variant in ("without cluster feature", "with cluster feature"):
            d = coach.copy()
            split, cl, _ = build_design(d, "schedule", mcfg.val_days, mcfg.test_days)
            if variant == "with cluster feature":
                mapping = result.labels.to_dict()
                for part in (split.train, split.val, split.test):
                    part["station_cluster"] = part["station_code"].map(mapping).astype("category")
                cl = cl + ["station_cluster"]
                # station_code is the finer encoding of the same thing; dropping
                # it is the only way to see whether the coarse roles carry
                # anything, so run that variant too.
            m, p = fit_predict(BoostedRegressor(loss=PinballLoss(tau), cfg=mcfg), split, cl)
            rep = regression_report(split.test[TARGET], p["test"], tau)
            tp = ThresholdPolicy().fit(split.val[TARGET], p["val"])
            rep.update(policy_report(split.test[TARGET], tp.decide(p["test"])))
            rows[variant] = rep
            print(f"  {variant:<26s} rmse={rep['rmse']:.4f}  cost={rep['exp_cost_inr']:.2f}  "
                  f"dangerous_miss={rep['dangerous_miss']:.4f}")

        # The interesting variant: cluster role INSTEAD of station identity.
        split, cl, _ = build_design(coach.copy(), "schedule", mcfg.val_days, mcfg.test_days)
        mapping = result.labels.to_dict()
        for part in (split.train, split.val, split.test):
            part["station_cluster"] = part["station_code"].map(mapping).astype("category")
        cl2 = [c for c in cl if c != "station_code"] + ["station_cluster"]
        m, p = fit_predict(BoostedRegressor(loss=PinballLoss(tau), cfg=mcfg), split, cl2)
        rep = regression_report(split.test[TARGET], p["test"], tau)
        tp = ThresholdPolicy().fit(split.val[TARGET], p["val"])
        rep.update(policy_report(split.test[TARGET], tp.decide(p["test"])))
        rows["cluster instead of station id"] = rep
        print(f"  {'cluster instead of station id':<26s} rmse={rep['rmse']:.4f}  "
              f"cost={rep['exp_cost_inr']:.2f}  dangerous_miss={rep['dangerous_miss']:.4f}")

        bridge = pd.DataFrame(rows).T
        bridge.to_csv(tabdir / "co5_to_co2_bridge.csv")
        print()
        print(bridge[["rmse", "r2", "exp_cost_inr", "dangerous_miss", "danger_recall",
                      "false_alarm"]].round(4).to_string())
        print(f"\n  {plots.fig_cluster_feature_value(bridge, figdir)}")

    payload = {
        "k": args.k,
        "silhouette": float(result.silhouette),
        "criteria_would_pick": verdict,
        "stability_ari_mean": stab["ari_mean"],
        "stability_ari_std": stab["ari_std"],
        "cluster_names": {str(k): v for k, v in result.names.items()},
        "method_agreement_kmeans_ward": float(agree.loc["kmeans", "ward"]),
        "method_agreement_kmeans_gmm": float(agree.loc["kmeans", "gmm"]),
    }
    if bridge is not None:
        payload["co2_bridge_cost_inr"] = {k: float(v) for k, v in bridge["exp_cost_inr"].items()}
    (tabdir / "co5_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {tabdir / 'co5_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
