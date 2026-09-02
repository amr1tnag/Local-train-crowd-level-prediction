#!/usr/bin/env python3
"""CO2 -- train and compare crowd-density regressors under asymmetric loss.

    python scripts/02_train_regression.py --data data --out reports

What it does, in order:

1.  builds the design matrix with a strict temporal split and leakage guard;
2.  trains a model zoo that differs *only* in the loss function;
3.  scores everything on statistical and decision metrics side by side;
4.  fits cost-optimal alert thresholds on validation and applies them to test;
5.  fits a quantile ensemble and takes the Bayes-optimal action from it;
6.  sweeps the assumed cost ratio to show it is a policy dial, not a magic number;
7.  ablates the feature sets (schedule-only vs. with real-time signals);
8.  writes every table to CSV and every figure to PNG.
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

from mumbai_crowd import plots
from mumbai_crowd.classification import (
    BandClassifier,
    CalibratedBandClassifier,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    split_validation_by_date,
)
from mumbai_crowd.config import (
    COST_OVER,
    COST_UNDER,
    ModelConfig,
    SimConfig,
    optimal_quantile,
)
from mumbai_crowd.decision import (
    DistributionalPolicy,
    NaivePolicy,
    ProbabilityPolicy,
    ThresholdPolicy,
    action_confusion,
    policy_report,
)
from mumbai_crowd.features import TARGET, build_design, prepare_matrix
from mumbai_crowd.losses import (
    AsymmetricHuber,
    AsymmetricSquaredError,
    LinexLoss,
    PinballLoss,
    SquaredError,
)
from mumbai_crowd.metrics import (
    METRIC_GLOSSARY,
    band_confusion,
    danger_reliability_table,
    leaderboard,
    regression_report,
)
from mumbai_crowd.regression import (
    BoostedRegressor,
    HistoricalBaseline,
    LinearAsymmetric,
    MarginBaseline,
    MeanBaseline,
    QuantileEnsemble,
    fit_predict,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

#: Quantile grid.  Deliberately dense above 0.9: the decision that matters
#: turns on a ~1.5% event, and a grid that stops at 0.99 cannot resolve it.
TAUS = (0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.995)


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78, flush=True)


def load_data(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "coach_observations.csv.gz"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run:  python scripts/01_generate_data.py --days 180"
        )
    return pd.read_csv(path, parse_dates=["date", "wx_date", "timestamp"], low_memory=False)


def build_zoo(mcfg: ModelConfig, tau: float) -> dict:
    """The model zoo.  Everything after the baselines differs only in its loss."""
    return {
        "mean_l2": MeanBaseline(SquaredError()),
        "historical_profile": HistoricalBaseline(),
        "linear_l2": LinearAsymmetric(SquaredError()),
        "linear_asym": LinearAsymmetric(AsymmetricSquaredError(COST_UNDER, COST_OVER)),
        "lgbm_l2": BoostedRegressor(loss=SquaredError(), cfg=mcfg, label="lgbm_l2"),
        "lgbm_asym_l2": BoostedRegressor(
            loss=AsymmetricSquaredError(COST_UNDER, COST_OVER), cfg=mcfg, label="lgbm_asym_l2"
        ),
        "lgbm_asym_huber": BoostedRegressor(
            loss=AsymmetricHuber(2.0, COST_UNDER, COST_OVER), cfg=mcfg, label="lgbm_asym_huber"
        ),
        "lgbm_pinball": BoostedRegressor(loss=PinballLoss(tau), cfg=mcfg, label="lgbm_pinball"),
        "lgbm_linex": BoostedRegressor(loss=LinexLoss(mcfg.linex_a), cfg=mcfg, label="lgbm_linex"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--out", type=str, default="reports")
    ap.add_argument("--feature-set", type=str, default="schedule", choices=["schedule", "realtime"])
    ap.add_argument("--rounds", type=int, default=ModelConfig.n_estimators)
    ap.add_argument("--val-days", type=int, default=ModelConfig.val_days,
                    help="days held out for early stopping and threshold tuning")
    ap.add_argument("--test-days", type=int, default=ModelConfig.test_days,
                    help="final days held out for reporting")
    ap.add_argument("--skip-quantile", action="store_true")
    ap.add_argument("--skip-classifier", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    mcfg = ModelConfig(n_estimators=args.rounds, val_days=args.val_days,
                       test_days=args.test_days)
    tau = optimal_quantile()

    banner("CO2 | Loading data and building the design matrix")
    df = load_data(Path(args.data))
    print(f"  {len(df):,} coach-arrival observations, {df['date'].nunique()} operating days")
    split, cols, encoder = build_design(
        df, args.feature_set, val_days=mcfg.val_days, test_days=mcfg.test_days, cfg=SimConfig()
    )
    print(split.describe())
    print(f"  feature set '{args.feature_set}': {len(cols)} predictors")
    print(f"  cost ratio c_under/c_over = {COST_UNDER / COST_OVER:g}  ->  implied quantile tau = {tau:.4f}")

    y_val = split.val[TARGET].to_numpy(float)
    y_test = split.test[TARGET].to_numpy(float)

    # ---------------------------------------------------------------- zoo --
    banner("CO2 | Training the model zoo (identical features, different losses)")
    zoo = build_zoo(mcfg, tau)
    preds_val, preds_test, fitted = {}, {}, {}
    for name, model in zoo.items():
        t0 = time.time()
        model, preds = fit_predict(model, split, cols)
        fitted[name] = model
        preds_val[name] = preds["val"]
        preds_test[name] = preds["test"]
        extra = f" best_iter={model.best_iteration_}" if hasattr(model, "best_iteration_") else ""
        print(f"  {name:<20s} {time.time() - t0:6.1f}s{extra}")

    # The margin strawman needs a fitted base model, so it comes second.
    margin = MarginBaseline(fitted["lgbm_l2"])
    margin.fit(prepare_matrix(split.train, cols), split.train[TARGET],
               prepare_matrix(split.val, cols), y_val)
    preds_val["lgbm_l2_margin"] = margin.predict(prepare_matrix(split.val, cols))
    preds_test["lgbm_l2_margin"] = margin.predict(prepare_matrix(split.test, cols))
    fitted["lgbm_l2_margin"] = margin
    print(f"  {'lgbm_l2_margin':<20s} tuned flat margin = +{margin.margin_:.2f} standees/m²")

    banner("CO2 | Leaderboard on the held-out test period (naive band edges)")
    board = leaderboard({k: regression_report(y_test, v, tau) for k, v in preds_test.items()})
    print(board.round(4).to_string())
    print("\n  metric glossary")
    for k, v in METRIC_GLOSSARY.items():
        print(f"    {k:<18s} {v}")
    board.to_csv(tabdir / "co2_leaderboard.csv")

    # ------------------------------------------------------------ policies --
    banner("CO2 | Decision layer: cost-optimal thresholds, fitted on validation")
    best_reg = board.index[0]
    print(f"  lowest-cost regressor on test: {best_reg}")

    policy_rows = {}
    naive = NaivePolicy()
    for name in ("lgbm_l2", "lgbm_asym_l2", "lgbm_pinball", best_reg):
        policy_rows[f"{name} + naive edges"] = policy_report(y_test, naive.decide(preds_test[name]))

    thresholds = {}
    for name in ("lgbm_l2", "lgbm_asym_l2", "lgbm_pinball"):
        tp = ThresholdPolicy().fit(y_val, preds_val[name])
        thresholds[name] = tp
        policy_rows[f"{name} + cost-optimal thresholds"] = policy_report(
            y_test, tp.decide(preds_test[name])
        )
        print(f"\n  {name}: thresholds fitted on validation")
        print(tp.describe())

    # ------------------------------------------------------ quantile model --
    qe = None
    if not args.skip_quantile:
        banner("CO2 | Distributional model: one LightGBM per quantile, then Bayes action")
        t0 = time.time()
        qe = QuantileEnsemble(taus=TAUS, cfg=mcfg)
        Xtr = prepare_matrix(split.train, cols)
        Xva = prepare_matrix(split.val, cols)
        Xte = prepare_matrix(split.test, cols)
        qe.fit(Xtr, split.train[TARGET].to_numpy(float), Xva, y_val)
        q_test = qe.predict_quantiles(Xte)
        print(f"  fitted {len(TAUS)} quantile models in {time.time() - t0:.1f}s")

        dp = DistributionalPolicy(taus=np.asarray(TAUS))
        policy_rows["quantile ensemble + Bayes action"] = policy_report(y_test, dp.decide(q_test))

        # The Bayes rule is optimal only if its probabilities are conditionally
        # calibrated, so check that rather than assuming it.
        probs = dp.band_probabilities(q_test)
        p_danger = probs[:, 3]
        observed = float((y_test >= 12.0).mean())
        print(f"\n  mean predicted P(DANGEROUS) = {p_danger.mean():.4f}  "
              f"vs observed base rate {observed:.4f}")
        print(f"  rows with P(DANGEROUS) > 0.25: {(p_danger > 0.25).sum():,} "
              f"of {len(p_danger):,}  ({(p_danger > 0.25).mean():.3%})")
        print("  (the Bayes rule needs roughly P>0.25 before a relief rake beats marshalling,")
        print("   so if the tail probability is under-stated the optimal policy simply never fires)")
        print()
        print(danger_reliability_table(p_danger, y_test).round(4).to_string(index=False))

        coverage = pd.DataFrame(
            {
                "tau": TAUS,
                "empirical_coverage": [float(np.mean(y_test <= q_test[:, j])) for j in range(len(TAUS))],
            }
        )
        coverage["gap"] = coverage["empirical_coverage"] - coverage["tau"]
        print("\n  quantile calibration on test:")
        print(coverage.round(4).to_string(index=False))
        coverage.to_csv(tabdir / "co2_quantile_calibration.csv", index=False)
        plots.fig_quantile_calibration(y_test, q_test, TAUS, figdir)

    # ------------------------------------------------- band classifier ------
    reliability_tables = {}
    if qe is not None:
        reliability_tables["quantile ensemble"] = danger_reliability_table(
            DistributionalPolicy(taus=np.asarray(TAUS)).band_probabilities(q_test)[:, 3], y_test
        )

    calib_rows = []
    if not args.skip_classifier:
        banner("CO2 | Modelling the bands directly: classifier x recalibration")
        print("  The quantile route infers a 1% tail from a distribution fitted to the whole")
        print("  range.  A multiclass model targets the bands themselves.  Two levers are")
        print("  crossed here because they interact: class weighting (which buys the tail")
        print("  capacity at the cost of calibration) and post-hoc recalibration (which is")
        print("  supposed to give the calibration back).")

        val_es, val_cal = split_validation_by_date(split.val)
        print(f"\n  validation split by date so calibration never sees the early-stopping rows:")
        print(f"    early stopping : {val_es['date'].min().date()} .. {val_es['date'].max().date()}  n={len(val_es):,}")
        print(f"    calibration    : {val_cal['date'].min().date()} .. {val_cal['date'].max().date()}  n={len(val_cal):,}")

        Xtr = prepare_matrix(split.train, cols)
        Xes, y_es = prepare_matrix(val_es, cols), val_es[TARGET].to_numpy(float)
        Xcal, y_cal = prepare_matrix(val_cal, cols), val_cal[TARGET].to_numpy(float)
        Xte = prepare_matrix(split.test, cols)
        ytr = split.train[TARGET].to_numpy(float)

        base_rates = {
            "train": float((split.train[TARGET] >= 12.0).mean()),
            "val (calibration half)": float((val_cal[TARGET] >= 12.0).mean()),
            "test": float((split.test[TARGET] >= 12.0).mean()),
        }
        print("\n  DANGEROUS base rate by period (this is the thing that shifts):")
        for k, v in base_rates.items():
            print(f"    {k:24s} {v:.4f}")

        prob_policy = ProbabilityPolicy()
        best_clf = None
        for weight in (None, "balanced"):
            for cal in (IdentityCalibrator(), TemperatureCalibrator(), IsotonicCalibrator()):
                model = CalibratedBandClassifier(
                    BandClassifier(cfg=mcfg, class_weight=weight), calibrator=cal
                )
                model.fit(Xtr, ytr, Xes, y_es, Xcal, y_cal)
                probs = model.predict_proba(Xte)
                rep = policy_report(y_test, prob_policy.decide(probs))
                kind = type(cal).__name__.replace("Calibrator", "").lower()
                rep.update({
                    "class_weight": "balanced" if weight else "none",
                    "calibrator": kind,
                    "mean_p_danger": float(probs[:, 3].mean()),
                    "temperature": float(getattr(cal, "temperature_", np.nan)),
                })
                calib_rows.append(rep)
                policy_rows[f"classifier[{rep['class_weight']}] + {kind} + Bayes action"] = {
                    k: v for k, v in rep.items()
                    if k not in ("class_weight", "calibrator", "mean_p_danger", "temperature")
                }
                print(f"  {rep['class_weight']:>8s} weighting / {kind:<11s} "
                      f"cost={rep['exp_cost_inr']:7.2f}  miss={rep['dangerous_miss']:.4f}  "
                      f"recall={rep['danger_recall']:.4f}  FA={rep['false_alarm']:.4f}  "
                      f"meanP(DANG)={rep['mean_p_danger']:.4f}")
                if best_clf is None or rep["exp_cost_inr"] < best_clf[0]:
                    best_clf = (rep["exp_cost_inr"], model, probs)

        calib_df = pd.DataFrame(calib_rows)
        calib_df.to_csv(tabdir / "co2_calibration_grid.csv", index=False)
        plots.fig_calibration_grid(calib_df, figdir)

        reliability_tables["band classifier"] = danger_reliability_table(best_clf[2][:, 3], y_test)
        print(f"\n  cheapest classifier variant: ₹{best_clf[0]:.2f}  ({best_clf[1].name})")

    if reliability_tables:
        print(f"\n  {plots.fig_danger_reliability(reliability_tables, figdir)}")
        for name, tab in reliability_tables.items():
            tab.to_csv(tabdir / f"co2_reliability_{name.replace(' ', '_')}.csv", index=False)

    policy_table = pd.DataFrame(policy_rows).T.sort_values("exp_cost_inr")
    banner("CO2 | Policy comparison on the test period")
    print(policy_table.round(4).to_string())
    policy_table.to_csv(tabdir / "co2_policy_comparison.csv")

    best_policy = policy_table.index[0]
    print(f"\n  cheapest policy: {best_policy}")
    print(f"  vs. the naive-edges L2 baseline: "
          f"{policy_table.loc['lgbm_l2 + naive edges', 'exp_cost_inr']:.1f} -> "
          f"{policy_table.loc[best_policy, 'exp_cost_inr']:.1f} ₹/arrival "
          f"({100 * (1 - policy_table.loc[best_policy, 'exp_cost_inr'] / policy_table.loc['lgbm_l2 + naive edges', 'exp_cost_inr']):.1f}% lower)")

    # ---------------------------------------------------------- sensitivity --
    curve = None
    if not args.skip_sensitivity:
        banner("CO2 | Sensitivity: the cost ratio is a dial the operator sets")
        rows = []
        for ratio in (1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 40.0):
            m = BoostedRegressor(loss=PinballLoss(ratio / (ratio + 1.0)), cfg=mcfg)
            m, p = fit_predict(m, split, cols)
            rep = regression_report(y_test, p["test"], tau)
            rep["ratio"] = ratio
            rep["tau"] = ratio / (ratio + 1.0)
            rows.append(rep)
            print(f"  ratio {ratio:>5.1f}  tau={rep['tau']:.3f}  bias={rep['bias']:+.3f}  "
                  f"dangerous_miss={rep['dangerous_miss']:.3f}  false_alarm={rep['false_alarm']:.3f}")
        curve = pd.DataFrame(rows).set_index("ratio", drop=False)
        curve.to_csv(tabdir / "co2_cost_sensitivity.csv", index=False)

    # ------------------------------------------------------------ ablation --
    banner("CO2 | Feature-set ablation: what does real-time telemetry buy?")
    ablation = {}
    for fs in ("schedule", "realtime"):
        sp, cl, _ = build_design(df, fs, val_days=mcfg.val_days, test_days=mcfg.test_days)
        m = BoostedRegressor(loss=PinballLoss(tau), cfg=mcfg)
        m, p = fit_predict(m, sp, cl)
        rep = regression_report(sp.test[TARGET], p["test"], tau)
        tp = ThresholdPolicy().fit(sp.val[TARGET], p["val"])
        rep.update(policy_report(sp.test[TARGET], tp.decide(p["test"])))
        rep["n_features"] = len(cl)
        ablation[fs] = rep
    ablation_df = pd.DataFrame(ablation).T
    print(ablation_df[["n_features", "rmse", "r2", "exp_cost_inr", "dangerous_miss",
                       "danger_recall", "false_alarm"]].round(4).to_string())
    ablation_df.to_csv(tabdir / "co2_feature_ablation.csv")

    # -------------------------------------------------------------- figures --
    banner("CO2 | Writing figures")
    shown = {
        "lgbm_l2 (symmetric)": preds_test["lgbm_l2"],
        "lgbm_asym_l2": preds_test["lgbm_asym_l2"],
        "lgbm_pinball": preds_test["lgbm_pinball"],
        "lgbm_linex": preds_test["lgbm_linex"],
    }
    paths = [
        plots.fig_loss_shapes(
            {
                "L2 (symmetric)": SquaredError(),
                f"asym L2 ({COST_UNDER:g}:{COST_OVER:g})": AsymmetricSquaredError(COST_UNDER, COST_OVER),
                f"pinball τ={tau:.2f}": PinballLoss(tau),
                f"LINEX a={mcfg.linex_a:g}": LinexLoss(mcfg.linex_a),
            },
            figdir,
        ),
        plots.fig_target_distribution(df, TARGET, figdir),
        plots.fig_demand_surface(df, figdir),
        plots.fig_rain_effect(df, figdir),
        plots.fig_model_comparison(board, figdir),
        plots.fig_prediction_scatter(y_test, shown, figdir),
        plots.fig_residual_asymmetry(y_test, shown, figdir),
        plots.fig_confusion_grid(y_test, shown, figdir),
        plots.fig_threshold_policy(thresholds["lgbm_pinball"], y_val, preds_val["lgbm_pinball"], figdir),
        plots.fig_feature_importance(fitted["lgbm_pinball"], outdir=figdir),
    ]
    if curve is not None:
        paths.append(plots.fig_cost_sensitivity(curve, figdir))
    for p in paths:
        print(f"  {p}")

    # -------------------------------------------------------------- summary --
    summary = {
        "feature_set": args.feature_set,
        "n_rows": int(len(df)),
        "n_features": len(cols),
        "cost_ratio": COST_UNDER / COST_OVER,
        "tau": tau,
        "best_regressor_by_cost": best_reg,
        "best_policy": best_policy,
        "baseline_cost_inr": float(policy_table.loc["lgbm_l2 + naive edges", "exp_cost_inr"]),
        "best_cost_inr": float(policy_table.loc[best_policy, "exp_cost_inr"]),
        "baseline_dangerous_miss": float(policy_table.loc["lgbm_l2 + naive edges", "dangerous_miss"]),
        "best_dangerous_miss": float(policy_table.loc[best_policy, "dangerous_miss"]),
        "tuned_flat_margin": float(margin.margin_),
        "thresholds_pinball": [float(x) for x in thresholds["lgbm_pinball"].thresholds_],
    }
    if calib_rows:
        best = min(calib_rows, key=lambda r: r["exp_cost_inr"])
        summary["classifier"] = {
            "best_variant": f"{best['class_weight']} weighting / {best['calibrator']} calibration",
            "cost_inr": float(best["exp_cost_inr"]),
            "dangerous_miss": float(best["dangerous_miss"]),
            "danger_recall": float(best["danger_recall"]),
            "grid_cost_inr": {
                f"{r['class_weight']}/{r['calibrator']}": float(r["exp_cost_inr"]) for r in calib_rows
            },
        }
    (tabdir / "co2_summary.json").write_text(json.dumps(summary, indent=2))
    band_confusion(y_test, preds_test["lgbm_pinball"], normalize="true").to_csv(
        tabdir / "co2_confusion_pinball.csv"
    )
    action_confusion(y_test, thresholds["lgbm_pinball"].decide(preds_test["lgbm_pinball"])).to_csv(
        tabdir / "co2_action_confusion.csv"
    )
    print(f"\n  wrote {tabdir / 'co2_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
