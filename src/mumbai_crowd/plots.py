"""Figures for the report.

Everything writes a PNG into ``reports/figures`` and returns the path, so the
pipeline scripts stay free of matplotlib boilerplate.  The house style is set
once in :func:`use_house_style`: no chartjunk, colour used only where it
carries information, and the four crowd bands always drawn in the same colours
so a reader learns the palette once.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

from .config import BAND_EDGES, BAND_LABELS, COST_MATRIX, density_to_band

FIGDIR = Path("reports/figures")

#: One colour per crowd band, used identically in every figure.
BAND_COLORS: list[str] = ["#3d7ea6", "#68a357", "#e0a53a", "#c0392b"]
ACCENT = "#c0392b"
NEUTRAL = "#4a5259"
GRID = "#d8dcdf"


def use_house_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def _save(fig, name: str, outdir: Path | None = None) -> Path:
    outdir = Path(outdir or FIGDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _band_bands(ax, orientation: str = "x", alpha: float = 0.07) -> None:
    """Shade the four crowd bands behind a plot."""
    edges = [0.0, *BAND_EDGES, max(ax.get_xlim()[1] if orientation == "x" else ax.get_ylim()[1], 20)]
    for i in range(4):
        span = ax.axvspan if orientation == "x" else ax.axhspan
        span(edges[i], edges[i + 1], color=BAND_COLORS[i], alpha=alpha, lw=0)


# ---------------------------------------------------------------------------
# CO2 -- the loss functions themselves
# ---------------------------------------------------------------------------


def fig_loss_shapes(losses: dict[str, object], outdir: Path | None = None) -> Path:
    """Plot each loss against the prediction error.

    The single most useful figure in the report: it shows, in one glance, that
    every asymmetric loss is a bowl with one steep wall, and which wall.

    Each curve is rescaled to pass through 1.0 at one unit of *under*-
    prediction.  Without that, the quadratic losses are two orders of
    magnitude taller than the piecewise-linear ones and the panel becomes a
    picture of their absolute scale -- which is arbitrary, since a loss can be
    multiplied by any positive constant without changing its minimiser.  What
    is *not* arbitrary is the ratio between the two walls, and normalising is
    what makes that ratio the visible thing.
    """
    use_house_style()
    r = np.linspace(-6, 6, 601)          # r = y_pred - y_true
    y_true = np.zeros_like(r)
    ref = np.array([-1.0])               # one unit short of the truth

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(losses)))
    for ax, (title, scale) in zip(axes, [("Loss", "linear"), ("Loss (log scale)", "log")]):
        for (name, loss), colour in zip(losses.items(), colours):
            scale_factor = float(loss.elementwise(np.zeros(1), ref)[0])
            curve = loss.elementwise(y_true, r) / max(scale_factor, 1e-12)
            ax.plot(r, curve, label=name, color=colour, lw=1.8)
        ax.axvline(0, color=NEUTRAL, lw=0.8, ls=":")
        ax.set_yscale(scale)
        ax.set_xlabel("prediction error  (predicted - actual, standees/m²)")
        ax.set_ylabel("loss, normalised to 1.0 at one unit short")
        ax.set_title(title)
        ax.axvspan(-6, 0, color=ACCENT, alpha=0.06, lw=0)
    axes[0].set_ylim(0, 26)
    axes[0].text(-5.85, 2.2, "UNDER-predicted\n(the dangerous side)",
                 color=ACCENT, fontsize=8.5, va="bottom", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), fontsize=8.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(
        "Same scale at one unit short: the asymmetry is the gap between the two walls",
        y=1.02, fontsize=11,
    )
    return _save(fig, "01_loss_shapes.png", outdir)


# ---------------------------------------------------------------------------
# CO2 -- the data
# ---------------------------------------------------------------------------


def fig_target_distribution(df: pd.DataFrame, target: str = "density_depart",
                            outdir: Path | None = None) -> Path:
    use_house_style()
    y = df[target].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

    axes[0].hist(y, bins=120, color=NEUTRAL, alpha=0.85)
    axes[0].set_yscale("log")
    for e, c in zip(BAND_EDGES, BAND_COLORS[1:]):
        axes[0].axvline(e, color=c, lw=1.4, ls="--")
    axes[0].set_xlabel("density (standees/m²)")
    axes[0].set_ylabel("coach-arrivals (log)")
    axes[0].set_title("Target is zero-inflated with a long, rare right tail")

    shares = pd.Series(density_to_band(y)).value_counts(normalize=True).reindex(range(4), fill_value=0)
    axes[1].bar(BAND_LABELS, shares.to_numpy() * 100, color=BAND_COLORS)
    for i, v in enumerate(shares.to_numpy() * 100):
        axes[1].text(i, v, f" {v:.2f}%", ha="center", va="bottom", fontsize=8.5)
    axes[1].set_ylabel("share of coach-arrivals (%)")
    axes[1].set_title("The class that matters is 1-2% of the data")
    axes[1].set_ylim(0, max(shares.max() * 115, 10))
    return _save(fig, "02_target_distribution.png", outdir)


def fig_demand_surface(df: pd.DataFrame, outdir: Path | None = None) -> Path:
    """Mean density by station and hour, one panel per direction."""
    use_house_style()
    from .network import ROUTES, load_stations

    stops = ROUTES["CSMT_PNVL"]
    st = load_stations()
    cmap = LinearSegmentedColormap.from_list("crowd", ["#f7f9fa", "#68a357", "#e0a53a", "#c0392b"])

    sub = df[(df["route"] == "CSMT_PNVL") & (df["is_weekend"] == 0) & (df["is_holiday"] == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, direction, title in zip(axes, ["UP", "DN"], ["UP  (suburbs → CSMT)", "DN  (CSMT → suburbs)"]):
        piv = (
            sub[sub["direction"] == direction]
            .pivot_table(index="station_code", columns="hour", values="density_depart", aggfunc="mean")
            .reindex(index=stops)
            # Hours 2-3 have no services, so the pivot's columns are not
            # contiguous.  Without reindexing to a full 0..23 range imshow
            # spreads the surviving columns evenly across the extent and every
            # hour label silently shifts.
            .reindex(columns=range(24))
        )
        im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=12,
                       extent=[-0.5, 23.5, len(stops) - 0.5, -0.5])
        ax.set_xticks(range(4, 24, 2))
        ax.set_xlim(3.5, 23.5)
        ax.set_title(f"{title}   weekdays")
        ax.set_xlabel("hour of day")
        ax.grid(False)
    axes[0].set_yticks(range(len(stops)))
    axes[0].set_yticklabels([st.loc[c, "name"][:18] for c in stops], fontsize=6.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("mean density (standees/m²)")
    fig.suptitle("The tidal reversal: load peaks mid-route, not at the terminus", y=1.0, fontsize=11)
    return _save(fig, "03_demand_surface.png", outdir)


def fig_rain_effect(df: pd.DataFrame, outdir: Path | None = None) -> Path:
    use_house_style()
    d = df[(df["is_weekend"] == 0) & (df["is_holiday"] == 0)].copy()
    d["rain_bin"] = pd.cut(d["rain_mm_hr"], [-0.01, 0.1, 1.0, 3.0, 7.5, 200],
                           labels=["dry", "<1 mm/h", "1-3", "3-7.5", "≥7.5 (heavy)"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

    peak = d[d["is_peak"] == 1] if "is_peak" in d.columns else d[d["hour"].isin([8, 9, 10, 18, 19, 20])]
    g = peak.groupby("rain_bin", observed=True)["density_depart"].agg(["mean", "size"])
    axes[0].bar(g.index.astype(str), g["mean"], color=NEUTRAL)
    axes[0].set_ylabel("mean density (standees/m²)")
    axes[0].set_title("Peak-hour density rises with rainfall")
    axes[0].tick_params(axis="x", rotation=20)

    g2 = peak.groupby("rain_bin", observed=True).apply(
        lambda x: (density_to_band(x["density_depart"]) == 3).mean() * 100, include_groups=False
    )
    axes[1].bar(g2.index.astype(str), g2.to_numpy(), color=ACCENT)
    axes[1].set_ylabel("% of coach-arrivals DANGEROUS")
    axes[1].set_title("...and the dangerous tail rises faster than the mean")
    axes[1].tick_params(axis="x", rotation=20)
    return _save(fig, "04_rain_effect.png", outdir)


# ---------------------------------------------------------------------------
# CO2 -- model comparison
# ---------------------------------------------------------------------------


def fig_model_comparison(board: pd.DataFrame, outdir: Path | None = None) -> Path:
    """The trade-off, made explicit: accuracy down, safety up."""
    use_house_style()
    board = board.copy()
    panels = [
        ("rmse", "RMSE (standees/m²)", "lower is better", False),
        ("exp_cost_inr", "Expected cost per arrival (₹)", "lower is better", False),
        ("dangerous_miss", "P(said safe | truly dangerous)", "lower is better", True),
        ("false_alarm", "P(escalated | truly below crush)", "lower is better", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), sharey=True)
    order = board.sort_values("exp_cost_inr").index
    for i, (ax, (col, title, sub, pct)) in enumerate(zip(axes, panels)):
        vals = board.loc[order, col].to_numpy() * (100 if pct else 1)
        colours = [ACCENT if j == int(np.argmin(vals)) else NEUTRAL for j in range(len(vals))]
        ax.barh(range(len(order)), vals, color=colours)
        ax.set_title(f"{title}\n({sub})", fontsize=9)
        if pct:
            ax.set_xlabel("%")
    # Model names once, on the left, instead of four times over the bars.
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=7.5)
    axes[0].invert_yaxis()
    fig.suptitle("Same features, same model family — only the loss changes", y=1.03, fontsize=11)
    return _save(fig, "05_model_comparison.png", outdir)


def fig_prediction_scatter(y_true, preds: dict[str, np.ndarray], outdir: Path | None = None) -> Path:
    use_house_style()
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    hi = float(np.quantile(y_true, 0.9995))
    for ax, (name, p) in zip(axes, preds.items()):
        ax.hexbin(y_true, p, gridsize=55, bins="log", cmap="Greys", extent=(0, hi, 0, hi))
        ax.plot([0, hi], [0, hi], color=ACCENT, lw=1.1, ls="--")
        for e in BAND_EDGES:
            ax.axhline(e, color=NEUTRAL, lw=0.5, ls=":")
            ax.axvline(e, color=NEUTRAL, lw=0.5, ls=":")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("actual density")
    axes[0].set_ylabel("predicted density")
    fig.suptitle("Points below the diagonal are the dangerous errors", y=1.02, fontsize=11)
    return _save(fig, "06_prediction_scatter.png", outdir)


def fig_residual_asymmetry(y_true, preds: dict[str, np.ndarray], outdir: Path | None = None) -> Path:
    use_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"wspace": 0.55})
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(preds)))

    for (name, p), c in zip(preds.items(), colours):
        r = np.asarray(p) - np.asarray(y_true)
        axes[0].hist(r, bins=160, range=(-8, 8), histtype="step", lw=1.5, label=name, color=c, density=True)
    axes[0].axvline(0, color=NEUTRAL, lw=0.8, ls=":")
    axes[0].axvspan(-8, 0, color=ACCENT, alpha=0.06, lw=0)
    axes[0].set_xlabel("residual (predicted - actual)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Residuals shift right")
    axes[0].legend(fontsize=7.5)

    danger = np.asarray(y_true) >= BAND_EDGES[-1]
    names, vals = [], []
    for name, p in preds.items():
        names.append(name)
        vals.append(float(np.mean(np.maximum(np.asarray(y_true)[danger] - np.asarray(p)[danger], 0))))
    axes[1].barh(range(len(names)), vals,
                 color=[ACCENT if v == min(vals) else NEUTRAL for v in vals])
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean shortfall on DANGEROUS coaches (standees/m²)")
    fig.suptitle("Asymmetric losses move the whole distribution, and shrink the dangerous tail",
                 y=1.03, fontsize=11)
    axes[1].set_title("Shortfall where it matters")
    return _save(fig, "07_residual_asymmetry.png", outdir)


def fig_confusion_grid(y_true, preds: dict[str, np.ndarray], outdir: Path | None = None) -> Path:
    use_house_style()
    from .metrics import band_confusion

    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.4))
    axes = np.atleast_1d(axes)
    for ax, (name, p) in zip(axes, preds.items()):
        cm = band_confusion(y_true, p, normalize="true").to_numpy()
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if cm[i, j] > 0.55 else "black")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([b[:5] for b in BAND_LABELS], fontsize=7, rotation=35)
        ax.set_yticklabels([b[:5] for b in BAND_LABELS], fontsize=7)
        ax.set_title(name, fontsize=8.5)
        ax.set_xlabel("predicted")
        ax.grid(False)
        # Outline the cells that cost the most.
        for i in (3,):
            for j in (0, 1):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=ACCENT, lw=1.8))
    axes[0].set_ylabel("actual")
    fig.suptitle("Red cells = told the operator a dangerous coach was safe", y=1.02, fontsize=10.5)
    return _save(fig, "08_confusion_grid.png", outdir)


def fig_threshold_policy(policy, y_val, pred_val, outdir: Path | None = None) -> Path:
    """Cost as a function of each alert cut-point, with the fitted value marked."""
    use_house_style()
    from .config import density_to_band as _band

    true_band = _band(y_val)
    grid = np.arange(0.0, float(np.quantile(pred_val, 0.999)) + 0.05, 0.05)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for k, (ax, label) in enumerate(zip(axes, BAND_LABELS[1:])):
        costs = []
        for c in grid:
            t = policy.thresholds_.copy()
            t[k] = c
            t = np.sort(t)
            action = np.searchsorted(t, pred_val, side="right")
            costs.append(float(np.mean(COST_MATRIX[true_band, action])))
        ax.plot(grid, costs, color=NEUTRAL, lw=1.6)
        ax.axvline(policy.thresholds_[k], color=ACCENT, lw=1.6,
                   label=f"fitted = {policy.thresholds_[k]:.2f}")
        ax.axvline(BAND_EDGES[k], color=NEUTRAL, ls="--", lw=1.2,
                   label=f"physical edge = {BAND_EDGES[k]:.0f}")
        ax.set_title(f"cut-point for {label}", fontsize=9)
        ax.set_xlabel("threshold on predicted density")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("expected cost per arrival (₹)")
    fig.suptitle("Cost-optimal alert thresholds sit below the physical band edges", y=1.03, fontsize=11)
    return _save(fig, "09_threshold_policy.png", outdir)


def fig_cost_sensitivity(curve: pd.DataFrame, outdir: Path | None = None) -> Path:
    """How the chosen model changes as the assumed cost ratio changes."""
    use_house_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    axes[0].plot(curve["ratio"], curve["bias"], color=NEUTRAL, marker="o", ms=3.5)
    axes[0].set_ylabel("mean signed error"); axes[0].set_title("Learned safety margin")
    axes[1].plot(curve["ratio"], curve["dangerous_miss"] * 100, color=ACCENT, marker="o", ms=3.5)
    axes[1].set_ylabel("%"); axes[1].set_title("P(said safe | truly dangerous)")
    axes[2].plot(curve["ratio"], curve["false_alarm"] * 100, color=NEUTRAL, marker="o", ms=3.5)
    axes[2].set_ylabel("%"); axes[2].set_title("False-alarm rate")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("assumed cost ratio  c_under / c_over")
    fig.suptitle("The cost ratio is a policy dial, and it behaves monotonically", y=1.03, fontsize=11)
    return _save(fig, "10_cost_sensitivity.png", outdir)


def fig_feature_importance(model, top: int = 22, outdir: Path | None = None) -> Path:
    use_house_style()
    imp = model.feature_importance("gain").head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(imp) + 1.2))
    ax.barh(imp.index, imp.to_numpy(), color=NEUTRAL)
    ax.set_xlabel("total gain")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title(f"What the model uses  ({model.name})")
    return _save(fig, "11_feature_importance.png", outdir)


def fig_quantile_calibration(y_true, qpred: np.ndarray, taus, outdir: Path | None = None) -> Path:
    """Do the predicted quantiles actually have the coverage they claim?"""
    use_house_style()
    taus = np.asarray(taus, dtype=float)
    empirical = np.array([float(np.mean(np.asarray(y_true) <= qpred[:, j])) for j in range(len(taus))])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=1.0, label="perfect calibration")
    ax.plot(taus, empirical, marker="o", color=ACCENT, lw=1.6, label="quantile ensemble")
    for t, e in zip(taus, empirical):
        ax.annotate(f"{e:.3f}", (t, e), textcoords="offset points", xytext=(6, -8), fontsize=7)
    ax.set_xlabel("nominal quantile τ"); ax.set_ylabel("empirical coverage")
    ax.set_title("Quantile calibration on the test period")
    ax.legend(fontsize=8)
    return _save(fig, "12_quantile_calibration.png", outdir)


def fig_danger_reliability(table: pd.DataFrame, outdir: Path | None = None) -> Path:
    """Plot a reliability table from :func:`metrics.danger_reliability_table`.

    The Bayes decision rule is only optimal if the probabilities it consumes
    are *conditionally* calibrated -- marginal coverage is not enough.  Points
    below the diagonal mean the model is over-confident about danger; points
    above mean it is under-stating it, and under-stating a 1.5% event is
    exactly how a theoretically optimal policy ends up refusing to act.
    """
    use_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    lim = max(float(table[["predicted", "observed"]].to_numpy().max()) * 1.15, 0.05)
    axes[0].plot([0, lim], [0, lim], ls="--", color=NEUTRAL, lw=1.0, label="perfect calibration")
    axes[0].plot(table["predicted"], table["observed"], marker="o", color=ACCENT, lw=1.6,
                 label="quantile ensemble")
    axes[0].set_xlabel("predicted P(DANGEROUS)")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_title("Reliability of the danger probability")
    axes[0].legend(fontsize=8)

    axes[1].bar(range(len(table)), table["n"], color=NEUTRAL)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(table)))
    axes[1].set_xticklabels([f"{v:.3f}" for v in table["predicted"]], rotation=45, fontsize=7)
    axes[1].set_xlabel("bin (mean predicted probability)")
    axes[1].set_ylabel("coach-arrivals (log)")
    axes[1].set_title("How much data sits in each bin")
    fig.suptitle("A Bayes rule is only as good as the probabilities it is fed", y=1.02, fontsize=11)
    return _save(fig, "13_danger_reliability.png", outdir)


# ---------------------------------------------------------------------------
# CO5 -- clustering
# ---------------------------------------------------------------------------


def fig_k_selection(indices: pd.DataFrame, gmm: pd.DataFrame, chosen_k: int,
                    stability: pd.Series | None = None, outdir: Path | None = None) -> Path:
    """Every criterion for choosing k, side by side, disagreeing openly.

    The last panel is the one that should carry the most weight and usually
    gets left out: how reproducible the partition is when the *days* are
    resampled.  Internal indices measure how tidy a partition looks on the
    sample you happen to have; stability measures whether you would have found
    the same partition at all on a different sample.
    """
    use_house_style()
    specs = [
        ("inertia", "Within-cluster SSE\n(+ GMM BIC)"),
        ("silhouette", "Silhouette\n(higher better)"),
        ("davies_bouldin", "Davies-Bouldin\n(lower better)"),
        ("calinski_harabasz", "Calinski-Harabasz\n(higher better)"),
    ]
    n_panels = len(specs) + (1 if stability is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.1 * n_panels, 3.4),
                             gridspec_kw={"wspace": 0.45})
    for ax, (col, title) in zip(axes, specs):
        ax.plot(indices.index, indices[col], marker="o", ms=4, color=NEUTRAL)
        ax.axvline(chosen_k, color=ACCENT, lw=1.4, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("k")
    ax2 = axes[0].twinx()
    ax2.plot(gmm.index, gmm["bic"], marker="s", ms=3.5, color="#7a5195", alpha=0.85)
    ax2.set_ylabel("GMM BIC", color="#7a5195", fontsize=8)
    ax2.grid(False)

    if stability is not None:
        ax = axes[-1]
        st = stability.dropna()
        ax.plot(st.index, st.to_numpy(), marker="o", ms=5, color=ACCENT, lw=1.8)
        ax.axvline(chosen_k, color=ACCENT, lw=1.4, ls="--")
        ax.set_ylim(0, 1.05)
        ax.set_title("Bootstrap stability\n(ARI, higher better)", fontsize=9)
        ax.set_xlabel("k")

    fig.suptitle(
        f"Internal indices disagree; k = {chosen_k} is chosen on the elbow and on stability",
        y=1.05, fontsize=11,
    )
    return _save(fig, "20_k_selection.png", outdir)


def fig_dendrogram(Z: np.ndarray, labels: list[str], chosen_k: int,
                   outdir: Path | None = None) -> Path:
    use_house_style()
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    from scipy.cluster.hierarchy import set_link_color_palette

    set_link_color_palette(BAND_COLORS + ["#7a5195", "#7f8c8d"])
    heights = np.sort(Z[:, 2])
    cut = (heights[-chosen_k] + heights[-chosen_k + 1]) / 2 if chosen_k > 1 else heights[-1]
    dendrogram(Z, labels=labels, ax=ax, color_threshold=cut, leaf_font_size=7,
               above_threshold_color=NEUTRAL)
    ax.axhline(cut, color=ACCENT, ls="--", lw=1.2, label=f"cut for k = {chosen_k}")
    ax.set_ylabel("Ward linkage distance")
    ax.set_title("Hierarchical structure of Harbour-line station profiles")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=90)
    return _save(fig, "21_dendrogram.png", outdir)


def fig_cluster_pca(result, outdir: Path | None = None) -> Path:
    use_house_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    coords, labels = result.coords, result.labels.to_numpy()
    palette = BAND_COLORS + ["#7a5195", "#00798c", "#8d6e63"]
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=64, color=palette[c % len(palette)],
                   label=result.names.get(int(c), f"cluster {c}"), edgecolor="white", lw=0.8, zorder=3)
    for i, code in enumerate(result.profiles.index):
        ax.annotate(code, (coords[i, 0], coords[i, 1]), fontsize=6.2,
                    textcoords="offset points", xytext=(5, 3), color=NEUTRAL)
    ev = result.pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax.set_title(f"Station profiles in PCA space   (silhouette = {result.silhouette:.3f})")
    ax.legend(fontsize=8, loc="best")
    return _save(fig, "22_cluster_pca.png", outdir)


def fig_cluster_profiles(result, outdir: Path | None = None) -> Path:
    """Mean normalised boarding and alighting curve for each cluster."""
    use_house_style()
    from .clustering import PROFILE_HOURS

    prof = result.profiles.copy()
    prof["cluster"] = result.labels
    bcols = [f"board_h{h:02d}" for h in PROFILE_HOURS]
    acols = [f"alight_h{h:02d}" for h in PROFILE_HOURS]
    clusters = sorted(prof["cluster"].unique())
    palette = BAND_COLORS + ["#7a5195", "#00798c", "#8d6e63"]

    fig, axes = plt.subplots(1, len(clusters), figsize=(2.9 * len(clusters), 3.4),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, clusters):
        g = prof[prof["cluster"] == c]
        ax.plot(PROFILE_HOURS, g[bcols].mean().to_numpy() * 100, color=palette[c % len(palette)],
                lw=2.0, label="boardings")
        ax.plot(PROFILE_HOURS, g[acols].mean().to_numpy() * 100, color=NEUTRAL, lw=1.6,
                ls="--", label="alightings")
        ax.set_title(f"{result.names.get(int(c), c)}\n(n={len(g)})", fontsize=8)
        ax.set_xlabel("hour")
        ax.set_xticks(range(6, 24, 4))
    axes[0].set_ylabel("% of the station's daily flow")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Cluster signatures: when each kind of station is a source and when it is a sink",
                 y=1.04, fontsize=11)
    return _save(fig, "23_cluster_profiles.png", outdir)


def fig_line_map(result, outdir: Path | None = None) -> Path:
    """Schematic Harbour-line map with stations coloured by cluster role.

    Drawn as a branching schematic on *sequence* rather than chainage, for the
    same reason every real transit map is: on a true distance axis the eight
    stations between CSMT and Vadala Road occupy nine kilometres and their
    labels become an unreadable pile, while Kharghar to Panvel gets eight
    kilometres of empty paper.  The trunk is drawn once and the two branches
    diverge from Vadala Road, which is also the honest topology -- the earlier
    two-row version duplicated every trunk station.
    """
    use_house_style()
    from .network import ROUTES, load_stations

    st = load_stations()
    labels, names = result.labels, result.names
    palette = BAND_COLORS + ["#7a5195", "#00798c", "#8d6e63"]

    trunk = ROUTES["CSMT_PNVL"][:8]          # CSMT .. Vadala Road, shared
    panvel = ROUTES["CSMT_PNVL"][8:]
    goregaon = ROUTES["CSMT_GMN"][8:]
    assert ROUTES["CSMT_GMN"][:8] == trunk, "the two patterns no longer share a trunk"

    fig, ax = plt.subplots(figsize=(13.5, 5.6))

    def _draw(codes, xs, y, label_above: bool):
        ax.plot(xs, [y] * len(xs), color="#b9c0c5", lw=4.0, zorder=1, solid_capstyle="round")
        for code, x in zip(codes, xs):
            cid = int(labels[code])
            ax.scatter([x], [y], s=130, color=palette[cid % len(palette)], zorder=3,
                       edgecolor="white", lw=1.4)
            ax.annotate(
                st.loc[code, "name"], (x, y), rotation=90, fontsize=7,
                textcoords="offset points", xytext=(0, 11 if label_above else -11),
                ha="center", va="bottom" if label_above else "top", color="#22282c",
            )

    trunk_x = list(range(len(trunk)))
    branch_x0 = len(trunk)
    gore_x = [branch_x0 + i for i in range(len(goregaon))]
    pnvl_x = [branch_x0 + i for i in range(len(panvel))]

    # Connectors from the last trunk station into each branch.
    ax.plot([trunk_x[-1], gore_x[0]], [0, 1.9], color="#b9c0c5", lw=4.0, zorder=1)
    ax.plot([trunk_x[-1], pnvl_x[0]], [0, -1.9], color="#b9c0c5", lw=4.0, zorder=1)

    _draw(trunk, trunk_x, 0.0, label_above=True)
    _draw(goregaon, gore_x, 1.9, label_above=True)
    _draw(panvel, pnvl_x, -1.9, label_above=False)

    ax.text(gore_x[-1] + 0.6, 1.9, "to Goregaon", fontsize=9, fontweight="bold",
            va="center", color="#22282c")
    ax.text(pnvl_x[-1] + 0.6, -1.9, "to Panvel", fontsize=9, fontweight="bold",
            va="center", color="#22282c")
    ax.text(-0.6, 0.0, "CSMT\ntrunk", fontsize=9, fontweight="bold", ha="right",
            va="center", color="#22282c")

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=9, color=palette[c % len(palette)],
                   label=names.get(int(c), f"cluster {c}"))
        for c in sorted(set(labels))
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper center",
              ncol=min(len(handles), 4), bbox_to_anchor=(0.5, 1.14))
    ax.set_xlim(-3.0, max(pnvl_x[-1], gore_x[-1]) + 3.4)
    ax.set_ylim(-4.6, 4.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return _save(fig, "24_line_map.png", outdir)


def fig_silhouette(result, outdir: Path | None = None) -> Path:
    use_house_style()
    palette = BAND_COLORS + ["#7a5195", "#00798c", "#8d6e63"]
    sil = result.silhouette_per_station
    labels = result.labels
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    y = 0
    for c in sorted(labels.unique()):
        vals = sil[labels == c].sort_values()
        ax.barh(range(y, y + len(vals)), vals.to_numpy(), color=palette[c % len(palette)], height=0.85)
        for i, code in enumerate(vals.index):
            ax.text(0.002, y + i, code, fontsize=6, va="center", ha="left", color="#22282c")
        y += len(vals) + 1
    ax.axvline(result.silhouette, color=ACCENT, ls="--", lw=1.3,
               label=f"mean = {result.silhouette:.3f}")
    ax.set_yticks([])
    ax.set_xlabel("silhouette coefficient")
    ax.set_title("Per-station silhouette; negative bars are stations in the wrong cluster")
    ax.legend(fontsize=8)
    return _save(fig, "25_silhouette.png", outdir)


def fig_cluster_feature_value(table: pd.DataFrame, outdir: Path | None = None) -> Path:
    """Does the CO5 structure actually help the CO2 model?"""
    use_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, col, title in zip(axes, ["rmse", "exp_cost_inr"],
                              ["RMSE (standees/m²)", "Expected cost per arrival (₹)"]):
        vals = table[col].to_numpy()
        ax.bar(table.index, vals, color=[ACCENT if v == vals.min() else NEUTRAL for v in vals])
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=12, labelsize=8)
        ax.set_ylim(vals.min() * 0.97, vals.max() * 1.02)
    fig.suptitle("CO5 → CO2: feeding the cluster label back into the regression", y=1.03, fontsize=11)
    return _save(fig, "26_cluster_feature_value.png", outdir)


__all__ = [name for name in dir() if name.startswith("fig_")] + ["use_house_style", "FIGDIR"]
