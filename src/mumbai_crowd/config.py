"""Central configuration: crowd taxonomy, cost model and simulation knobs.

The single most important object here is :data:`COST_MATRIX`.  Everything the
project claims about "asymmetric loss" is grounded in it, so it lives in one
place, is expressed in one unit (rupees of expected social + operational
cost per coach-arrival), and every model is scored against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 1. Crowd-level taxonomy
# ---------------------------------------------------------------------------
# Density is measured in **standees per m^2 of usable standing floor area**
# inside a single coach (seated passengers do not contribute).  Cut-points
# follow Indian Railways' own suburban load taxonomy -- normal crush ~6/m^2,
# dense crush ~10/m^2, super-dense crush ~14-16/m^2 -- rather than the Fruin /
# TCQSM scale, which tops out long before a Mumbai local does.

CROWD_BANDS: list[tuple[str, float, float, str]] = [
    # (label, lower bound inclusive, upper bound exclusive, operator action)
    ("COMFORTABLE", 0.0, 4.0, "no action"),
    ("BUSY",        4.0, 8.0, "advisory on PIS displays"),
    ("CRUSH",       8.0, 12.0, "platform marshalling, hold doors, RPF on the FOB"),
    ("DANGEROUS",  12.0, np.inf, "inject relief service, gate-control station entry"),
]

BAND_LABELS: list[str] = [b[0] for b in CROWD_BANDS]
BAND_EDGES: np.ndarray = np.array([b[1] for b in CROWD_BANDS[1:]], dtype=float)  # [4, 8, 12]

#: A coach at or above this density is treated as a *safety event*: this is
#: IR's "super-dense crush load", the regime in which people ride footboards
#: and fall from moving trains.
DANGER_DENSITY: float = 12.0
#: Anything below this is what an operator would casually call "safe".
SAFE_DENSITY: float = 8.0


def density_to_band(density: np.ndarray | float) -> np.ndarray:
    """Map continuous density (pax/m^2) to a crowd-band index 0..3."""
    return np.searchsorted(BAND_EDGES, np.asarray(density, dtype=float), side="right")


def band_name(idx: int) -> str:
    return BAND_LABELS[int(idx)]


# ---------------------------------------------------------------------------
# 2. The cost model that makes the loss asymmetric
# ---------------------------------------------------------------------------
# Rows = true band, columns = predicted band.  Units: indicative rupees of
# expected cost per coach-arrival.
#
# Reading the matrix:
#   * Over-prediction (upper triangle) costs *operational* money: an
#     unnecessary relief rake, staff pulled to a platform, a false advisory
#     that erodes commuter trust.  Real, bounded, recoverable.
#   * Under-prediction (lower triangle) costs *safety*: nobody is sent, the
#     footboard fills, and the Mumbai suburban system's ~2,000 deaths a year
#     get another entry.  Calling a DANGEROUS coach COMFORTABLE is the single
#     worst cell in the matrix and is priced two orders of magnitude above the
#     symmetric mistake in the opposite corner.
COST_MATRIX: np.ndarray = np.array(
    [
        #  pred: COMF   BUSY   CRUSH   DANGER      <- true:
        [      0.0,    60.0,   400.0,  1500.0],   # COMFORTABLE
        [    350.0,     0.0,   180.0,   900.0],   # BUSY
        [   4200.0,  1800.0,     0.0,   500.0],   # CRUSH
        [  22000.0, 12000.0,  3000.0,     0.0],   # DANGEROUS
    ],
    dtype=float,
)

#: Marginal cost of a 1 standee/m^2 *under*-estimate, used by the continuous losses.
COST_UNDER: float = 6.0
#: Marginal cost of a 1 standee/m^2 *over*-estimate.
COST_OVER: float = 1.0


def asymmetry_ratio() -> float:
    """How much worse an under-estimate is than an over-estimate."""
    return COST_UNDER / COST_OVER


def optimal_quantile() -> float:
    """Pinball-loss quantile implied by the cost ratio.

    Under a piecewise-linear cost with slopes ``COST_UNDER`` below the truth
    and ``COST_OVER`` above it, the cost-minimising point forecast is the
    ``tau``-quantile of the predictive distribution with
    ``tau = c_under / (c_under + c_over)``.  This is the theoretical bridge
    between "safety matters more" and a concrete training objective.
    """
    return COST_UNDER / (COST_UNDER + COST_OVER)


# ---------------------------------------------------------------------------
# 3. Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Knobs for the synthetic Harbour-line demand simulator."""

    start_date: str = "2024-06-01"      # starts inside the monsoon on purpose
    n_days: int = 120
    seed: int = 20240601

    # Service pattern
    service_start_hour: int = 4          # first local ~04:00
    service_end_hour: int = 24           # last local just before midnight
    peak_headway_min: float = 4.0
    offpeak_headway_min: float = 10.0
    dwell_seconds: float = 25.0
    run_speed_kmph: float = 33.0         # incl. dwell, matches ~90 min CSMT-Panvel

    # Rake composition
    rake_lengths: tuple[int, ...] = (12, 15)
    rake_length_weights: tuple[float, ...] = (0.82, 0.18)

    # Demand scale
    daily_riders_scale: float = 1.0      # multiply all demand (stress testing)
    ladies_share: float = 0.215          # share of demand using ladies coaches
    first_class_share: float = 0.045
    ladies_first_share: float = 0.020    # the rest is the "general" pool

    # Observation model: only a fraction of rakes carry the CCTV / load-cell
    # instrumentation that gives us coach-level ground truth.
    monitored_service_fraction: float = 0.12
    density_measurement_noise: float = 0.16   # sd of multiplicative sensor error

    # Weather
    monsoon_months: tuple[int, ...] = (6, 7, 8, 9)

    # Disruption
    p_megablock_sunday: float = 0.55
    p_service_disruption: float = 0.035

    # Output
    out_dir: str = "data"


@dataclass
class ModelConfig:
    """Knobs for the CO2 regression experiments."""

    test_days: int = 24            # final N days held out (temporal split)
    val_days: int = 18             # days before that, used for tuning/calibration
    n_estimators: int = 1200       # generous; early stopping decides the real number
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 40
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    random_state: int = 7
    linex_a: float = 0.30          # LINEX asymmetry parameter
    asym_weight_under: float = field(default_factory=lambda: COST_UNDER)
    asym_weight_over: float = field(default_factory=lambda: COST_OVER)


@dataclass
class ClusterConfig:
    """Knobs for the CO5 station-profile clustering."""

    k_range: tuple[int, ...] = tuple(range(2, 11))
    random_state: int = 7
    n_init: int = 25
    pca_components: int = 2
    bootstrap_runs: int = 60       # for stability (ARI) analysis


__all__ = [
    "BAND_EDGES",
    "BAND_LABELS",
    "COST_MATRIX",
    "COST_OVER",
    "COST_UNDER",
    "CROWD_BANDS",
    "ClusterConfig",
    "DANGER_DENSITY",
    "ModelConfig",
    "SAFE_DENSITY",
    "SimConfig",
    "asymmetry_ratio",
    "band_name",
    "density_to_band",
    "optimal_quantile",
]
