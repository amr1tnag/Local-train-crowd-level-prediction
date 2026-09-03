# Local Train Crowd-Level Prediction — Mumbai Harbour Line

A semester-end ML project predicting coach-level crowd severity on Mumbai's
Harbour Line, built as four connected components around one shared dataset:

1. **Synthetic data generation** — realistic ridership + weather + station
   metadata for the CSMT↔Panvel corridor (this phase).
2. **CO5 — Station clustering**: group stations by the *shape* of their daily
   ridership curve (K-means / hierarchical), not raw volume.
3. **CO2 — Regression**: predict `occupancy_pct` from time/weather/station
   features (Linear/Polynomial, Random Forest, XGBoost).
4. **Asymmetric-loss classifier**: bucket occupancy into
   Safe/Moderate/Full/Dangerous tiers and train a cost-sensitive classifier
   that specifically minimizes missed "Dangerous" coaches.
5. **YOLO crowd-density demo**: a small, decoupled CV component showing how
   the `occupancy_pct` target in Phases 1-4 would actually be *measured* in
   a real deployment (CCTV → person count → density → occupancy tier).

Every dataset in this repo is synthetic — there is no public Mumbai Suburban
Railway occupancy dataset. **[`DATA_GENERATION.md`](DATA_GENERATION.md)**
documents every generation assumption and its provenance (what's a real,
cited fact vs. a modeling choice) — read that first if you're evaluating
this project.

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | Synthetic data generation | ✅ Done |
| 2 | CO5 — Station clustering | ⏳ Not started |
| 3 | CO2 — Regression | ⏳ Not started |
| 4 | Asymmetric-loss classifier | ⏳ Not started |
| 5 | YOLO crowd-density demo | ⏳ Not started |

## Repository structure

```
├── DATA_GENERATION.md        # assumptions & provenance for the synthetic dataset
├── data/
│   ├── processed/            # harbour_line_ridership.parquet + a CSV sample
│   └── raw/                  # weather-cache scratch space (gitignored)
├── notebooks/
│   └── 01_eda.ipynb          # Phase 1 EDA (executed, with saved plots)
├── src/
│   └── data_generation/      # station metadata, weather, calendar, simulator
├── models/                   # saved model artifacts (later phases)
├── reports/
│   └── phase1/               # PNG plots from the Phase 1 EDA notebook
├── tests/                    # sanity tests for the data generator
├── requirements.txt
└── Makefile
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(or `make install`)

## Phase 1 — Synthetic data generation

**Station scope:** the real CSMT↔Panvel Harbour Line trunk, 25 stations,
verified against Wikipedia/route-guide sources (see `DATA_GENERATION.md`
§1) — not the fictional example list in the original brief (which included
two stations not actually on this line; corrected and documented).

**Dataset:** `data/processed/harbour_line_ridership.parquet` — ~1.04M rows:
365 days (2023) × 25 stations × 19 service hours (05:00–23:00) × 2
directions (UP = towards CSMT, DOWN = towards Panvel) × 3 coach types
(General, Ladies, First Class). A 20,000-row random sample is also kept as
CSV for quick inspection: `data/processed/harbour_line_ridership_sample.csv`.

**Columns:**

| Column | Description |
|---|---|
| `date`, `day_of_week`, `is_weekend`, `month` | Calendar |
| `is_holiday`, `holiday_name` | Maharashtra/India public holidays, 2023 |
| `is_monsoon` | month ∈ {Jun, Jul, Aug, Sep} |
| `is_mega_block_day` | simulated Sunday maintenance block |
| `station_id`, `station_name`, `station_order`, `distance_km`, `station_type`, `is_interchange` | Station metadata |
| `direction` | `UP` (CBD-bound) / `DOWN` (Panvel-bound) |
| `hour` | service hour, 5–23 |
| `coach_type` | General / Ladies / First Class |
| `rainfall_mm`, `temp_max_c`, `temp_min_c`, `temperature_c`, `rain_intensity_band` | Weather, calibrated to real IMD Mumbai climate normals |
| `rain_regime` | simulated day-level monsoon disruption state |
| `occupancy_pct` | **target** — % of rated coach capacity, 0–300 (see below) |

`occupancy_pct` is allowed to exceed 100%: Mumbai's suburban trains are
well documented to run at 2.5×–3× rated capacity ("super-dense crush load")
during peak hours — 100% means "at rated capacity", not "the ceiling". See
`DATA_GENERATION.md` §3 for the citation and the full simulation logic
(direction-dependent load buildup, station-type shape, monsoon disruption
regimes, mega-block effects, coach-type differences).

**Regenerate the dataset:**

```bash
python -m src.data_generation.generate_dataset      # full year (~1.04M rows)
python -m src.data_generation.generate_dataset --quick   # one month, fast iteration
```

Deterministic given `--seed` (default 42) — the exact file committed here
came from the default seed and date range.

**Run the EDA notebook:**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

(or open it directly in Jupyter). It validates the generator's own design
assumptions — the AM/PM directional buildup, station-type shape diversity,
day-type ordering, and monsoon disruption effects all actually appear in
the data — and saves 8 plots to `reports/phase1/`.

**Run the sanity test suite:**

```bash
pytest tests/ -q
```

Checks structural invariants (row counts, value bounds, determinism) and
the core behavioral assumptions (AM peak load builds towards CSMT for UP
trains, PM peak builds towards Panvel for DOWN trains, weekday > holiday
crowding, General > First Class crowding at peak).

## Later phases

Sections for Phases 2–5 will be added here as each is built.
