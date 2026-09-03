"""
CLI entry point: generates the full synthetic Harbour Line ridership
dataset and writes it to data/processed/.

Usage:
    python -m src.data_generation.generate_dataset [--quick]

--quick restricts the date range to one month, for fast iteration while
developing the pipeline (~85k rows instead of ~1M). The committed dataset
in this repo is generated with the full year.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .calendar_features import build_calendar
from .weather import generate_weather
from .simulate import build_grid, simulate_occupancy, compute_monsoon_regime

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"

FINAL_COLUMNS = [
    "date", "day_of_week", "is_weekend", "month", "is_holiday", "holiday_name",
    "is_monsoon", "is_mega_block_day",
    "station_id", "station_name", "station_order", "distance_km", "station_type",
    "is_interchange", "direction", "hour", "coach_type",
    "rainfall_mm", "temp_max_c", "temp_min_c", "temperature_c", "rain_intensity_band",
    "rain_regime", "occupancy_pct",
]


def generate(start: str, end: str, seed: int) -> pd.DataFrame:
    calendar_df = build_calendar(start=start, end=end, seed=seed)
    weather_df = generate_weather(pd.DatetimeIndex(calendar_df["date"]), seed=seed + 1)
    calendar_df = calendar_df.merge(weather_df, on="date", how="left")

    regime_df = compute_monsoon_regime(weather_df, seed=seed + 2)
    calendar_df = calendar_df.merge(regime_df, on="date", how="left")

    grid = build_grid(calendar_df)
    result = simulate_occupancy(grid, seed=seed + 3)
    return result[FINAL_COLUMNS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Generate one month only, for fast dev iteration.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rows", type=int, default=20000)
    args = parser.parse_args()

    end = "2023-01-31" if args.quick else args.end

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating dataset from {args.start} to {end} (seed={args.seed})...")
    df = generate(args.start, end, args.seed)
    print(f"Generated {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory.")

    parquet_path = PROCESSED_DIR / "harbour_line_ridership.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    sample_path = PROCESSED_DIR / "harbour_line_ridership_sample.csv"
    sample_n = min(args.sample_rows, len(df))
    df.sample(n=sample_n, random_state=args.seed).sort_values(
        ["date", "station_order", "hour", "direction", "coach_type"]
    ).to_csv(sample_path, index=False)
    print(f"Wrote {sample_path} ({sample_path.stat().st_size / 1e6:.1f} MB, {sample_n:,} rows)")

    print("\nSample rows:")
    print(df.sample(5, random_state=args.seed).to_string(index=False))

    print("\nOccupancy summary by station_type:")
    print(df.groupby("station_type")["occupancy_pct"].describe()[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
