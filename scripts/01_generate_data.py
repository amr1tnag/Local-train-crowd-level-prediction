#!/usr/bin/env python3
"""Generate the synthetic Harbour-line dataset.

    python scripts/01_generate_data.py --days 180 --out data

Writes (gzipped CSV so the repo stays cloneable):
    data/coach_observations.csv.gz   CO2 supervised table
    data/station_hour_flows.csv.gz   CO5 clustering table
    data/service_records.csv.gz      train-level diagnostics
    data/weather.csv.gz, data/calendar.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mumbai_crowd.config import SimConfig
from mumbai_crowd.simulate import simulate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=SimConfig.n_days)
    ap.add_argument("--start", type=str, default=SimConfig.start_date)
    ap.add_argument("--seed", type=int, default=SimConfig.seed)
    ap.add_argument("--monitored", type=float, default=SimConfig.monitored_service_fraction,
                    help="fraction of services carrying coach-level instrumentation")
    ap.add_argument("--riders-scale", type=float, default=1.0,
                    help="multiply all demand (>1 stress-tests the danger regime)")
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()

    cfg = SimConfig(
        start_date=args.start,
        n_days=args.days,
        seed=args.seed,
        monitored_service_fraction=args.monitored,
        daily_riders_scale=args.riders_scale,
        out_dir=args.out,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {cfg.n_days} days from {cfg.start_date} (seed={cfg.seed}) ...")
    result = simulate(cfg)

    result.coach_observations.to_csv(out_dir / "coach_observations.csv.gz", index=False)
    result.station_hour_flows.to_csv(out_dir / "station_hour_flows.csv.gz", index=False)
    result.service_records.to_csv(out_dir / "service_records.csv.gz", index=False)
    result.weather.to_csv(out_dir / "weather.csv.gz", index=False)
    result.calendar.to_csv(out_dir / "calendar.csv", index=False)

    print()
    print(result.summary())
    print()
    for f in sorted(out_dir.iterdir()):
        print(f"  wrote {f}  ({f.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
