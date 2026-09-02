#!/usr/bin/env python3
"""Run the whole project end to end.

    python scripts/run_all.py                 # full run (~15 min)
    python scripts/run_all.py --quick         # 60 days, fewer rounds (~5 min)

Equivalent to running 01, 02 and 03 in order; exists so that a fresh clone
reproduces every number and every figure in the report with one command.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"step failed with exit code {result.returncode}: {' '.join(cmd)}")
    print(f"\n[done in {time.time() - t0:.0f}s]", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="60 days instead of 180, fewer boosting rounds, smaller holdout")
    ap.add_argument("--skip-data", action="store_true", help="reuse the dataset already in data/")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    py = sys.executable
    days = 60 if args.quick else 180
    rounds = 300 if args.quick else 1200
    bootstrap = 15 if args.quick else 60
    # The default 18/24-day holdout is sized for a 180-day run; on a short run
    # it would leave almost nothing to train on, so scale it with the horizon.
    val_days = 8 if args.quick else 18
    test_days = 12 if args.quick else 24
    scan_bootstrap = 8 if args.quick else 25

    if not args.skip_data:
        run([py, "scripts/01_generate_data.py", "--days", str(days), "--monitored", "0.08"])
    run([py, "scripts/02_train_regression.py", "--rounds", str(rounds),
         "--val-days", str(val_days), "--test-days", str(test_days)])
    run([py, "scripts/03_cluster_stations.py", "--k", str(args.k),
         "--bootstrap", str(bootstrap), "--scan-bootstrap", str(scan_bootstrap)])

    print("\nAll steps complete.")
    print(f"  figures : {ROOT / 'reports' / 'figures'}")
    print(f"  tables  : {ROOT / 'reports' / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
