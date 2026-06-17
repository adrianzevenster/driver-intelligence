#!/usr/bin/env python
"""Backtest the tire-cliff Monte Carlo projection (Phase 1) against real pit laps.

For each driver/race/stint (excluding each driver's final stint, since there's
no subsequent pit to compare a projection against), replays every lap of the
stint through project_cliff_for_window() and records, whenever the projection
is confident enough to produce an eta_laps:
  - predicted_pit_lap = lap + eta_laps
  - actual_pit_lap     = the real lap the driver pitted on
  - error_laps         = predicted - actual

This is the same discipline as the CV/regression-guard work elsewhere in this
repo: report calibration honestly rather than assume a projection that looks
reasonable on a couple of hand-picked laps actually predicts real pit stops.
If the model is bad, this script is what tells you that before it ships as
a driver-facing number.

Usage:
    uv run python scripts/backtest_cliff_projection.py --year 2024 --rounds 1 2 3 4 5
    uv run python scripts/backtest_cliff_projection.py --year 2024 --rounds 1 --drivers VER HAM
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def backtest_race(year: int, round_num: int, session_type: str = "R", drivers: list[str] | None = None) -> list[dict]:
    from f1di.agents.thresholds import get as get_thresholds
    from f1di.agents.tire_projection import project_cliff_for_window
    from f1di.features.extractor import extract_features
    from f1di.knowledge.fastf1_session import actual_strategy, build_all_lap_windows, get_drivers

    codes = drivers or [d["code"] for d in get_drivers(year, round_num, session_type=session_type)]
    records: list[dict] = []

    for code in codes:
        try:
            stints = actual_strategy(year, round_num, code, session_type=session_type)
            windows = build_all_lap_windows(year, round_num, code, session_type=session_type)
        except Exception as exc:
            print(f"  {_YELLOW}skip {code}: {exc}{_RESET}")
            continue
        if len(stints) < 2 or not windows:
            continue  # need at least one real pit to backtest against

        t = None
        for stint in stints[:-1]:
            actual_pit_lap = stint["end_lap"] + 1
            for lap in range(stint["start_lap"], stint["end_lap"] + 1):
                window = windows.get(lap)
                if window is None:
                    continue
                features = extract_features(window)
                if t is None:
                    t = get_thresholds(window.track_id)
                cliff = project_cliff_for_window(window, features, t.wear_critical)
                record = {
                    "year": year, "round": round_num, "driver": code,
                    "stint": stint["stint"], "lap": lap,
                    "laps_before_pit": actual_pit_lap - lap,
                    "confident": cliff["eta_laps"] is not None,
                }
                if cliff["eta_laps"] is not None:
                    predicted_pit_lap = lap + cliff["eta_laps"]
                    record["predicted_pit_lap"] = predicted_pit_lap
                    record["actual_pit_lap"] = actual_pit_lap
                    record["error_laps"] = predicted_pit_lap - actual_pit_lap
                records.append(record)
    return records


def summarize(records: list[dict]) -> dict:
    import numpy as np

    n_total = len(records)
    confident = [r for r in records if r["confident"]]
    n_confident = len(confident)

    if not confident:
        return {
            "n_laps_evaluated": n_total,
            "n_confident_calls": 0,
            "confidence_rate": 0.0,
            "message": "No confident cliff calls were produced — cannot evaluate accuracy.",
        }

    errors = np.array([r["error_laps"] for r in confident])
    abs_errors = np.abs(errors)

    # The most operationally relevant signal: the LAST confident call made
    # before each stint actually ended (closest-to-real-time prediction).
    last_call_per_stint: dict[tuple, dict] = {}
    for r in confident:
        key = (r["year"], r["round"], r["driver"], r["stint"])
        if key not in last_call_per_stint or r["lap"] > last_call_per_stint[key]["lap"]:
            last_call_per_stint[key] = r
    last_call_errors = np.array([abs(r["error_laps"]) for r in last_call_per_stint.values()])

    return {
        "n_laps_evaluated": n_total,
        "n_confident_calls": n_confident,
        "confidence_rate": round(n_confident / n_total, 4) if n_total else 0.0,
        "n_stints_with_a_confident_call": len(last_call_per_stint),
        "all_calls": {
            "median_abs_error_laps": round(float(np.median(abs_errors)), 2),
            "mean_signed_error_laps": round(float(np.mean(errors)), 2),
            "within_2_laps_pct": round(float(np.mean(abs_errors <= 2)) * 100, 1),
            "within_3_laps_pct": round(float(np.mean(abs_errors <= 3)) * 100, 1),
            "within_5_laps_pct": round(float(np.mean(abs_errors <= 5)) * 100, 1),
        },
        "last_confident_call_per_stint": {
            "median_abs_error_laps": round(float(np.median(last_call_errors)), 2),
            "within_2_laps_pct": round(float(np.mean(last_call_errors <= 2)) * 100, 1),
            "within_3_laps_pct": round(float(np.mean(last_call_errors <= 3)) * 100, 1),
            "within_5_laps_pct": round(float(np.mean(last_call_errors <= 5)) * 100, 1),
        },
    }


def main(year: int, rounds: list[int], drivers: list[str] | None, output: Path) -> None:
    all_records: list[dict] = []
    for round_num in rounds:
        print(f"\nBacktesting {year} R{round_num}...")
        records = backtest_race(year, round_num, drivers=drivers)
        all_records.extend(records)
        print(f"  {len(records)} (lap, stint) observations")

    summary = summarize(all_records)
    print(f"\n{'='*60}")
    print(json.dumps(summary, indent=2))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "records": all_records}, indent=2))
    print(f"\nFull results written to {output}")

    if summary.get("n_confident_calls", 0) == 0:
        print(f"{_YELLOW}No confident calls produced — projection is too conservative to evaluate.{_RESET}")
    else:
        pct = summary["last_confident_call_per_stint"]["within_3_laps_pct"]
        color = _GREEN if pct >= 50 else _YELLOW
        print(f"{color}Last confident call before each pit was within 3 laps {pct}% of the time.{_RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the tire-cliff Monte Carlo projection")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--drivers", nargs="+", default=None, help="Limit to specific driver codes")
    parser.add_argument("--output", type=Path, default=Path("data/calibration/cliff_projection_backtest.json"))
    args = parser.parse_args()
    main(args.year, args.rounds, args.drivers, args.output)
