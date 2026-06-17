#!/usr/bin/env python
"""Backtest the undercut model against real pit-stop and position data.

Two checks:

1. PIT_LOSS_S calibration
   Measures actual pit-lane traversal time (PitInTime → PitOutTime) from
   FastF1 per circuit and compares to the 22.0 s constant in undercut.py.
   This tells you whether the constant is systematically wrong for any circuit.

2. Undercut success calibration
   Identifies cases where driver A pitted and driver B (who was ≤3 positions
   ahead within 2 seconds on track) pitted within 3 laps. Runs
   undercut_window() at the moment of A's pit and compares
   undercut_success_probability against the actual position outcome after
   both drivers completed their out-laps (+3 settling laps).
   Reports a Brier score and a calibration curve binned by probability decile.

Usage:
    uv run python scripts/backtest_undercut.py --year 2024 --rounds 1 2 3 4 5
    uv run python scripts/backtest_undercut.py --year 2024 --rounds 1 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"

# Settling laps to wait after the later driver's out-lap before reading positions.
_SETTLE_LAPS = 3
# Maximum lap gap between the two drivers' pits for an "undercut scenario"
_MAX_PIT_LAP_GAP = 3
# Position gap threshold: B must be within this many positions ahead of A
_MAX_POS_GAP = 3


def _load_session(year: int, round_num: int):
    import fastf1
    import os
    cache = os.environ.get("F1DI_FASTF1_CACHE", str(Path(__file__).parents[1] / "data" / "fastf1_cache"))
    os.makedirs(cache, exist_ok=True)
    fastf1.Cache.enable_cache(cache)
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


def _pit_times(session) -> list[dict]:
    """Collect (driver, circuit, pit_lap, pit_time_s) for every pit stop."""
    import pandas as pd
    laps = session.laps
    circuit = str(session.event.get("Location", "unknown"))
    records = []
    for driver in laps["Driver"].dropna().unique():
        dlaps = laps.pick_drivers(driver).sort_values("LapNumber")
        for i, (_, row) in enumerate(dlaps.iterrows()):
            pit_in = row.get("PitInTime")
            if pd.isna(pit_in):
                continue
            # PitOutTime is on the following lap row
            if i + 1 >= len(dlaps):
                continue
            next_row = dlaps.iloc[i + 1]
            pit_out = next_row.get("PitOutTime")
            if pd.isna(pit_out):
                continue
            try:
                pit_s = (pit_out - pit_in).total_seconds()
            except Exception:
                continue
            if 10 < pit_s < 60:  # sanity: real pit stops are 10-50 s
                records.append({
                    "circuit": circuit,
                    "driver": str(driver),
                    "pit_lap": int(row["LapNumber"]),
                    "pit_time_s": round(pit_s, 2),
                })
    return records


def _position_at(laps, driver: str, lap: int) -> int | None:
    """Race position for driver at given lap number."""
    import pandas as pd
    dlaps = laps.pick_drivers(driver)
    match = dlaps[dlaps["LapNumber"] == lap]
    if match.empty:
        return None
    pos = match.iloc[0].get("Position")
    return int(pos) if not pd.isna(pos) else None


def _undercut_scenarios(session, year: int, round_num: int, verbose: bool) -> list[dict]:
    """Find undercut attempts and evaluate model predictions vs actual outcomes."""
    import pandas as pd

    laps = session.laps
    drivers = list(laps["Driver"].dropna().unique())
    # Build {driver: [(pit_lap, out_lap)]} mapping
    pit_map: dict[str, list[tuple[int, int]]] = {}
    for driver in drivers:
        dlaps = laps.pick_drivers(driver).sort_values("LapNumber")
        stops = []
        for i, (_, row) in enumerate(dlaps.iterrows()):
            if pd.isna(row.get("PitInTime")):
                continue
            pit_lap = int(row["LapNumber"])
            out_lap = pit_lap + 1
            stops.append((pit_lap, out_lap))
        pit_map[driver] = stops

    records = []
    # For each pit stop by A, look for B who pitted shortly after
    for a in drivers:
        for a_pit, a_out in pit_map.get(a, []):
            pos_a_before = _position_at(laps, a, a_pit - 1)
            if pos_a_before is None:
                continue
            for b in drivers:
                if b == a:
                    continue
                for b_pit, b_out in pit_map.get(b, []):
                    if not (1 <= b_pit - a_pit <= _MAX_PIT_LAP_GAP):
                        continue
                    pos_b_before = _position_at(laps, b, a_pit - 1)
                    if pos_b_before is None:
                        continue
                    # B must be ahead of A (lower position number), close
                    if not (0 < pos_a_before - pos_b_before <= _MAX_POS_GAP):
                        continue

                    settle_lap = b_out + _SETTLE_LAPS
                    pos_a_after = _position_at(laps, a, settle_lap)
                    pos_b_after = _position_at(laps, b, settle_lap)
                    if pos_a_after is None or pos_b_after is None:
                        continue

                    # Success: A is now ahead of B (lower position number)
                    actual_success = 1 if pos_a_after < pos_b_after else 0

                    try:
                        from f1di.strategy.undercut import undercut_window
                        result = undercut_window(year, round_num, a, b, a_pit)
                        predicted_prob = result["undercut_success_probability"]
                    except Exception as exc:
                        if verbose:
                            print(f"  {_YELLOW}undercut_window({a},{b},{a_pit}) failed: {exc}{_RESET}")
                        continue

                    records.append({
                        "year": year,
                        "round": round_num,
                        "driver": a,
                        "rival": b,
                        "driver_pit_lap": a_pit,
                        "rival_pit_lap": b_pit,
                        "pos_driver_before": pos_a_before,
                        "pos_rival_before": pos_b_before,
                        "pos_driver_after": pos_a_after,
                        "pos_rival_after": pos_b_after,
                        "predicted_prob": round(predicted_prob, 4),
                        "actual_success": actual_success,
                    })
                    if verbose:
                        outcome = f"{_GREEN}SUCCESS{_RESET}" if actual_success else f"{_RED}FAIL{_RESET}"
                        print(
                            f"  {a} pits L{a_pit} (P{pos_a_before}) vs {b} L{b_pit} (P{pos_b_before}) "
                            f"→ pred={predicted_prob:.2f} actual={outcome}"
                        )
    return records


def _summarise_pit_times(pit_records: list[dict]) -> dict:
    from statistics import mean, median, stdev
    from f1di.strategy.undercut import PIT_LOSS_S

    if not pit_records:
        return {}

    all_times = [r["pit_time_s"] for r in pit_records]
    by_circuit: dict[str, list[float]] = {}
    for r in pit_records:
        by_circuit.setdefault(r["circuit"], []).append(r["pit_time_s"])

    summary = {
        "model_constant_s": PIT_LOSS_S,
        "n_stops": len(all_times),
        "overall_median_s": round(median(all_times), 2),
        "overall_mean_s": round(mean(all_times), 2),
        "overall_std_s": round(stdev(all_times), 2) if len(all_times) > 1 else None,
        "per_circuit": {
            c: {
                "n": len(v),
                "median_s": round(median(v), 2),
                "mean_s": round(mean(v), 2),
                "bias_vs_model_s": round(median(v) - PIT_LOSS_S, 2),
            }
            for c, v in sorted(by_circuit.items())
        },
    }
    return summary


def _summarise_undercut(records: list[dict]) -> dict:
    if not records:
        return {}

    n = len(records)
    actual = [r["actual_success"] for r in records]
    pred = [r["predicted_prob"] for r in records]

    brier = sum((p - a) ** 2 for p, a in zip(pred, actual)) / n
    actual_rate = sum(actual) / n

    # Calibration by decile
    bins: dict[int, list] = {i: [] for i in range(10)}
    for p, a in zip(pred, actual):
        b = min(9, int(p * 10))
        bins[b].append((p, a))

    calibration = {}
    for b, pairs in bins.items():
        if not pairs:
            continue
        lo, hi = b / 10, (b + 1) / 10
        mean_pred = sum(p for p, _ in pairs) / len(pairs)
        mean_actual = sum(a for _, a in pairs) / len(pairs)
        calibration[f"{lo:.1f}-{hi:.1f}"] = {
            "n": len(pairs),
            "mean_predicted": round(mean_pred, 3),
            "mean_actual": round(mean_actual, 3),
        }

    return {
        "n_scenarios": n,
        "actual_success_rate": round(actual_rate, 3),
        "brier_score": round(brier, 4),
        "calibration_by_decile": calibration,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest undercut model against FastF1 data.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1])
    parser.add_argument("--output", default="data/calibration/undercut_backtest.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    all_pit_records: list[dict] = []
    all_undercut_records: list[dict] = []

    for rnd in args.rounds:
        print(f"\n{'='*60}")
        print(f"Round {rnd} ({args.year})")
        print(f"{'='*60}")
        try:
            session = _load_session(args.year, rnd)
        except Exception as exc:
            print(f"{_YELLOW}  Skip: {exc}{_RESET}")
            continue

        pit_recs = _pit_times(session)
        all_pit_records.extend(pit_recs)
        print(f"  Pit stops found: {len(pit_recs)}")

        print("  Evaluating undercut scenarios...")
        uc_recs = _undercut_scenarios(session, args.year, rnd, args.verbose)
        all_undercut_records.extend(uc_recs)
        print(f"  Undercut scenarios evaluated: {len(uc_recs)}")

    pit_summary = _summarise_pit_times(all_pit_records)
    uc_summary = _summarise_undercut(all_undercut_records)

    output = {
        "pit_time_calibration": pit_summary,
        "undercut_success_calibration": uc_summary,
        "records": all_undercut_records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))

    print(f"\n{'='*60}")
    print("PIT-TIME CALIBRATION")
    print(f"{'='*60}")
    if pit_summary:
        from f1di.strategy.undercut import PIT_LOSS_S
        med = pit_summary["overall_median_s"]
        bias = med - PIT_LOSS_S
        colour = _GREEN if abs(bias) < 2 else (_YELLOW if abs(bias) < 5 else _RED)
        print(f"  Model constant: {PIT_LOSS_S:.1f} s")
        print(f"  Measured median: {med:.2f} s  bias={colour}{bias:+.2f} s{_RESET}")
        print(f"  n stops: {pit_summary['n_stops']}")
        for c, d in pit_summary.get("per_circuit", {}).items():
            b = d["bias_vs_model_s"]
            col = _GREEN if abs(b) < 2 else (_YELLOW if abs(b) < 5 else _RED)
            print(f"    {c}: median={d['median_s']:.1f}s n={d['n']} bias={col}{b:+.1f}s{_RESET}")
    else:
        print("  (no data)")

    print(f"\n{'='*60}")
    print("UNDERCUT CALIBRATION")
    print(f"{'='*60}")
    if uc_summary:
        brier = uc_summary["brier_score"]
        col = _GREEN if brier < 0.20 else (_YELLOW if brier < 0.25 else _RED)
        print(f"  n scenarios: {uc_summary['n_scenarios']}")
        print(f"  Actual success rate: {uc_summary['actual_success_rate']:.1%}")
        print(f"  Brier score: {col}{brier:.4f}{_RESET}  (0.25 = coin-flip, lower is better)")
        print("\n  Calibration by predicted probability decile:")
        print(f"  {'Bin':<12} {'n':>5} {'pred':>8} {'actual':>8}")
        for b, d in uc_summary["calibration_by_decile"].items():
            print(f"  {b:<12} {d['n']:>5} {d['mean_predicted']:>8.3f} {d['mean_actual']:>8.3f}")
    else:
        print("  (no scenarios found — check position data availability)")

    print(f"\nOutput written to {args.output}")


if __name__ == "__main__":
    main()
