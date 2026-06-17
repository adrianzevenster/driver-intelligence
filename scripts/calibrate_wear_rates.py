#!/usr/bin/env python
"""Fit per-circuit, per-compound wear rates from real FastF1 stint data.

For each circuit and compound, measures the median non-final stint length
from actual race strategy (FastF1) and derives a wear_rate such that:
    wear_rate = wear_critical / median_stint_laps

This ensures that a typical tire reaches wear_critical (0.78) at the circuit's
actual median pit window, making the cliff projection calibrated to real
degradation rather than using global rule-of-thumb constants.

Results are saved to data/calibration/circuit_wear_rates.json and loaded
automatically by fastf1_session.py at runtime via _get_wear_rate().

Usage:
    uv run python scripts/calibrate_wear_rates.py --year 2024 --rounds 1 2 3 4 5 6 7
    uv run python scripts/calibrate_wear_rates.py --year 2023 --rounds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_OUTPUT = Path("data/calibration/circuit_wear_rates.json")
_WEAR_CRITICAL = 0.78
_MIN_STINTS = 5       # discard circuits with fewer observed stints per compound
_MIN_MEDIAN_LAPS = 8  # guard against safety-car-distorted or sprint-weekend data where
                      # median stint length collapses below what any compound can physically
                      # degrade to critical wear — such entries produce implausible rates

_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _load_session(year: int, round_num: int):
    import fastf1
    import os
    cache = os.environ.get("F1DI_FASTF1_CACHE", str(Path(__file__).parents[1] / "data" / "fastf1_cache"))
    os.makedirs(cache, exist_ok=True)
    fastf1.Cache.enable_cache(cache)
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


def collect_stint_lengths(
    year: int, rounds: list[int]
) -> dict[str, dict[str, list[int]]]:
    """Returns {track_id: {compound: [non_final_stint_lengths]}}.

    Parses stints directly from the session laps frame (one load per session,
    laps only — no telemetry) rather than going through actual_strategy which
    triggers the heavier _load_race_session with telemetry=True.
    """
    import pandas as pd
    from f1di.knowledge.track_ids import canonical as canonical_track_id

    data: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for rnd in rounds:
        print(f"  Round {rnd}...", end=" ", flush=True)
        try:
            session = _load_session(year, rnd)
            location = str(session.event.get("Location", "unknown"))
            track_id = canonical_track_id(location)
            laps = session.laps
        except Exception as exc:
            print(f"{_YELLOW}skip: {exc}{_RESET}")
            continue

        n_stints = 0
        for driver in laps["Driver"].dropna().unique():
            dlaps = laps.pick_drivers(driver).sort_values("LapNumber")
            if "Stint" not in dlaps.columns:
                continue
            driver_stints = sorted(dlaps["Stint"].dropna().unique())
            for i, stint_n in enumerate(driver_stints):
                is_final = (i == len(driver_stints) - 1)
                if is_final:
                    continue  # skip: driver never pits out, can't observe wear limit
                grp = dlaps[dlaps["Stint"] == stint_n]
                compound = str(grp["Compound"].iloc[0]).upper() if "Compound" in grp.columns else "UNKNOWN"
                if compound not in ("SOFT", "MEDIUM", "HARD"):
                    continue
                tl = grp["TyreLife"].max()
                stint_laps = int(tl) if pd.notna(tl) else len(grp)
                if 5 < stint_laps < 60:
                    data[track_id][compound].append(stint_laps)
                    n_stints += 1

        print(f"track={track_id}  stints={n_stints}")

    return {k: dict(v) for k, v in data.items()}


def fit_wear_rates(
    stint_data: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, float]]:
    """Derive per-circuit wear rates from median stint lengths."""
    from f1di.knowledge.fastf1_session import _WEAR_RATE as GLOBAL

    rates: dict[str, dict[str, float]] = {}
    for track_id, compounds in stint_data.items():
        circuit_rates: dict[str, float] = {}
        for compound, lengths in compounds.items():
            if len(lengths) < _MIN_STINTS:
                continue
            med = median(lengths)
            if med < _MIN_MEDIAN_LAPS:
                print(
                    f"    {track_id:20s} {compound:12s}  "
                    f"{_YELLOW}SKIP: median_laps={med:.1f} < {_MIN_MEDIAN_LAPS} "
                    f"(safety-car/sprint distortion suspected){_RESET}"
                )
                continue
            fitted = round(_WEAR_CRITICAL / med, 4)
            global_rate = GLOBAL.get(compound, 0.030)
            delta_pct = (fitted - global_rate) / global_rate * 100
            colour = _GREEN if abs(delta_pct) < 20 else _YELLOW
            print(
                f"    {track_id:20s} {compound:12s}  "
                f"median_laps={med:.1f}  "
                f"wear_rate={fitted:.4f}  "
                f"vs_global={colour}{delta_pct:+.0f}%{_RESET}"
            )
            circuit_rates[compound] = fitted
        if circuit_rates:
            rates[track_id] = circuit_rates

    return rates


def main():
    parser = argparse.ArgumentParser(description="Calibrate per-circuit tire wear rates from FastF1 stint data.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--rounds", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--output", default=str(_OUTPUT))
    args = parser.parse_args()

    print(f"\nCollecting stint data: {args.year} rounds {args.rounds}")
    stint_data = collect_stint_lengths(args.year, args.rounds)

    print("\nFitting wear rates:")
    rates = fit_wear_rates(stint_data)

    # Merge with existing rates (so a partial re-run doesn't wipe other circuits)
    out_path = Path(args.output)
    existing: dict[str, dict[str, float]] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            pass
    merged = {**existing, **rates}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    print(f"\nSaved {len(merged)} circuit entries to {args.output}")

    # Invalidate fastf1_session.py's in-memory cache so next import picks up new rates
    try:
        import f1di.knowledge.fastf1_session as _fs
        _fs._circuit_wear_rates = None
    except Exception:
        pass


if __name__ == "__main__":
    main()
