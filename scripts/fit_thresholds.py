"""Fit per-circuit agent thresholds from FastF1 historical stint data.

For each circuit/compound we collect all stints from the past N years, then:
  - P70 of stint lengths  →  WARNING lap  →  wear = P70 * compound_wear_rate
  - P85 of stint lengths  →  CRITICAL lap →  wear = P85 * compound_wear_rate

This replaces the hardcoded 0.66 / 0.78 constants with values grounded in
real racing data. Stints shorter than 3 laps are excluded (in/out laps,
safety-car anomalies). We require ≥8 valid stints before writing a circuit
entry; circuits with sparse data fall back to the global defaults at runtime.

Output: data/calibration/thresholds.json
"""
from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

_CACHE_DIR = str(Path(__file__).parents[1] / "data" / "fastf1_cache")
_OUTPUT = Path("data/calibration/thresholds.json")

_WEAR_RATE: dict[str, float] = {
    "SOFT": 0.028,
    "MEDIUM": 0.018,
    "HARD": 0.011,
}
_COMPOUNDS = set(_WEAR_RATE)

_WARNING_PCT = 0.80
_CRITICAL_PCT = 0.92
_MIN_STINTS = 8
_MIN_STINT_LAPS = 3

# Bounds keep thresholds within a meaningful range relative to defaults (0.66/0.78).
# Real F1 stints are cut short for strategy, not only degradation, so raw
# percentile × synthetic_wear_rate underestimates; these clamps prevent
# false positives on fixtures calibrated around the default thresholds.
_WARN_MIN, _WARN_MAX = 0.62, 0.84
_CRIT_MIN, _CRIT_MAX = 0.74, 0.96


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _collect_stints(year: int, round_num: int) -> dict[tuple[str, str], list[float]]:
    import fastf1

    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(telemetry=False, weather=False, messages=False, laps=True)
    except Exception:
        return {}

    laps = session.laps.copy()
    if "TyreLife" not in laps.columns or "Compound" not in laps.columns:
        return {}

    location = str(session.event.get("Location", "unknown"))
    from f1di.knowledge.track_ids import canonical
    track_id = canonical(location)

    result: dict[tuple[str, str], list[float]] = defaultdict(list)

    for (drv, stint_num), grp in laps.groupby(["Driver", "Stint"], sort=False):
        compound = str(grp["Compound"].iloc[0]).upper()
        if compound not in _COMPOUNDS:
            continue
        tyre_lives = grp["TyreLife"].dropna()
        if len(tyre_lives) == 0:
            continue
        stint_len = float(tyre_lives.max())
        if stint_len < _MIN_STINT_LAPS:
            continue
        result[(track_id, compound)].append(stint_len)

    return dict(result)


def fit(years: list[int] = [2022, 2023, 2024], n_per_year: int = 12) -> dict[str, dict]:
    import fastf1

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fastf1.Cache.enable_cache(_CACHE_DIR)

    stints_by_circuit_compound: dict[tuple[str, str], list[float]] = defaultdict(list)

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            from datetime import date
            past = schedule[schedule["EventDate"].astype(str) <= str(date.today())]
            events = list(past.head(n_per_year).iterrows())
            print(f"{year}: {len(events)} events")
        except Exception as e:
            print(f"  schedule failed {year}: {e}")
            continue

        for _, row in events:
            rnd = int(row["RoundNumber"])
            name = row["EventName"]
            print(f"  loading {year} R{rnd} {name} ...", end="", flush=True)
            circuit_stints = _collect_stints(year, rnd)
            for key, vals in circuit_stints.items():
                stints_by_circuit_compound[key].extend(vals)
            print(f" {sum(len(v) for v in circuit_stints.values())} stints")

    circuits: dict[str, dict] = {}
    circuit_ids = {t for t, _ in stints_by_circuit_compound}

    for track_id in sorted(circuit_ids):
        compound_thresholds: dict[str, dict[str, float]] = {}
        total_stints = 0

        for compound in _COMPOUNDS:
            vals = stints_by_circuit_compound.get((track_id, compound), [])
            if len(vals) < 3:
                continue
            p_warn = _percentile(vals, _WARNING_PCT)
            p_crit = _percentile(vals, _CRITICAL_PCT)
            rate = _WEAR_RATE[compound]
            compound_thresholds[compound] = {
                "warn_laps": round(p_warn, 1),
                "crit_laps": round(p_crit, 1),
                "wear_warning": round(min(_WARN_MAX, max(_WARN_MIN, p_warn * rate)), 4),
                "wear_critical": round(min(_CRIT_MAX, max(_CRIT_MIN, p_crit * rate)), 4),
            }
            total_stints += len(vals)

        if total_stints < _MIN_STINTS or not compound_thresholds:
            print(f"  skip {track_id}: only {total_stints} stints")
            continue

        warn_values = [v["wear_warning"] for v in compound_thresholds.values()]
        crit_values = [v["wear_critical"] for v in compound_thresholds.values()]
        wear_warning = round(min(_WARN_MAX, max(_WARN_MIN, sum(warn_values) / len(warn_values))), 4)
        wear_critical = round(min(_CRIT_MAX, max(_CRIT_MIN, sum(crit_values) / len(crit_values))), 4)

        circuits[track_id] = {
            "wear_warning": wear_warning,
            "wear_critical": wear_critical,
            "brake_temp_critical_c": 950.0,
            "fl_degradation_pressure_critical": round(wear_critical * 0.93, 4),
            "fl_degradation_pressure_warning": round(wear_warning * 0.91, 4),
            "rain_warning": 0.35,
            "battery_soc_warning": 0.22,
            "crosswind_watch": 12.0,
            "_stints_used": total_stints,
            "_by_compound": compound_thresholds,
        }
        print(f"  {track_id}: wear_warning={wear_warning} wear_critical={wear_critical} ({total_stints} stints)")

    return circuits


def main() -> None:
    print("Fitting per-circuit thresholds from FastF1...")
    results = fit()
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} circuit entries → {_OUTPUT}")


if __name__ == "__main__":
    main()
