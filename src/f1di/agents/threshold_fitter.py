"""Per-circuit threshold fitter using FastF1 historical stint data.

Derives wear_critical / wear_warning thresholds from the empirical distribution
of TyreLife at pit-stop for each circuit × compound combination. Blends
circuit-specific evidence with global priors via a simple Bayesian shrinkage
so circuits with few observations stay close to the defaults.

Usage (CLI / API):
    from f1di.agents.threshold_fitter import fit_and_save
    report = fit_and_save(years=[2022, 2023, 2024])
"""
from __future__ import annotations

import logging
import warnings
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

from f1di.agents.thresholds import CircuitThresholds, _PATH as _DEFAULT_PATH, save

logger = logging.getLogger("f1di.agents.threshold_fitter")

_CACHE_DIR = str(Path(__file__).parents[3] / "data" / "fastf1_cache")

# Global prior — used as a Bayesian anchor when circuit evidence is sparse.
_PRIOR = CircuitThresholds()

# Minimum number of stint observations before trusting circuit-specific estimates.
_MIN_STINTS = 8

# Compound life priors (P50 expected stint laps) — used when FastF1 data is
# unavailable to keep the fitting function deterministic in offline mode.
_COMPOUND_LIFE_PRIOR = {
    "SOFT": 18,
    "MEDIUM": 26,
    "HARD": 35,
    "INTERMEDIATE": 25,
    "WET": 20,
}

# Circuits known to cause heavier-than-average tire degradation.
# Used to apply a conservative correction when data is sparse.
_HIGH_WEAR_CIRCUITS = {
    "bahrain", "barcelona", "silverstone", "spielberg", "budapest",
    "austin", "interlagos", "mexico_city", "lusail",
}
_LOW_WEAR_CIRCUITS = {
    "monaco", "baku", "jeddah", "las_vegas",
}


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[idx]


def _shrink(circuit_val: float, prior_val: float, n: int, min_n: int = _MIN_STINTS) -> float:
    """Bayesian shrinkage: blend circuit estimate toward prior when n is small."""
    weight = min(1.0, n / min_n)
    return round(weight * circuit_val + (1 - weight) * prior_val, 3)


def _wear_thresholds_from_stints(
    stints: list[float],
    circuit: str,
) -> tuple[float, float]:
    """Return (wear_critical, wear_warning) from a list of TyreLife values at pit.

    The reference life (P90) represents the longest viable stint.
    wear_critical ≈ P55 / P90 — tire is past its efficient working range.
    wear_warning  ≈ P35 / P90 — prepare for pit but not urgent.
    """
    if not stints:
        base_warn = _PRIOR.wear_warning
        base_crit = _PRIOR.wear_critical
        if circuit in _HIGH_WEAR_CIRCUITS:
            base_warn -= 0.05
            base_crit -= 0.05
        elif circuit in _LOW_WEAR_CIRCUITS:
            base_warn += 0.05
            base_crit += 0.05
        return round(base_crit, 3), round(base_warn, 3)

    p35 = _percentile(stints, 35)
    p55 = _percentile(stints, 55)
    p90 = _percentile(stints, 90)

    if p90 < 3:
        return _PRIOR.wear_critical, _PRIOR.wear_warning

    raw_warn = p35 / p90
    raw_crit = p55 / p90

    # Clamp to sensible physical bounds
    warn = max(0.40, min(0.82, raw_warn))
    crit = max(max(warn + 0.06, 0.52), min(0.90, raw_crit))
    return round(crit, 3), round(warn, 3)


def fit_from_fastf1(
    years: list[int] | None = None,
    n_per_year: int = 8,
) -> dict[str, CircuitThresholds]:
    """Download FastF1 stint data and fit per-circuit thresholds.

    Returns a dict keyed by track_id. Falls back to default thresholds for
    any circuit where we couldn't gather enough data.
    """
    try:
        import fastf1
        from f1di.knowledge.track_ids import canonical as canonical_track_id
    except ImportError:
        logger.warning("fastf1 not installed — returning default thresholds")
        return {}

    if years is None:
        current = date.today().year
        years = [current - 1, current - 2, current - 3]

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fastf1.Cache.enable_cache(_CACHE_DIR)

    # Collect stint end TyreLife by (track_id, compound)
    stint_pool: dict[str, list[float]] = defaultdict(list)
    rain_laps: dict[str, list[float]] = defaultdict(list)

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            today = str(date.today())
            past = schedule[schedule["EventDate"].astype(str) <= today]
            if len(past) <= n_per_year:
                events = past
            else:
                # Sample evenly across the season to avoid late-season bias
                step = len(past) / n_per_year
                idx = [int(i * step) for i in range(n_per_year)]
                events = past.iloc[idx]
        except Exception as exc:
            logger.warning("fastf1_schedule_failed year=%s: %s", year, exc)
            continue

        for _, row in events.iterrows():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    session = fastf1.get_session(year, int(row["RoundNumber"]), "R")
                    session.load(
                        telemetry=False,
                        weather=True,
                        messages=False,
                        laps=True,
                    )

                location = session.event.get("Location", row["EventName"])
                track_id = canonical_track_id(location)

                laps = session.laps
                valid = laps[laps["LapTime"].notna()].copy()

                # ── Stint TyreLife at pit ────────────────────────────────
                if "TyreLife" in valid.columns and "Stint" in valid.columns:
                    for (driver, stint_n), grp in valid.groupby(["Driver", "Stint"]):
                        compound = grp["Compound"].iloc[0] if "Compound" in grp.columns else "UNKNOWN"
                        if compound not in _COMPOUND_LIFE_PRIOR:
                            continue
                        life_at_exit = float(grp["TyreLife"].max())
                        if life_at_exit >= 2:
                            key = f"{track_id}:{compound}"
                            stint_pool[key].append(life_at_exit)

                # ── Rain detection ───────────────────────────────────────
                wd = session.weather_data
                if wd is not None and "Rainfall" in wd.columns and "TrackTemp" in wd.columns:
                    rain_rows = wd[wd["Rainfall"].astype(bool)]
                    if not rain_rows.empty:
                        # mark this circuit as rain-sensitive (lower rain_warning threshold)
                        rain_laps[track_id].append(float(rain_rows["TrackTemp"].mean()))

            except Exception as exc:
                logger.debug(
                    "fastf1_session_skipped year=%s round=%s: %s",
                    year, row.get("RoundNumber"), exc,
                )

    # ── Build per-circuit thresholds ────────────────────────────────────
    circuits: set[str] = {k.split(":")[0] for k in stint_pool}
    result: dict[str, CircuitThresholds] = {}

    for circuit in circuits:
        soft_stints = stint_pool.get(f"{circuit}:SOFT", [])
        medium_stints = stint_pool.get(f"{circuit}:MEDIUM", [])
        # Use the compound with most observations as the primary signal
        primary = max(
            [("SOFT", soft_stints), ("MEDIUM", medium_stints)],
            key=lambda x: len(x[1]),
        )[1]

        raw_crit, raw_warn = _wear_thresholds_from_stints(primary, circuit)
        n_obs = len(primary)

        wear_critical = _shrink(raw_crit, _PRIOR.wear_critical, n_obs)
        wear_warning = _shrink(raw_warn, _PRIOR.wear_warning, n_obs)

        # Rain warning: lower threshold for circuits where we've seen rain laps
        rain_warning = _PRIOR.rain_warning
        if rain_laps.get(circuit):
            rain_warning = max(0.20, _PRIOR.rain_warning - 0.05)

        result[circuit] = CircuitThresholds(
            wear_critical=wear_critical,
            wear_warning=wear_warning,
            brake_temp_critical_c=_PRIOR.brake_temp_critical_c,
            fl_degradation_pressure_critical=_shrink(
                max(0.55, wear_critical - 0.05), _PRIOR.fl_degradation_pressure_critical, n_obs
            ),
            fl_degradation_pressure_warning=_shrink(
                max(0.45, wear_warning - 0.08), _PRIOR.fl_degradation_pressure_warning, n_obs
            ),
            rain_warning=round(rain_warning, 3),
            battery_soc_warning=_PRIOR.battery_soc_warning,
            crosswind_watch=_PRIOR.crosswind_watch,
        )

        logger.info(
            "threshold_fitted circuit=%s  wear_crit=%.3f  wear_warn=%.3f  n_stints=%d",
            circuit, wear_critical, wear_warning, n_obs,
        )

    return result


def fit_and_save(
    years: list[int] | None = None,
    n_per_year: int = 8,
    output_path: Path = _DEFAULT_PATH,
    merge: bool = True,
) -> dict:
    """Fit thresholds and write to thresholds.json.

    Args:
        merge: If True, keep existing circuit entries that weren't updated.
               If False, replace the entire file.

    Returns a report dict with fitted/skipped circuit lists and statistics.
    """
    import json

    fitted_thresholds = fit_from_fastf1(years=years, n_per_year=n_per_year)
    fitted_circuits = list(fitted_thresholds.keys())
    skipped_circuits: list[str] = []

    if merge and output_path.exists():
        try:
            existing_data = json.loads(output_path.read_text())
            existing: dict[str, CircuitThresholds] = {}
            for k, v in existing_data.items():
                fields = {f: v[f] for f in CircuitThresholds.__dataclass_fields__ if f in v}
                existing[k] = CircuitThresholds(**fields)
            existing.update(fitted_thresholds)
            final = existing
        except Exception:
            final = fitted_thresholds
    else:
        final = fitted_thresholds

    if final:
        save(final, output_path)
    else:
        skipped_circuits = ["all — fastf1 unavailable or no data"]

    deltas: dict[str, dict] = {}
    if output_path.exists() and fitted_thresholds:
        try:
            old_data = json.loads(output_path.read_text())
            for circuit, new_t in fitted_thresholds.items():
                old = old_data.get(circuit, {})
                deltas[circuit] = {
                    "wear_critical_delta": round(new_t.wear_critical - old.get("wear_critical", _PRIOR.wear_critical), 3),
                    "wear_warning_delta": round(new_t.wear_warning - old.get("wear_warning", _PRIOR.wear_warning), 3),
                }
        except Exception:
            pass

    return {
        "fitted": fitted_circuits,
        "skipped": skipped_circuits,
        "n_fitted": len(fitted_circuits),
        "output_path": str(output_path),
        "deltas": deltas,
    }


def adjust_from_labels(
    output_path: Path = _DEFAULT_PATH,
    min_labeled: int = 5,
    target_precision: float = 0.75,
) -> dict:
    """Adjust per-circuit wear thresholds using outcome-labeled insights from the DB.

    For each circuit where the tire_strategy agent has ≥ min_labeled labeled
    WARNING/CRITICAL predictions:
    - precision < target_precision - 0.10 → raise thresholds (too many false alarms)
    - precision > target_precision + 0.15 → lower thresholds (catching real events reliably)
    - otherwise → leave unchanged

    Bayesian shrinkage toward the prior keeps circuits with sparse data stable.
    Requires per-agent features to be stored in findings_json (repository.py v2+).
    """
    import json as _json

    try:
        from sqlalchemy import select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
    except Exception as exc:
        return {"error": f"persistence not available: {exc}"}

    # (track_id) -> list of (wear_pressure, is_correct)
    circuit_data: dict[str, list[tuple[float, bool]]] = {}

    try:
        with db_session() as session:
            stmt = (
                select(FeedbackRecord, InsightRecord)
                .outerjoin(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
                .where(InsightRecord.risk.in_(["WARNING", "CRITICAL"]))
            )
            for fb, ins in session.execute(stmt).all():
                if ins is None or not ins.track_id:
                    continue
                if fb.correct is not None:
                    is_correct = fb.correct
                elif fb.rating is not None:
                    is_correct = fb.rating >= 4
                else:
                    continue

                try:
                    findings = _json.loads(ins.findings_json or "[]")
                except Exception:
                    continue

                tire_finding = next(
                    (f for f in findings
                     if f.get("agent") == "tire_strategy"
                     and f.get("risk") in ("WARNING", "CRITICAL")),
                    None,
                )
                if tire_finding is None:
                    continue

                # wear_pressure is stored in finding features (repository.py v2+).
                # Fall back to the insight confidence as a coarser proxy.
                wear = tire_finding.get("features", {}).get("wear_pressure", ins.confidence)
                circuit_data.setdefault(ins.track_id, []).append((float(wear), is_correct))
    except Exception as exc:
        logger.warning("adjust_from_labels db query failed: %s", exc)
        return {"error": str(exc)}

    # Load current thresholds from disk.
    current: dict[str, CircuitThresholds] = {}
    if output_path.exists():
        try:
            raw = _json.loads(output_path.read_text())
            for k, v in raw.items():
                fields = {f: v[f] for f in CircuitThresholds.__dataclass_fields__ if f in v}
                current[k] = CircuitThresholds(**fields)
        except Exception:
            pass

    adjusted: list[dict] = []

    for track_id, pairs in circuit_data.items():
        if len(pairs) < min_labeled:
            continue

        correct = sum(1 for _, c in pairs if c)
        precision = correct / len(pairs)
        prior = current.get(track_id, _PRIOR)

        if precision < target_precision - 0.10:
            # Too many false alarms: raise thresholds to be more conservative.
            new_warn = min(0.82, prior.wear_warning + 0.02)
            new_crit = min(0.90, prior.wear_critical + 0.02)
        elif precision > target_precision + 0.15:
            # Very precise: lower thresholds slightly to catch more true positives.
            new_warn = max(0.55, prior.wear_warning - 0.01)
            new_crit = max(0.65, prior.wear_critical - 0.01)
        else:
            continue  # precision within acceptable band, no adjustment

        # Bayesian shrinkage: require ≥20 samples before fully trusting the adjustment.
        weight = min(1.0, len(pairs) / 20)
        adj_warn = round(weight * new_warn + (1 - weight) * prior.wear_warning, 3)
        adj_crit = round(weight * new_crit + (1 - weight) * prior.wear_critical, 3)

        updated = CircuitThresholds(
            wear_warning=adj_warn,
            wear_critical=adj_crit,
            brake_temp_critical_c=prior.brake_temp_critical_c,
            fl_degradation_pressure_critical=round(max(0.55, adj_crit - 0.05), 3),
            fl_degradation_pressure_warning=round(max(0.45, adj_warn - 0.08), 3),
            rain_warning=prior.rain_warning,
            battery_soc_warning=prior.battery_soc_warning,
            crosswind_watch=prior.crosswind_watch,
        )
        current[track_id] = updated

        adjusted.append({
            "track_id": track_id,
            "n_labeled": len(pairs),
            "precision": round(precision, 3),
            "wear_warning": f"{prior.wear_warning} → {adj_warn}",
            "wear_critical": f"{prior.wear_critical} → {adj_crit}",
        })
        logger.info(
            "threshold_adjusted circuit=%s precision=%.2f n=%d warn=%.3f→%.3f crit=%.3f→%.3f",
            track_id, precision, len(pairs),
            prior.wear_warning, adj_warn, prior.wear_critical, adj_crit,
        )

    if adjusted and current:
        save(current, output_path)

    return {"adjusted": adjusted, "n_circuits": len(adjusted)}
