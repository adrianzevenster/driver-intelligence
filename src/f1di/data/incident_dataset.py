"""Labeled incident dataset builder from FastF1 historical race data.

Extracts (features_proxy, label) pairs by identifying high-risk race situations
— forced pit stops, retirements, safety cars, and late-stint degradation cliffs
— and labeling the preceding laps as high-risk. Output is stored as JSONL and
used to augment the synthetic calibration dataset in confidence/fitting.py.

Label semantics (0.0–1.0, matching _ground_truth_label in fitting.py):
  ≥ 0.85  — imminent incident (retirement, lockup, forced pit within 2 laps)
  0.70–0.84 — high-risk window (forced pit in 3–5 laps, safety car ahead)
  0.50–0.69 — elevated risk (late stint, degradation cliff projected)
  0.15–0.35 — normal racing
  ≤ 0.14   — early/mid stint, all nominal
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

logger = logging.getLogger("f1di.data.incident")

_CACHE_DIR = str(Path(__file__).parents[3] / "data" / "fastf1_cache")
_DEFAULT_OUTPUT = Path("data/incidents/labeled_dataset.jsonl")

# Look-ahead window: label laps this far ahead of an incident as high-risk.
_LOOKAHEAD_CRITICAL = 2   # laps: CRITICAL label
_LOOKAHEAD_WARNING = 5    # laps: WARNING label
_LOOKAHEAD_WATCH = 8      # laps: WATCH label

# Expected compound stint life (P50) used to detect forced/early pits.
_EXPECTED_LIFE = {
    "SOFT": 18, "MEDIUM": 26, "HARD": 35, "INTERMEDIATE": 22, "WET": 18,
}


@dataclass
class LabeledWindow:
    track_id: str
    year: int
    event_round: int
    driver: str
    lap: int
    compound: str
    stint_lap: int
    tyre_life_fraction: float    # stint_lap / expected_compound_life
    laptime_delta_s: float       # lap-over-lap Δ lap time (positive = getting slower)
    relative_laptime: float      # lap time relative to session median
    track_temp_c: float
    rain: bool
    incident_type: str           # "none" | "forced_pit" | "retirement" | "safety_car" | "cliff"
    laps_to_incident: int        # 0 = this lap, n = n laps before incident
    label: float                 # ground truth confidence label


def _label_from_proximity(incident_type: str, laps_ahead: int) -> float:
    """Convert (incident_type, laps_ahead) to a ground-truth label."""
    if incident_type == "retirement":
        if laps_ahead <= _LOOKAHEAD_CRITICAL:
            return 0.90
        if laps_ahead <= _LOOKAHEAD_WARNING:
            return 0.75
        return 0.60
    if incident_type == "forced_pit":
        if laps_ahead <= _LOOKAHEAD_CRITICAL:
            return 0.85
        if laps_ahead <= _LOOKAHEAD_WARNING:
            return 0.72
        return 0.58
    if incident_type == "safety_car":
        if laps_ahead <= _LOOKAHEAD_CRITICAL:
            return 0.78
        if laps_ahead <= _LOOKAHEAD_WARNING:
            return 0.62
        return 0.48
    if incident_type == "cliff":
        return min(0.70, 0.42 + (1.0 - laps_ahead / _LOOKAHEAD_WATCH) * 0.28)
    return 0.20  # normal


def _extract_incidents_from_session(
    session,
    track_id: str,
    year: int,
    event_round: int,
) -> list[LabeledWindow]:
    """Process one FastF1 race session and return labeled windows."""
    import pandas as pd

    laps = session.laps.copy()
    if laps.empty:
        return []

    valid = laps[laps["LapTime"].notna()].copy()
    if valid.empty:
        return []

    # ── Weather ──────────────────────────────────────────────────────────
    weather_map: dict[int, tuple[float, bool]] = {}  # lap → (track_temp, is_raining)
    wd = session.weather_data
    if wd is not None and not wd.empty and "TrackTemp" in wd.columns:
        # Align weather to lap numbers via approximate timestamp matching
        for lap_n in valid["LapNumber"].unique():
            lap_rows = valid[valid["LapNumber"] == int(lap_n)]
            if "Time" in lap_rows.columns:
                lap_time = lap_rows["Time"].iloc[0] if not lap_rows.empty else None
                if lap_time is not None:
                    wd_before = wd[wd.index <= lap_time] if hasattr(wd.index, "dtype") else wd
                    if not wd_before.empty:
                        row = wd_before.iloc[-1]
                        temp = float(row.get("TrackTemp", 30.0))
                        rain = bool(row.get("Rainfall", False))
                        weather_map[int(lap_n)] = (temp, rain)

    # ── Identify incidents ───────────────────────────────────────────────
    all_laps = set(valid["LapNumber"].astype(int).unique())
    max_lap = int(valid["LapNumber"].max()) if not valid.empty else 0
    session_median_s = valid["LapTime"].dt.total_seconds().median()

    # Track lap time per driver for cliff detection
    driver_laps: dict[str, list[tuple[int, float, str, int]]] = defaultdict(list)
    for _, row in valid.iterrows():
        drv = str(row["Driver"])
        lap_n = int(row["LapNumber"])
        lt_s = row["LapTime"].total_seconds() if pd.notna(row["LapTime"]) else None
        compound = str(row["Compound"]) if "Compound" in row.index and pd.notna(row.get("Compound")) else "UNKNOWN"
        tyre_life = int(row["TyreLife"]) if "TyreLife" in row.index and pd.notna(row.get("TyreLife")) else 0
        if lt_s:
            driver_laps[drv].append((lap_n, lt_s, compound, tyre_life))

    # Incident log: {driver → [(incident_lap, incident_type), ...]}
    incident_log: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for drv, lap_list in driver_laps.items():
        lap_list_sorted = sorted(lap_list, key=lambda x: x[0])
        lap_nums = [x[0] for x in lap_list_sorted]

        # Retirement: driver's last lap is before max_lap - 3
        if lap_nums and lap_nums[-1] < max_lap - 3:
            incident_log[drv].append((lap_nums[-1], "retirement"))

        # Forced pit: stint ended before P30 of expected compound life
        stints: dict[int, list[tuple[int, float, str, int]]] = defaultdict(list)
        if "Stint" in valid.columns:
            for _, row in valid[valid["Driver"] == drv].iterrows():
                stint_n = int(row.get("Stint", 1))
                lap_n = int(row["LapNumber"])
                lt_s = row["LapTime"].total_seconds() if pd.notna(row["LapTime"]) else None
                compound = str(row.get("Compound", "UNKNOWN"))
                tyre_life = int(row.get("TyreLife", 0)) if pd.notna(row.get("TyreLife", 0)) else 0
                if lt_s:
                    stints[stint_n].append((lap_n, lt_s, compound, tyre_life))
        else:
            # Fallback: detect pit from TyreLife reset
            prev_life = 0
            stint_n = 0
            for lap_n, lt_s, compound, tyre_life in lap_list_sorted:
                if tyre_life < prev_life and prev_life > 0:
                    stint_n += 1
                stints[stint_n].append((lap_n, lt_s, compound, tyre_life))
                prev_life = tyre_life

        for stint_n, stint_laps in stints.items():
            if not stint_laps:
                continue
            compound = stint_laps[0][2]
            tyre_life_at_exit = max(x[3] for x in stint_laps)
            expected = _EXPECTED_LIFE.get(compound, 24)
            # "Forced pit" = pitted before 30% of expected life (rushed out)
            if tyre_life_at_exit < expected * 0.30 and len(stint_laps) >= 3:
                pit_lap = max(x[0] for x in stint_laps)
                incident_log[drv].append((pit_lap, "forced_pit"))

        # Degradation cliff: 3+ consecutive laps getting slower by >0.8s each
        for i in range(2, len(lap_list_sorted)):
            a, b, c = lap_list_sorted[i - 2], lap_list_sorted[i - 1], lap_list_sorted[i]
            if b[1] - a[1] > 0.8 and c[1] - b[1] > 0.8 and c[3] > 10:
                incident_log[drv].append((c[0], "cliff"))

    # ── Safety car: detect from large field-wide lap time spike ──────────
    lap_field_median: dict[int, float] = {}
    for lap_n in all_laps:
        lap_group = valid[valid["LapNumber"] == lap_n]["LapTime"]
        if len(lap_group) >= 5:
            lap_field_median[lap_n] = lap_group.dt.total_seconds().median()

    sorted_medians = sorted(lap_field_median.items())
    if len(sorted_medians) > 5:
        baseline = sorted([m for _, m in sorted_medians[:10]])[len(sorted_medians[:10]) // 2]
        for lap_n, median in sorted_medians:
            if median > baseline * 1.25 and lap_n > 3:
                for drv in driver_laps:
                    incident_log[drv].append((lap_n, "safety_car"))

    # ── Build labeled windows ────────────────────────────────────────────
    windows: list[LabeledWindow] = []
    for drv, lap_list in driver_laps.items():
        lap_list_sorted = sorted(lap_list, key=lambda x: x[0])

        # Build incident timeline for this driver
        driver_incidents: list[tuple[int, str]] = sorted(
            set(incident_log.get(drv, [])), key=lambda x: x[0]
        )

        prev_lt = None
        for lap_n, lt_s, compound, tyre_life in lap_list_sorted:
            expected_life = _EXPECTED_LIFE.get(compound, 24)
            tyre_life_frac = tyre_life / expected_life if expected_life > 0 else 0.0
            laptime_delta = (lt_s - prev_lt) if prev_lt else 0.0
            track_temp, rain = weather_map.get(lap_n, (30.0, False))
            rel_laptime = (lt_s - session_median_s) / session_median_s if session_median_s > 0 else 0.0
            prev_lt = lt_s

            # Find the nearest upcoming incident for this lap
            nearest_type = "none"
            nearest_dist = 999
            for inc_lap, inc_type in driver_incidents:
                dist = inc_lap - lap_n
                if 0 <= dist < nearest_dist and dist <= _LOOKAHEAD_WATCH:
                    nearest_dist = dist
                    nearest_type = inc_type

            label = _label_from_proximity(nearest_type, nearest_dist)

            # Normal laps in early/mid stint get a conservative low label
            if nearest_type == "none":
                if tyre_life_frac < 0.40:
                    label = 0.12
                elif tyre_life_frac < 0.65:
                    label = 0.22
                else:
                    label = 0.35 if not rain else 0.45

            windows.append(LabeledWindow(
                track_id=track_id,
                year=year,
                event_round=event_round,
                driver=drv,
                lap=lap_n,
                compound=compound,
                stint_lap=tyre_life,
                tyre_life_fraction=round(tyre_life_frac, 4),
                laptime_delta_s=round(laptime_delta, 4),
                relative_laptime=round(rel_laptime, 4),
                track_temp_c=track_temp,
                rain=rain,
                incident_type=nearest_type,
                laps_to_incident=nearest_dist if nearest_type != "none" else -1,
                label=round(label, 3),
            ))

    return windows


def build_dataset(
    years: list[int] | None = None,
    n_per_year: int = 6,
    output_path: Path = _DEFAULT_OUTPUT,
) -> dict:
    """Fetch FastF1 data and write a labeled incident dataset.

    Returns a summary dict with counts by incident_type and label distribution.
    """
    try:
        import fastf1
        from f1di.knowledge.track_ids import canonical as canonical_track_id
    except ImportError:
        logger.warning("fastf1 not installed — cannot build incident dataset")
        return {"error": "fastf1_not_installed", "n_windows": 0}

    if years is None:
        current = date.today().year
        years = [current - 1, current - 2, current - 3]

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fastf1.Cache.enable_cache(_CACHE_DIR)

    all_windows: list[LabeledWindow] = []

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            today = str(date.today())
            past = schedule[schedule["EventDate"].astype(str) <= today]
            events = past.tail(n_per_year)
        except Exception as exc:
            logger.warning("schedule_failed year=%s: %s", year, exc)
            continue

        for _, row in events.iterrows():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    session = fastf1.get_session(year, int(row["RoundNumber"]), "R")
                    session.load(telemetry=False, weather=True, messages=False, laps=True)

                location = session.event.get("Location", row["EventName"])
                track_id = canonical_track_id(location)
                event_round = int(row["RoundNumber"])

                windows = _extract_incidents_from_session(session, track_id, year, event_round)
                all_windows.extend(windows)
                logger.info(
                    "incident_dataset_session year=%s round=%s track=%s windows=%d",
                    year, event_round, track_id, len(windows),
                )
            except Exception as exc:
                logger.warning(
                    "session_failed year=%s round=%s: %s",
                    year, row.get("RoundNumber"), exc,
                )

    if not all_windows:
        return {"n_windows": 0, "by_incident": {}, "label_stats": {}}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for w in all_windows:
            fh.write(json.dumps(asdict(w)) + "\n")

    by_type: dict[str, int] = defaultdict(int)
    label_sum = 0.0
    high_risk_count = 0
    for w in all_windows:
        by_type[w.incident_type] += 1
        label_sum += w.label
        if w.label >= 0.70:
            high_risk_count += 1

    logger.info(
        "incident_dataset_built  n=%d  high_risk=%d  path=%s",
        len(all_windows), high_risk_count, output_path,
    )
    return {
        "n_windows": len(all_windows),
        "by_incident": dict(by_type),
        "label_stats": {
            "mean": round(label_sum / len(all_windows), 3),
            "high_risk_pct": round(high_risk_count / len(all_windows), 3),
        },
        "output_path": str(output_path),
    }


def load_dataset(
    path: Path = _DEFAULT_OUTPUT,
) -> tuple[list[float], list[float]]:
    """Load saved dataset as (X_proxy, y) for calibration augmentation.

    X_proxy uses tyre_life_fraction + relative_laptime + rain as a simple
    composite score that correlates with our raw calibration score space.
    """
    if not path.exists():
        return [], []

    X: list[float] = []
    y: list[float] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            w = json.loads(line)
            # Simple composite proxy score in [0, 1] that maps the FastF1
            # lap-level features into the same scale as our raw confidence score.
            wear_signal = min(1.0, w.get("tyre_life_fraction", 0.0))
            deg_signal = min(1.0, max(0.0, w.get("laptime_delta_s", 0.0) / 3.0))
            rain_bonus = 0.10 if w.get("rain", False) else 0.0
            proxy = min(1.0, 0.55 * wear_signal + 0.35 * deg_signal + rain_bonus)
            X.append(proxy)
            y.append(w["label"])

    return X, y
