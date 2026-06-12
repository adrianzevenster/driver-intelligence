"""Post-race outcome labeler — closes the data flywheel.

After each race, this module:
  1. Fetches FastF1 race data to identify actual incidents
     (retirements, safety cars, unplanned pit stops).
  2. Queries stored InsightRecords for the matching session.
  3. Labels each WARNING/CRITICAL insight as correct or incorrect
     based on whether a matching incident occurred within a look-ahead
     window, and writes FeedbackRecord rows that feed the calibration
     retraining loop.

This is the missing link between "the model predicted risk" and
"something actually happened" — without it, calibration can only
learn from users rating chat responses, not from race outcomes.

Usage:
    from f1di.data.outcome_labeler import label_race
    report = label_race(year=2024, round_num=5)
"""
from __future__ import annotations

import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger("f1di.data.outcome_labeler")

_CACHE_DIR = "/tmp/f1di_fastf1_cache"

# How many laps ahead an incident must occur for a WARNING/CRITICAL prediction
# to be considered "correct".
_CORRECT_WINDOW_LAPS = 5

# How many laps must pass WITHOUT an incident before a WARNING/CRITICAL prediction
# is labeled "incorrect" (false alarm).
_FALSE_ALARM_WINDOW_LAPS = 8


@dataclass
class Incident:
    driver: str
    lap: int
    incident_type: str   # "retirement" | "safety_car" | "forced_pit" | "lockup_proxy"
    severity: float      # 0.0–1.0


@dataclass
class OutcomeReport:
    year: int
    round_num: int
    track_id: str
    n_insights_examined: int
    n_labeled_correct: int
    n_labeled_incorrect: int
    n_no_match: int
    incidents_found: list[dict]


def _extract_incidents(session) -> list[Incident]:
    """Extract incident list from a loaded FastF1 race session."""
    import pandas as pd

    laps = session.laps.copy()
    if laps.empty:
        return []

    valid = laps[laps["LapTime"].notna()].copy()
    max_lap = int(valid["LapNumber"].max()) if not valid.empty else 0
    incidents: list[Incident] = []

    # ── Retirements ──────────────────────────────────────────────────────
    for drv, grp in valid.groupby("Driver"):
        last_lap = int(grp["LapNumber"].max())
        if last_lap < max_lap - 3:
            incidents.append(Incident(
                driver=str(drv),
                lap=last_lap,
                incident_type="retirement",
                severity=0.95,
            ))

    # ── Safety car: field-wide lap time spike ────────────────────────────
    lap_medians: dict[int, float] = {}
    for lap_n in valid["LapNumber"].unique():
        grp = valid[valid["LapNumber"] == int(lap_n)]["LapTime"]
        if len(grp) >= 5:
            lap_medians[int(lap_n)] = grp.dt.total_seconds().median()

    sorted_med = sorted(lap_medians.items())
    if len(sorted_med) > 5:
        baseline_s = sorted([m for _, m in sorted_med[:10]])[len(sorted_med[:10]) // 2]
        sc_laps: list[int] = []
        for lap_n, med in sorted_med:
            if med > baseline_s * 1.22 and lap_n > 3:
                sc_laps.append(lap_n)
        for sc_lap in sc_laps:
            for drv in valid["Driver"].unique():
                incidents.append(Incident(
                    driver=str(drv),
                    lap=sc_lap,
                    incident_type="safety_car",
                    severity=0.70,
                ))

    # ── Forced pits: stint ended before 30% of expected compound life ────
    _EXPECTED = {"SOFT": 18, "MEDIUM": 26, "HARD": 35, "INTERMEDIATE": 22, "WET": 18}
    if "Stint" in valid.columns and "TyreLife" in valid.columns:
        for (drv, stint_n), grp in valid.groupby(["Driver", "Stint"]):
            compound = str(grp["Compound"].iloc[0]) if "Compound" in grp.columns else "UNKNOWN"
            life = float(grp["TyreLife"].max()) if pd.notna(grp["TyreLife"].max()) else 0
            expected = _EXPECTED.get(compound, 24)
            if life < expected * 0.30 and len(grp) >= 3:
                pit_lap = int(grp["LapNumber"].max())
                incidents.append(Incident(
                    driver=str(drv),
                    lap=pit_lap,
                    incident_type="forced_pit",
                    severity=0.80,
                ))

    # ── Lockup proxy: 3+ consecutive lap-time spikes for one driver ──────
    for drv, grp in valid.groupby("Driver"):
        grp_sorted = grp.sort_values("LapNumber")
        lt = grp_sorted["LapTime"].dt.total_seconds().tolist()
        laps_num = grp_sorted["LapNumber"].tolist()
        for i in range(2, len(lt)):
            if lt[i] - lt[i - 1] > 1.5 and lt[i - 1] - lt[i - 2] > 1.5:
                incidents.append(Incident(
                    driver=str(drv),
                    lap=int(laps_num[i]),
                    incident_type="lockup_proxy",
                    severity=0.65,
                ))

    return incidents


def _session_id_for_race(year: int, round_num: int, track_id: str) -> list[str]:
    """Return candidate session_id prefixes that might match stored insights."""
    return [
        f"live_{year}_{round_num}",
        f"replay_{year}_{round_num}",
        f"{track_id}_{year}",
        f"f1_{year}_{round_num}",
    ]


def label_race(
    year: int,
    round_num: int,
    *,
    dry_run: bool = False,
) -> OutcomeReport:
    """Download FastF1 data for one race and label stored insights.

    Args:
        dry_run: If True, compute labels but do not write FeedbackRecord rows.

    Returns an OutcomeReport with labeling statistics.
    """
    try:
        import fastf1
        from f1di.knowledge.track_ids import canonical as canonical_track_id
    except ImportError:
        logger.warning("fastf1 not installed — outcome labeling unavailable")
        return OutcomeReport(
            year=year, round_num=round_num, track_id="unknown",
            n_insights_examined=0, n_labeled_correct=0,
            n_labeled_incorrect=0, n_no_match=0, incidents_found=[],
        )

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fastf1.Cache.enable_cache(_CACHE_DIR)
        try:
            session = fastf1.get_session(year, round_num, "R")
            session.load(telemetry=False, weather=True, messages=False, laps=True)
        except Exception as exc:
            logger.error("fastf1_load_failed year=%s round=%s: %s", year, round_num, exc)
            return OutcomeReport(
                year=year, round_num=round_num, track_id="unknown",
                n_insights_examined=0, n_labeled_correct=0,
                n_labeled_incorrect=0, n_no_match=0, incidents_found=[],
            )

    location = session.event.get("Location", "")
    track_id = canonical_track_id(location)
    incidents = _extract_incidents(session)

    logger.info(
        "outcome_labeler found %d incidents  year=%s round=%s track=%s",
        len(incidents), year, round_num, track_id,
    )

    # Build incident index: driver → [(lap, type, severity)]
    incident_index: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for inc in incidents:
        incident_index[inc.driver].append((inc.lap, inc.incident_type, inc.severity))
        # Safety car affects all drivers — also index globally
        if inc.incident_type == "safety_car":
            incident_index["*"].append((inc.lap, inc.incident_type, inc.severity))

    # Load stored insights for this race from the database
    try:
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
        from sqlalchemy import select, or_
    except Exception as exc:
        logger.warning("db_unavailable: %s", exc)
        return OutcomeReport(
            year=year, round_num=round_num, track_id=track_id,
            n_insights_examined=0, n_labeled_correct=0,
            n_labeled_incorrect=0, n_no_match=0,
            incidents_found=[{"driver": i.driver, "lap": i.lap, "type": i.incident_type} for i in incidents],
        )

    candidate_prefixes = _session_id_for_race(year, round_num, track_id)
    n_correct = 0
    n_incorrect = 0
    n_no_match = 0
    n_examined = 0

    with db_session() as db:
        for prefix in candidate_prefixes:
            stmt = select(InsightRecord).where(
                InsightRecord.session_id.like(f"{prefix}%"),
                InsightRecord.risk.in_(["WARNING", "CRITICAL"]),
            )
            insights = db.execute(stmt).scalars().all()

            for ins in insights:
                n_examined += 1
                drv = ins.driver_id
                lap = ins.lap or 0

                driver_incidents = incident_index.get(drv, []) + incident_index.get("*", [])
                matching = [
                    (inc_lap, inc_type, sev)
                    for inc_lap, inc_type, sev in driver_incidents
                    if 0 <= inc_lap - lap <= _CORRECT_WINDOW_LAPS
                ]

                if matching:
                    label = True
                    severity = max(m[2] for m in matching)
                    rating = 5 if severity >= 0.85 else 4
                    n_correct += 1
                elif all(
                    inc_lap > lap + _FALSE_ALARM_WINDOW_LAPS
                    for inc_lap, _, _ in driver_incidents
                    if inc_lap >= lap
                ):
                    label = False
                    rating = 2
                    n_incorrect += 1
                else:
                    n_no_match += 1
                    continue

                if dry_run:
                    continue

                # Check we haven't already labeled this insight from outcome data
                existing = db.execute(
                    select(FeedbackRecord).where(
                        FeedbackRecord.insight_id == ins.insight_id,
                        FeedbackRecord.submitted_by == "outcome_labeler",
                    )
                ).scalar_one_or_none()
                if existing:
                    continue

                fb = FeedbackRecord(
                    insight_id=ins.insight_id,
                    rating=rating,
                    correct=label,
                    comment=f"outcome_label year={year} round={round_num} track={track_id}",
                    submitted_by="outcome_labeler",
                )
                db.add(fb)

        if not dry_run:
            try:
                db.commit()
            except Exception as exc:
                logger.warning("outcome_label_commit_failed: %s", exc)
                db.rollback()

    logger.info(
        "outcome_labeler complete  examined=%d correct=%d incorrect=%d no_match=%d",
        n_examined, n_correct, n_incorrect, n_no_match,
    )
    return OutcomeReport(
        year=year,
        round_num=round_num,
        track_id=track_id,
        n_insights_examined=n_examined,
        n_labeled_correct=n_correct,
        n_labeled_incorrect=n_incorrect,
        n_no_match=n_no_match,
        incidents_found=[
            {"driver": i.driver, "lap": i.lap, "type": i.incident_type, "severity": i.severity}
            for i in incidents
        ],
    )
