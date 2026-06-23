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
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass

from pathlib import Path

logger = logging.getLogger("f1di.data.outcome_labeler")

_CACHE_DIR = str(Path(__file__).parents[3] / "data" / "fastf1_cache")

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


def _session_time_to_lap(laps_df, t) -> int:
    """Map a session timedelta to the race lap number it falls within."""
    by_driver = laps_df.groupby("Driver")["LapNumber"].count()
    if by_driver.empty:
        return -1
    best = by_driver.idxmax()
    drv = laps_df[laps_df["Driver"] == best].sort_values("LapNumber")
    drv = drv[drv["LapTime"].notna()]
    for _, row in drv.iterrows():
        if t <= row["Time"]:
            return int(row["LapNumber"])
    nums = drv["LapNumber"].tolist()
    return int(nums[-1]) if nums else -1


def _normalize_rcm_time(t, session):
    """Convert RCM Time to a Timedelta (session-elapsed). Some seasons return Timestamps."""
    import pandas as pd
    if t is None:
        return None
    if isinstance(t, pd.Timedelta):
        return t
    if isinstance(t, pd.Timestamp):
        try:
            t0 = session.t0_date
            if not isinstance(t0, pd.Timestamp):
                t0 = pd.Timestamp(t0)
            delta = t - t0
            return delta if delta.total_seconds() >= 0 else None
        except Exception:
            return None
    return None


def _extract_sc_laps_from_race_control(session, valid_laps) -> list[int]:
    """Return lap numbers of confirmed SC/VSC deployments from race control messages."""
    try:
        rcm = getattr(session, "race_control_messages", None)
    except Exception as exc:
        logger.info("race_control_messages_unavailable: %s", exc)
        return []
    if rcm is None or not hasattr(rcm, "empty") or rcm.empty:
        return []
    sc_laps: list[int] = []
    for _, msg in rcm.iterrows():
        text = str(msg.get("Message", "")).upper()
        status = str(msg.get("Status", "")).upper()
        category = str(msg.get("Category", "")).upper()
        deployed = "DEPLOYED" in text or "DEPLOYED" in status
        is_sc = deployed and (
            "SAFETY CAR" in text
            or "SAFETYCAR" in category
            or "VIRTUAL SAFETY CAR" in text
        )
        if not is_sc:
            continue
        t = _normalize_rcm_time(msg.get("Time"), session)
        if t is None:
            continue
        lap = _session_time_to_lap(valid_laps, t)
        if lap > 0:
            sc_laps.append(lap)
    return sorted(set(sc_laps))


def _extract_sc_laps_from_spike(valid) -> list[int]:
    """Fallback: infer SC laps from a field-wide lap-time spike (>22% above baseline)."""
    lap_medians: dict[int, float] = {}
    for lap_n in valid["LapNumber"].unique():
        grp = valid[valid["LapNumber"] == int(lap_n)]["LapTime"]
        if len(grp) >= 5:
            lap_medians[int(lap_n)] = grp.dt.total_seconds().median()
    sorted_med = sorted(lap_medians.items())
    if len(sorted_med) <= 5:
        return []
    baseline_s = sorted([m for _, m in sorted_med[:10]])[len(sorted_med[:10]) // 2]
    return [lap_n for lap_n, med in sorted_med if med > baseline_s * 1.22 and lap_n > 3]


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

    # ── Safety car: race control messages (authoritative) with lap-spike fallback ──
    sc_laps: list[int] = _extract_sc_laps_from_race_control(session, valid)
    if not sc_laps:
        sc_laps = _extract_sc_laps_from_spike(valid)
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

    # ── Wet-tyre switch: any driver moves to INTERMEDIATE or WET compound ──
    # This confirms weather-agent WARNINGs: if anyone boxed for rain tyres,
    # the rain was real. Indexed globally ("*") so all drivers' weather
    # insights see the confirmation.
    if "Compound" in valid.columns and "Stint" in valid.columns:
        wet_compounds = {"INTERMEDIATE", "WET"}
        seen_wet_laps: set[int] = set()
        for (drv, stint_n), grp in valid.groupby(["Driver", "Stint"]):
            compound = str(grp["Compound"].iloc[0]).upper() if not grp.empty else ""
            if compound in wet_compounds:
                switch_lap = int(grp["LapNumber"].min())
                if switch_lap not in seen_wet_laps:
                    seen_wet_laps.add(switch_lap)
                    incidents.append(Incident(
                        driver="*",
                        lap=switch_lap,
                        incident_type="wet_tyre_switch",
                        severity=0.85,
                    ))

    return incidents


def _session_id_for_race(year: int, round_num: int, track_id: str) -> list[str]:
    """Return candidate session_id prefixes that might match stored insights."""
    return [
        f"live_{year}_{round_num}",
        f"replay_{year}_{round_num}",
        f"{track_id}_{year}",
        f"f1_{year}_{round_num}",
        # build_window() in knowledge/fastf1_session.py — the only path that
        # replays real FastF1 telemetry — stamps insights with this prefix.
        # Without it, no real FastF1-replayed insight could ever be matched.
        f"fastf1_{year}_{round_num}",
    ]


def label_quiet_stints(year: int, round_num: int) -> int:
    """Label SUPPRESS/LOW insights that had no incident as correct (negative class).

    For each insight with risk LOW or policy SUPPRESS that has no existing feedback
    and was generated at least 2 hours before now, writes a
    FeedbackRecord(correct=True, rating=4, submitted_by='null_outcome').
    This prevents the flywheel from being biased purely toward incidents.

    Returns the number of new FeedbackRecord rows written.
    """
    import datetime as _dt
    try:
        from sqlalchemy import select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
    except Exception as exc:
        logger.warning("label_quiet_stints: db unavailable: %s", exc)
        return 0

    candidate_prefixes = [
        f"live_{year}_{round_num}",
        f"replay_{year}_{round_num}",
        f"f1_{year}_{round_num}",
        f"fastf1_{year}_{round_num}",
    ]

    cutoff = _dt.datetime.utcnow() - _dt.timedelta(hours=2)
    n_written = 0

    try:
        with db_session() as db:
            for prefix in candidate_prefixes:
                stmt = (
                    select(InsightRecord)
                    .where(
                        InsightRecord.session_id.like(f"{prefix}%"),
                        InsightRecord.created_at <= cutoff,
                    )
                    .where(
                        (InsightRecord.risk == "LOW")
                        | (InsightRecord.policy == "SUPPRESS")
                        | (InsightRecord.risk == "INFO")
                    )
                )
                insights = db.execute(stmt).scalars().all()

                for ins in insights:
                    existing = db.execute(
                        select(FeedbackRecord).where(
                            FeedbackRecord.insight_id == ins.insight_id,
                        )
                    ).scalar_one_or_none()
                    if existing:
                        continue

                    fb = FeedbackRecord(
                        insight_id=ins.insight_id,
                        rating=4,
                        correct=True,
                        comment=f"null_outcome year={year} round={round_num}",
                        submitted_by="null_outcome",
                    )
                    db.add(fb)
                    n_written += 1

            if n_written > 0:
                try:
                    db.commit()
                except Exception as exc:
                    logger.warning("label_quiet_stints commit failed: %s", exc)
                    db.rollback()
                    n_written = 0
    except Exception as exc:
        logger.error("label_quiet_stints db error year=%s round=%s: %s", year, round_num, exc)
        return 0

    if n_written > 0:
        logger.info("null_outcome_labels year=%d round=%d n=%d", year, round_num, n_written)
    return n_written


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

    try:
        return _label_race_inner(fastf1, canonical_track_id, year, round_num, dry_run=dry_run)
    except Exception:
        logger.error(
            "outcome_labeler_unhandled year=%s round=%s:\n%s",
            year, round_num, traceback.format_exc(),
        )
        return OutcomeReport(
            year=year, round_num=round_num, track_id="unknown",
            n_insights_examined=0, n_labeled_correct=0,
            n_labeled_incorrect=0, n_no_match=0, incidents_found=[],
        )


def _openf1_get(endpoint: str, **params) -> list[dict]:
    """Fetch from OpenF1 API with simple retry on rate-limit."""
    import time
    import urllib.request
    import urllib.parse
    qs = urllib.parse.urlencode(params)
    url = f"https://api.openf1.org/v1/{endpoint}?{qs}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                import json
                return json.loads(r.read())
        except Exception as exc:
            if "429" in str(exc) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise
    return []


def _fetch_openf1_incidents(year: int, round_num: int) -> tuple[str, list[Incident], int]:
    """Return (location, incidents, session_key) using the OpenF1 API (works from server IPs)."""
    sessions = _openf1_get("sessions", year=year, session_name="Race")
    sessions = [s for s in sessions if s.get("date_start")]
    sessions.sort(key=lambda s: s["date_start"])
    if len(sessions) < round_num:
        raise ValueError(f"OpenF1 has only {len(sessions)} races for {year}, want round {round_num}")
    session = sessions[round_num - 1]
    skey = session["session_key"]
    location = session.get("location") or session.get("country_name") or ""

    rc = _openf1_get("race_control", session_key=skey)
    stints = _openf1_get("stints", session_key=skey)

    # Max race lap from stints
    max_lap = max((s.get("lap_end") or 0 for s in stints), default=0)
    # All driver numbers seen in stints
    all_drivers = {str(s["driver_number"]) for s in stints}

    incidents: list[Incident] = []

    # ── Retirements ──────────────────────────────────────────────────────────
    drv_max: dict[str, int] = defaultdict(int)
    for s in stints:
        drv = str(s["driver_number"])
        drv_max[drv] = max(drv_max[drv], s.get("lap_end") or 0)
    for drv, last in drv_max.items():
        if max_lap > 0 and last < max_lap - 3:
            incidents.append(Incident(driver=drv, lap=max(last, 1),
                                      incident_type="retirement", severity=0.95))

    # ── Safety car / Red flag from race control ──────────────────────────────
    _SC_DEPLOY = ("SAFETY CAR DEPLOYED", "VIRTUAL SAFETY CAR DEPLOYED", "VSC DEPLOYED")
    for msg in rc:
        text = (msg.get("message") or "").upper()
        flag = (msg.get("flag") or "").upper()
        lap = msg.get("lap_number") or 0
        if not lap:
            continue
        if flag == "RED" or "RED FLAG" in text:
            for drv in all_drivers:
                incidents.append(Incident(driver=drv, lap=lap,
                                          incident_type="safety_car", severity=0.85))
        elif any(kw in text for kw in _SC_DEPLOY):
            for drv in all_drivers:
                incidents.append(Incident(driver=drv, lap=lap,
                                          incident_type="safety_car", severity=0.70))

    # ── Forced pits: stint ended before 30% of expected compound life ────────
    _EXPECTED = {"SOFT": 18, "MEDIUM": 26, "HARD": 35, "INTERMEDIATE": 22, "WET": 18}
    drv_stints: dict[str, list] = defaultdict(list)
    for s in stints:
        drv_stints[str(s["driver_number"])].append(s)
    for drv, ds in drv_stints.items():
        ds.sort(key=lambda s: s.get("stint_number", 0))
        for s in ds[:-1]:  # exclude last stint
            compound = (s.get("compound") or "UNKNOWN").upper()
            start = s.get("lap_start") or 0
            end = s.get("lap_end") or 0
            life = (end - start + 1) if end >= start and end > 0 else 0
            expected = _EXPECTED.get(compound, 24)
            if 0 < life < expected * 0.30 and life >= 3:
                incidents.append(Incident(driver=drv, lap=end,
                                          incident_type="forced_pit", severity=0.80))

    # ── Wet-tyre switch: any driver moves to INTERMEDIATE or WET ─────────────
    # Confirms weather-agent WARNINGs globally — indexed as "*" so all drivers
    # share the signal.
    _WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
    seen_wet_laps: set[int] = set()
    for s in stints:
        compound = (s.get("compound") or "").upper()
        if compound in _WET_COMPOUNDS:
            start_lap = s.get("lap_start") or 0
            if start_lap > 0 and start_lap not in seen_wet_laps:
                seen_wet_laps.add(start_lap)
                incidents.append(Incident(driver="*", lap=start_lap,
                                          incident_type="wet_tyre_switch", severity=0.85))

    return location, incidents, skey


def _label_race_inner(fastf1, canonical_track_id, year: int, round_num: int, *, dry_run: bool) -> OutcomeReport:
    # ── Primary: OpenF1 (accessible from server IPs) ─────────────────────────
    openf1_session_key: int | None = None
    try:
        location, incidents, openf1_session_key = _fetch_openf1_incidents(year, round_num)
        track_id = canonical_track_id(location)
        logger.info("outcome_labeler openf1 found %d incidents year=%s round=%s track=%s skey=%s",
                    len(incidents), year, round_num, track_id, openf1_session_key)
    except Exception as exc:
        logger.warning("openf1_incident_fetch_failed year=%s round=%s: %s — trying FastF1 cache",
                       year, round_num, exc)
        # ── Fallback: FastF1 (works locally / from pre-populated cache) ──────
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fastf1.Cache.enable_cache(_CACHE_DIR)
            try:
                session = fastf1.get_session(year, round_num, "R")
                session.load(telemetry=False, weather=True, messages=True, laps=True)
            except Exception as exc2:
                logger.error("fastf1_load_failed year=%s round=%s: %s", year, round_num, exc2)
                return OutcomeReport(
                    year=year, round_num=round_num, track_id="unknown",
                    n_insights_examined=0, n_labeled_correct=0,
                    n_labeled_incorrect=0, n_no_match=0, incidents_found=[],
                )
        try:
            location = str(session.event.get("Location", "") or "")
            track_id = canonical_track_id(location)
            incidents = _extract_incidents(session)
        except Exception as exc2:
            logger.error("outcome_labeler_parse_failed year=%s round=%s: %s", year, round_num, exc2)
            return OutcomeReport(
                year=year, round_num=round_num, track_id="unknown",
                n_insights_examined=0, n_labeled_correct=0,
                n_labeled_incorrect=0, n_no_match=0, incidents_found=[],
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
        from sqlalchemy import select
    except Exception as exc:
        logger.warning("db_unavailable: %s", exc)
        return OutcomeReport(
            year=year, round_num=round_num, track_id=track_id,
            n_insights_examined=0, n_labeled_correct=0,
            n_labeled_incorrect=0, n_no_match=0,
            incidents_found=[{"driver": i.driver, "lap": i.lap, "type": i.incident_type} for i in incidents],
        )

    candidate_prefixes = _session_id_for_race(year, round_num, track_id)
    # Insights generated via the live stream / session replay use the OpenF1
    # session key directly: session_id = f"openf1_{session_key}".
    if openf1_session_key:
        candidate_prefixes.append(f"openf1_{openf1_session_key}")
    n_correct = 0
    n_incorrect = 0
    n_no_match = 0
    n_examined = 0

    try:
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
    except Exception as exc:
        logger.error("outcome_labeler_db_error year=%s round=%s: %s", year, round_num, exc)

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
