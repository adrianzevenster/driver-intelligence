"""Real-race backtesting: precision/recall from stored InsightRecord + FeedbackRecord.

Computes per-round and overall precision of WARNING/CRITICAL predictions that were
confirmed (or refuted) by the outcome_labeler. Unlike synthetic tests, this uses
real race data flowing through the live DB — it answers "is the model improving?"

Precision = confirmed_correct / (confirmed_correct + confirmed_incorrect)
  where "confirmed" means FeedbackRecord.submitted_by = 'outcome_labeler'

Recall is not computable here (we'd need to know all incidents we MISSED), so we
track precision over time as the primary improvement signal.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("f1di.evaluation.race_backtest")

_REPORT_PATH = Path("data/calibration/backtest_report.json")
_PRECISION_ALERT_THRESHOLD = 0.20
_MIN_LABELS_PER_SESSION = 10


def run_backtest(n_sessions: int = 10) -> dict:
    """Compute precision from the most recent N labeled sessions.

    Returns a dict with overall precision, per-session breakdown, trend
    (improving / degrading / stable vs the prior report), and alert status.
    Saves the result to data/calibration/backtest_report.json.
    """
    try:
        from sqlalchemy import func, select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
    except ImportError:
        return {"error": "persistence layer not installed"}

    rows: list[dict] = []

    with db_session() as session:
        # Per-session precision from outcome_labeler labels on WARNING/CRITICAL insights
        stmt = (
            select(
                InsightRecord.session_id,
                InsightRecord.track_id,
                FeedbackRecord.correct,
                func.count().label("n"),
            )
            .join(FeedbackRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
            .where(FeedbackRecord.submitted_by == "outcome_labeler")
            .where(InsightRecord.risk.in_(["WARNING", "CRITICAL"]))
            .where(InsightRecord.shadow == False)  # noqa: E712
            .group_by(InsightRecord.session_id, InsightRecord.track_id, FeedbackRecord.correct)
            .order_by(InsightRecord.session_id)
        )
        db_rows = session.execute(stmt).all()

        # Aggregate per session_id
        by_session: dict[str, dict] = {}
        for row in db_rows:
            s = by_session.setdefault(row.session_id, {
                "session_id": row.session_id,
                "track_id": row.track_id or "",
                "n_correct": 0,
                "n_incorrect": 0,
            })
            if row.correct is True:
                s["n_correct"] += row.n
            elif row.correct is False:
                s["n_incorrect"] += row.n

    # Filter to sessions with enough labels and sort by total desc (most active first)
    qualified = [
        s for s in by_session.values()
        if s["n_correct"] + s["n_incorrect"] >= _MIN_LABELS_PER_SESSION
    ]
    qualified.sort(key=lambda s: s["n_correct"] + s["n_incorrect"], reverse=True)
    recent = qualified[:n_sessions]

    for s in recent:
        total = s["n_correct"] + s["n_incorrect"]
        s["precision"] = round(s["n_correct"] / total, 4) if total > 0 else None
        s["n_total"] = total
        rows.append(s)

    # Overall precision
    total_correct = sum(s["n_correct"] for s in rows)
    total_incorrect = sum(s["n_incorrect"] for s in rows)
    total = total_correct + total_incorrect
    overall_precision = round(total_correct / total, 4) if total > 0 else None

    # Trend vs last report
    trend = "unknown"
    prev_precision: float | None = None
    if _REPORT_PATH.exists():
        try:
            prev = json.loads(_REPORT_PATH.read_text())
            prev_precision = prev.get("overall_precision")
            if overall_precision is not None and prev_precision is not None:
                delta = overall_precision - prev_precision
                if delta > 0.02:
                    trend = "improving"
                elif delta < -0.02:
                    trend = "degrading"
                else:
                    trend = "stable"
        except Exception:
            pass

    alert = (
        overall_precision is not None
        and overall_precision < _PRECISION_ALERT_THRESHOLD
        and total >= 50
    )

    import datetime
    result = {
        "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_precision": overall_precision,
        "previous_precision": prev_precision,
        "trend": trend,
        "alert": alert,
        "alert_threshold": _PRECISION_ALERT_THRESHOLD,
        "n_total": total,
        "n_correct": total_correct,
        "n_incorrect": total_incorrect,
        "n_sessions": len(rows),
        "sessions": rows,
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(result, indent=2))

    logger.info(
        "race_backtest: precision=%.3f (prev=%.3f) trend=%s n=%d sessions=%d alert=%s",
        overall_precision or 0, prev_precision or 0, trend, total, len(rows), alert,
    )
    return result


def load_last_report() -> dict | None:
    if not _REPORT_PATH.exists():
        return None
    try:
        return json.loads(_REPORT_PATH.read_text())
    except Exception:
        return None


# Cache for circuit_precision_lookup — reloaded on file mtime change.
_CIRCUIT_PRECISION_CACHE: dict[str, float] = {}
_CIRCUIT_PRECISION_MTIME: float = 0.0
_CIRCUIT_PRECISION_OVERALL: float = 0.28
_MIN_CIRCUIT_N = 500  # minimum labeled examples to trust a circuit-level precision


def circuit_precision_lookup(track_id: str) -> float:
    """Return historical precision for *track_id* from the cached backtest report.

    Uses the overall precision as a fallback for unknown/sparse circuits.
    Reloads from disk when backtest_report.json changes.
    """
    global _CIRCUIT_PRECISION_CACHE, _CIRCUIT_PRECISION_MTIME, _CIRCUIT_PRECISION_OVERALL
    try:
        mtime = _REPORT_PATH.stat().st_mtime
        if mtime != _CIRCUIT_PRECISION_MTIME:
            data = json.loads(_REPORT_PATH.read_text())
            _CIRCUIT_PRECISION_OVERALL = data.get("overall_precision") or 0.28
            _CIRCUIT_PRECISION_CACHE = {
                s["track_id"]: s["precision"]
                for s in data.get("sessions", [])
                if s.get("n_total", 0) >= _MIN_CIRCUIT_N and s.get("precision") is not None
            }
            _CIRCUIT_PRECISION_MTIME = mtime
    except Exception:
        pass
    return _CIRCUIT_PRECISION_CACHE.get(track_id or "", _CIRCUIT_PRECISION_OVERALL)
