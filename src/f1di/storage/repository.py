"""Repository layer: clean interface between the domain and the database.

All queries live here so the rest of the codebase never imports SQLAlchemy
directly.  Pass a ``Session`` obtained from ``db_session()`` — the caller
owns the transaction.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from f1di.domain.schemas import DriverInsight, TelemetryWindow
from f1di.storage.models import FeedbackRecord, IngestionRecord, InsightRecord, JudgeScoreRecord, TelemetrySampleRecord


# ---------------------------------------------------------------------------
# Telemetry repository
# ---------------------------------------------------------------------------


def save_telemetry_bulk(session: Session, window: TelemetryWindow) -> None:
    """Save all samples from a window for analytical persistence."""
    records = []
    for s in window.samples:
        records.append(
            TelemetrySampleRecord(
                session_id=s.session_id,
                driver_id=s.driver_id,
                track_id=s.track_id,
                lap=s.lap,
                timestamp_ms=s.timestamp_ms,
                speed_kph=s.speed_kph,
                throttle_pct=s.throttle_pct,
                brake_pressure=s.brake_pressure_bar,
                compound=s.compound.value,
                stint_lap=s.stint_lap,
                tire_wear_fl=s.tire_wear_fl,
                tire_wear_fr=s.tire_wear_fr,
                tire_wear_rl=s.tire_wear_rl,
                tire_wear_rr=s.tire_wear_rr,
                grip_estimate=s.grip_estimate,
                battery_soc=s.battery_soc,
                track_temp_c=s.track_temp_c,
                rain_intensity=s.rain_intensity,
            )
        )
    session.add_all(records)


# ---------------------------------------------------------------------------
# Insight repository
# ---------------------------------------------------------------------------


def save_insight(
    session: Session,
    insight: DriverInsight,
    window: TelemetryWindow | None = None,
    *,
    shadow: bool = False,
    challenger_version: str | None = None,
) -> InsightRecord:
    record = InsightRecord(
        insight_id=insight.insight_id,
        session_id=insight.session_id,
        driver_id=insight.driver_id,
        track_id=window.track_id if window else "",
        lap=window.latest.lap if window else None,
        compound=window.latest.compound.value if window else None,
        risk=insight.risk.value,
        confidence=insight.confidence,
        uncertainty=insight.uncertainty,
        policy=insight.policy,
        audience=insight.audience.value,
        recommendation=insight.recommendation,
        findings_json=json.dumps(
            [
                {
                    "agent": f.agent,
                    "risk": f.risk.value,
                    "confidence": f.confidence,
                    "summary": f.summary,
                    "features": {
                        k: v for k, v in f.features.items() if isinstance(v, (int, float))
                    },
                }
                for f in insight.findings
            ]
        ),
        evidence_json=json.dumps(
            [
                {
                    "source_id": e.source_id,
                    "title": e.title,
                    "score": e.score,
                }
                for e in insight.evidence[:5]
            ]
        ),
        latency_ms=insight.latency_ms,
        shadow=shadow,
        challenger_version=challenger_version,
        raw_score=insight.raw_score,
    )
    session.add(record)
    return record


def list_insights(
    session: Session,
    *,
    driver_id: str | None = None,
    track_id: str | None = None,
    risk: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[InsightRecord]:
    stmt = select(InsightRecord).order_by(InsightRecord.created_at.desc())
    if driver_id:
        stmt = stmt.where(InsightRecord.driver_id == driver_id)
    if track_id:
        stmt = stmt.where(InsightRecord.track_id == track_id)
    if risk:
        stmt = stmt.where(InsightRecord.risk == risk)
    stmt = stmt.offset(offset).limit(limit)
    return list(session.scalars(stmt))


def driver_trend(
    session: Session,
    driver_id: str,
    days: int = 30,
) -> dict[str, Any]:
    """Aggregate statistics for a driver over the last N days."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    stmt = (
        select(
            InsightRecord.risk,
            func.count(InsightRecord.id).label("count"),
            func.avg(InsightRecord.confidence).label("avg_conf"),
        )
        .where(InsightRecord.driver_id == driver_id)
        .where(InsightRecord.created_at >= cutoff)
        .group_by(InsightRecord.risk)
    )
    rows = session.execute(stmt).all()
    return {
        "driver_id": driver_id,
        "period_days": days,
        "by_risk": {r.risk: {"count": r.count, "avg_confidence": round(r.avg_conf, 4)} for r in rows},
        "total": sum(r.count for r in rows),
    }


def circuit_heatmap(session: Session, track_id: str) -> dict[str, Any]:
    """Risk distribution and average confidence for a circuit across all history."""
    stmt = (
        select(
            InsightRecord.risk,
            InsightRecord.driver_id,
            func.count(InsightRecord.id).label("count"),
            func.avg(InsightRecord.confidence).label("avg_conf"),
        )
        .where(InsightRecord.track_id == track_id)
        .group_by(InsightRecord.risk, InsightRecord.driver_id)
        .order_by(InsightRecord.risk)
    )
    rows = session.execute(stmt).all()
    return {
        "track_id": track_id,
        "rows": [
            {
                "risk": r.risk,
                "driver_id": r.driver_id,
                "count": r.count,
                "avg_confidence": round(r.avg_conf, 4),
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Judge score repository
# ---------------------------------------------------------------------------


def save_judge_score(
    session: Session,
    *,
    insight_id: str,
    safety: float,
    actionability: float,
    register: float,
    calibration: float,
    mean_score: float,
    rationale: str = "",
) -> JudgeScoreRecord:
    record = JudgeScoreRecord(
        insight_id=insight_id,
        safety=safety,
        actionability=actionability,
        register=register,
        calibration=calibration,
        mean_score=mean_score,
        rationale=rationale,
    )
    session.merge(record)
    return record


def get_judge_score(session: Session, insight_id: str) -> JudgeScoreRecord | None:
    return session.scalar(
        select(JudgeScoreRecord).where(JudgeScoreRecord.insight_id == insight_id)
    )


def insight_exists(session: Session, insight_id: str) -> bool:
    return session.scalar(
        select(InsightRecord.id).where(InsightRecord.insight_id == insight_id).limit(1)
    ) is not None


def get_judge_scores_bulk(session: Session, insight_ids: list[str]) -> dict[str, float]:
    """Return {insight_id: mean_score} for all scored insights in the given list."""
    if not insight_ids:
        return {}
    rows = session.execute(
        select(JudgeScoreRecord.insight_id, JudgeScoreRecord.mean_score).where(
            JudgeScoreRecord.insight_id.in_(insight_ids)
        )
    ).all()
    return {r.insight_id: r.mean_score for r in rows}


# ---------------------------------------------------------------------------
# Feedback repository
# ---------------------------------------------------------------------------


def save_feedback(
    session: Session,
    *,
    insight_id: str,
    rating: int,
    correct: bool | None = None,
    comment: str | None = None,
    submitted_by: str | None = None,
) -> FeedbackRecord:
    record = FeedbackRecord(
        insight_id=insight_id,
        rating=max(1, min(5, rating)),
        correct=correct,
        comment=comment,
        submitted_by=submitted_by,
    )
    session.add(record)
    return record


def feedback_for_insight(session: Session, insight_id: str) -> list[FeedbackRecord]:
    stmt = select(FeedbackRecord).where(FeedbackRecord.insight_id == insight_id)
    return list(session.scalars(stmt))


def review_queue(
    session: Session,
    *,
    limit: int = 50,
) -> list[InsightRecord]:
    """Return insights that have no feedback, ordered by risk severity then uncertainty."""
    from sqlalchemy import case

    risk_rank = case(
        {"CRITICAL": 3, "WARNING": 2, "WATCH": 1, "INFO": 0},
        value=InsightRecord.risk,
        else_=0,
    )
    feedback_exists = (
        select(FeedbackRecord.id)
        .where(FeedbackRecord.insight_id == InsightRecord.insight_id)
        .correlate(InsightRecord)
        .exists()
    )
    stmt = (
        select(InsightRecord)
        .where(~feedback_exists)
        .order_by(risk_rank.desc(), InsightRecord.uncertainty.desc(), InsightRecord.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def shadow_compare(
    session: Session,
    challenger_version: str,
    *,
    limit: int = 200,
) -> dict[str, Any]:
    """Compare shadow vs. production insight distributions for a challenger version."""

    def _distribution(records: list[InsightRecord]) -> dict[str, Any]:
        if not records:
            return {"n": 0}
        risk_counts: dict[str, int] = {}
        for r in records:
            risk_counts[r.risk] = risk_counts.get(r.risk, 0) + 1
        avg_conf = sum(r.confidence for r in records) / len(records)
        avg_unc = sum(r.uncertainty for r in records) / len(records)
        return {
            "n": len(records),
            "risk_distribution": risk_counts,
            "avg_confidence": round(avg_conf, 4),
            "avg_uncertainty": round(avg_unc, 4),
        }

    shadow_stmt = (
        select(InsightRecord)
        .where(InsightRecord.shadow.is_(True))
        .where(InsightRecord.challenger_version == challenger_version)
        .order_by(InsightRecord.created_at.desc())
        .limit(limit)
    )
    prod_stmt = (
        select(InsightRecord)
        .where(InsightRecord.shadow.is_(False))
        .order_by(InsightRecord.created_at.desc())
        .limit(limit)
    )
    shadow_records = list(session.scalars(shadow_stmt))
    prod_records = list(session.scalars(prod_stmt))
    return {
        "challenger_version": challenger_version,
        "production": _distribution(prod_records),
        "shadow": _distribution(shadow_records),
    }


# ---------------------------------------------------------------------------
# Ingestion tracking
# ---------------------------------------------------------------------------


def mark_ingested(
    session: Session,
    *,
    source: str,
    year: int,
    round_num: int,
    track_id: str = "",
    event_name: str = "",
    documents_added: int = 0,
) -> IngestionRecord:
    record = IngestionRecord(
        source=source,
        year=year,
        round_num=round_num,
        track_id=track_id,
        event_name=event_name,
        documents_added=documents_added,
    )
    session.merge(record)
    return record


def already_ingested(session: Session, *, source: str, year: int, round_num: int) -> bool:
    stmt = (
        select(IngestionRecord.id)
        .where(IngestionRecord.source == source)
        .where(IngestionRecord.year == year)
        .where(IngestionRecord.round_num == round_num)
        .limit(1)
    )
    return session.scalar(stmt) is not None


def list_ingestion_runs(session: Session, source: str | None = None) -> list[IngestionRecord]:
    stmt = select(IngestionRecord).order_by(IngestionRecord.completed_at.desc())
    if source:
        stmt = stmt.where(IngestionRecord.source == source)
    return list(session.scalars(stmt))


def shadow_evaluate(
    session: Session,
    challenger_version: str,
    *,
    limit: int = 500,
    min_n: int = 30,
) -> dict[str, Any]:
    """Statistical comparison of shadow challenger vs production.

    Returns a promotion recommendation based on Mann-Whitney U test on
    confidence distributions and a risk-escalation rate comparison.
    """
    shadow_stmt = (
        select(InsightRecord)
        .where(InsightRecord.shadow.is_(True))
        .where(InsightRecord.challenger_version == challenger_version)
        .order_by(InsightRecord.created_at.desc())
        .limit(limit)
    )
    prod_stmt = (
        select(InsightRecord)
        .where(InsightRecord.shadow.is_(False))
        .order_by(InsightRecord.created_at.desc())
        .limit(limit)
    )
    shadow_records = list(session.scalars(shadow_stmt))
    prod_records = list(session.scalars(prod_stmt))

    n_shadow = len(shadow_records)
    n_prod = len(prod_records)

    if n_shadow < min_n or n_prod < min_n:
        return {
            "challenger_version": challenger_version,
            "recommendation": "insufficient_data",
            "n_shadow": n_shadow,
            "n_prod": n_prod,
            "min_n": min_n,
            "promote": False,
        }

    shadow_conf = [r.confidence for r in shadow_records]
    prod_conf = [r.confidence for r in prod_records]

    u_stat, p_value = _mann_whitney_u(shadow_conf, prod_conf)
    n1, n2 = len(shadow_conf), len(prod_conf)
    rbc = 1.0 - (2 * u_stat) / (n1 * n2)

    risk_weights = {"CRITICAL": 3, "WARNING": 2, "WATCH": 1, "INFO": 0}

    def _escalation_rate(records):
        total = len(records)
        weighted = sum(risk_weights.get(r.risk, 0) for r in records)
        return weighted / (total * 3) if total > 0 else 0.0

    shadow_escalation = _escalation_rate(shadow_records)
    prod_escalation = _escalation_rate(prod_records)

    promote = (
        p_value < 0.05
        and rbc > 0.05
        and shadow_escalation <= prod_escalation * 1.10
    )

    return {
        "challenger_version": challenger_version,
        "n_shadow": n_shadow,
        "n_prod": n_prod,
        "shadow_mean_confidence": round(sum(shadow_conf) / n_shadow, 4),
        "prod_mean_confidence": round(sum(prod_conf) / n_prod, 4),
        "u_statistic": round(u_stat, 1),
        "p_value": round(p_value, 4),
        "rank_biserial_correlation": round(rbc, 4),
        "shadow_escalation_rate": round(shadow_escalation, 4),
        "prod_escalation_rate": round(prod_escalation, 4),
        "promote": promote,
        "recommendation": (
            "promote_challenger" if promote
            else "insufficient_effect" if p_value >= 0.05
            else "significant_but_escalates_risk" if shadow_escalation > prod_escalation * 1.10
            else "no_improvement"
        ),
    }


def _mann_whitney_u(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Mann-Whitney U statistic and approximate p-value (normal approximation)."""
    import math
    n1, n2 = len(x), len(y)
    u = sum(1 for xi in x for yj in y if xi > yj) + 0.5 * sum(1 for xi in x for yj in y if xi == yj)
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma < 1e-9:
        return u, 1.0
    z = (u - mu) / sigma
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p_one_tail = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
    p_value = p_one_tail if z >= 0 else 1.0 - p_one_tail
    return u, max(0.0, min(1.0, p_value))
