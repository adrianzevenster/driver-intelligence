from __future__ import annotations

import datetime
import json

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class InsightRecord(Base):
    __tablename__ = "insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    insight_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    driver_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    track_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    lap: Mapped[int | None] = mapped_column(Integer, nullable=True)
    compound: Mapped[str | None] = mapped_column(String(16), nullable=True)
    risk: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty: Mapped[float] = mapped_column(Float, nullable=False)
    raw_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    policy: Mapped[str] = mapped_column(String(32), nullable=False)
    audience: Mapped[str] = mapped_column(String(32), nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    findings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    shadow: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false", index=True)
    challenger_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    @property
    def findings(self) -> list[dict]:
        return json.loads(self.findings_json)

    @property
    def evidence(self) -> list[dict]:
        return json.loads(self.evidence_json)


class FeedbackRecord(Base):
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("insight_id", "submitted_by", name="uq_feedback_insight_submitter"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    insight_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    comment: Mapped[str] = mapped_column(Text, nullable=True)
    submitted_by: Mapped[str] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class JudgeScoreRecord(Base):
    __tablename__ = "judge_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    insight_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    safety: Mapped[float] = mapped_column(Float, nullable=False)
    actionability: Mapped[float] = mapped_column(Float, nullable=False)
    register: Mapped[float] = mapped_column(Float, nullable=False)
    calibration: Mapped[float] = mapped_column(Float, nullable=False)
    mean_score: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    scored_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class IngestionRecord(Base):
    __tablename__ = "ingestion_runs"
    __table_args__ = (UniqueConstraint("source", "year", "round_num", name="uq_ingestion_run"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    round_num: Mapped[int] = mapped_column(Integer, nullable=False)
    track_id: Mapped[str] = mapped_column(String(64), nullable=True)
    event_name: Mapped[str] = mapped_column(String(128), nullable=True)
    documents_added: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class TelemetrySampleRecord(Base):
    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    driver_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    track_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    lap: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    speed_kph: Mapped[float] = mapped_column(Float, nullable=False)
    throttle_pct: Mapped[float] = mapped_column(Float, nullable=False)
    brake_pressure: Mapped[float] = mapped_column(Float, nullable=False)
    compound: Mapped[str] = mapped_column(String(16), nullable=False)
    stint_lap: Mapped[int] = mapped_column(Integer, nullable=False)
    tire_wear_fl: Mapped[float] = mapped_column(Float, nullable=False)
    tire_wear_fr: Mapped[float] = mapped_column(Float, nullable=False)
    tire_wear_rl: Mapped[float] = mapped_column(Float, nullable=False)
    tire_wear_rr: Mapped[float] = mapped_column(Float, nullable=False)
    grip_estimate: Mapped[float] = mapped_column(Float, nullable=False)
    battery_soc: Mapped[float] = mapped_column(Float, nullable=False)
    track_temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    rain_intensity: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
