"""Add telemetry and judge_scores tables.

These tables existed in models.py and were created via create_all() on SQLite
but were never added to a migration. This revision adds them explicitly so
Postgres deployments get the full schema via alembic upgrade head.

Existing SQLite dev databases that already have these tables (from create_all)
are handled via an existence check — the tables are skipped if already present.

Revision ID: 004
Revises: 003
Create Date: 2026-06-17
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = inspect(op.get_bind()).get_table_names()

    if "judge_scores" not in existing:
        op.create_table(
            "judge_scores",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("insight_id", sa.String(36), nullable=False),
            sa.Column("safety", sa.Float(), nullable=False),
            sa.Column("actionability", sa.Float(), nullable=False),
            sa.Column("register", sa.Float(), nullable=False),
            sa.Column("calibration", sa.Float(), nullable=False),
            sa.Column("mean_score", sa.Float(), nullable=False),
            sa.Column("rationale", sa.Text(), nullable=False, server_default=""),
            sa.Column(
                "scored_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("insight_id"),
        )
        op.create_index("ix_judge_scores_insight_id", "judge_scores", ["insight_id"])

    if "telemetry" not in existing:
        op.create_table(
            "telemetry",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("session_id", sa.String(128), nullable=False),
            sa.Column("driver_id", sa.String(64), nullable=False),
            sa.Column("track_id", sa.String(64), nullable=False),
            sa.Column("lap", sa.Integer(), nullable=False),
            sa.Column("timestamp_ms", sa.Integer(), nullable=False),
            sa.Column("speed_kph", sa.Float(), nullable=False),
            sa.Column("throttle_pct", sa.Float(), nullable=False),
            sa.Column("brake_pressure", sa.Float(), nullable=False),
            sa.Column("compound", sa.String(16), nullable=False),
            sa.Column("stint_lap", sa.Integer(), nullable=False),
            sa.Column("tire_wear_fl", sa.Float(), nullable=False),
            sa.Column("tire_wear_fr", sa.Float(), nullable=False),
            sa.Column("tire_wear_rl", sa.Float(), nullable=False),
            sa.Column("tire_wear_rr", sa.Float(), nullable=False),
            sa.Column("grip_estimate", sa.Float(), nullable=False),
            sa.Column("battery_soc", sa.Float(), nullable=False),
            sa.Column("track_temp_c", sa.Float(), nullable=False),
            sa.Column("rain_intensity", sa.Float(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_telemetry_session_id", "telemetry", ["session_id"])
        op.create_index("ix_telemetry_driver_id", "telemetry", ["driver_id"])
        op.create_index("ix_telemetry_track_id", "telemetry", ["track_id"])
        op.create_index("ix_telemetry_lap", "telemetry", ["lap"])


def downgrade() -> None:
    op.drop_table("telemetry")
    op.drop_table("judge_scores")
