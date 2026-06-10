"""Initial schema: insights, feedback, ingestion_runs.

Revision ID: 001
Revises:
Create Date: 2026-06-08
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "insights",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("insight_id", sa.String(36), nullable=False),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("driver_id", sa.String(64), nullable=False),
        sa.Column("track_id", sa.String(64), nullable=False),
        sa.Column("lap", sa.Integer(), nullable=True),
        sa.Column("compound", sa.String(16), nullable=True),
        sa.Column("risk", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("uncertainty", sa.Float(), nullable=False),
        sa.Column("policy", sa.String(32), nullable=False),
        sa.Column("audience", sa.String(32), nullable=False),
        sa.Column("recommendation", sa.Text(), nullable=False, server_default=""),
        sa.Column("findings_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("evidence_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("insight_id"),
    )
    op.create_index("ix_insights_session_id", "insights", ["session_id"])
    op.create_index("ix_insights_driver_id", "insights", ["driver_id"])
    op.create_index("ix_insights_track_id", "insights", ["track_id"])
    op.create_index("ix_insights_risk", "insights", ["risk"])
    op.create_index("ix_insights_insight_id", "insights", ["insight_id"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("insight_id", sa.String(36), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("submitted_by", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedback_insight_id", "feedback", ["insight_id"])

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("round_num", sa.Integer(), nullable=False),
        sa.Column("track_id", sa.String(64), nullable=True),
        sa.Column("event_name", sa.String(128), nullable=True),
        sa.Column("documents_added", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "completed_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "year", "round_num", name="uq_ingestion_run"),
    )
    op.create_index("ix_ingestion_runs_source", "ingestion_runs", ["source"])


def downgrade() -> None:
    op.drop_table("ingestion_runs")
    op.drop_table("feedback")
    op.drop_table("insights")
