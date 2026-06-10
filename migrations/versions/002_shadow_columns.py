"""Add shadow and challenger_version columns to insights table.

Revision ID: 002
Revises: 001
Create Date: 2026-06-09
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "insights",
        sa.Column("shadow", sa.Boolean(), nullable=False, server_default="0"),
    )
    op.add_column(
        "insights",
        sa.Column("challenger_version", sa.String(64), nullable=True),
    )
    op.create_index("ix_insights_shadow", "insights", ["shadow"])


def downgrade() -> None:
    op.drop_index("ix_insights_shadow", table_name="insights")
    op.drop_column("insights", "challenger_version")
    op.drop_column("insights", "shadow")
