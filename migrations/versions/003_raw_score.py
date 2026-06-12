"""Add raw_score column to insights table.

Revision ID: 003
Revises: 002
Create Date: 2026-06-11
"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision = "002"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column("insights", sa.Column("raw_score", sa.Float(), nullable=True))

def downgrade() -> None:
    op.drop_column("insights", "raw_score")
