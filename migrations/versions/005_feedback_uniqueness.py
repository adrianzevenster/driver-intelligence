"""Add unique constraint on feedback(insight_id, submitted_by).

Prevents duplicate feedback rows from the same submitter per insight,
which was causing the flywheel to double-count null_outcome labels.

Revision ID: 005
Revises: 004
Create Date: 2026-07-05
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect, text


revision: str = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    # Remove duplicate rows before adding the constraint (keep the latest per pair).
    bind.execute(text(
        """
        DELETE FROM feedback
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM feedback
            GROUP BY insight_id, COALESCE(submitted_by, '')
        )
        """
    ))

    # SQLite doesn't support ADD CONSTRAINT; recreate the table.
    dialect = bind.dialect.name
    if dialect == "sqlite":
        bind.execute(text("ALTER TABLE feedback RENAME TO _feedback_old"))
        bind.execute(text(
            """
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_id VARCHAR(36) NOT NULL,
                rating INTEGER NOT NULL,
                correct BOOLEAN,
                comment TEXT,
                submitted_by VARCHAR(64),
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (insight_id, submitted_by)
            )
            """
        ))
        bind.execute(text(
            "INSERT INTO feedback SELECT id, insight_id, rating, correct, comment, submitted_by, created_at FROM _feedback_old"
        ))
        bind.execute(text("DROP TABLE _feedback_old"))
        bind.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_insight_id ON feedback (insight_id)"))
    else:
        op.create_unique_constraint(
            "uq_feedback_insight_submitter", "feedback", ["insight_id", "submitted_by"]
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "sqlite":
        pass  # recreating table just for downgrade is not worth it
    else:
        op.drop_constraint("uq_feedback_insight_submitter", "feedback", type_="unique")
