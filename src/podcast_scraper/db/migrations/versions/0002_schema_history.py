"""schema_history: one row per revision a ``db upgrade`` applied

Revision ID: 0002_schema_history
Revises: 0001_users
"""

import sqlalchemy as sa
from alembic import context, op

revision = "0002_schema_history"
down_revision = "0001_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the ``schema_history`` table."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "schema_history",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column(
            "applied_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("release", sa.Text(), nullable=True),
        sa.Column("release_source", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        schema=schema,
    )
