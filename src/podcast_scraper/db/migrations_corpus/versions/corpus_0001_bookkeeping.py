"""corpus: the schema's own schema_history

Revision ID: corpus_0001_bookkeeping
Revises:
"""

import sqlalchemy as sa
from alembic import context, op

revision = "corpus_0001_bookkeeping"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``schema_history`` in the corpus schema; infra creates the schema."""
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
