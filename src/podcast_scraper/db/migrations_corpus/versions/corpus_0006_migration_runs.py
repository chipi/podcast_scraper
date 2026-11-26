"""corpus: every attempt of a corpus migration on rows, and the receipts tied to it (RFC-135)

Revision ID: corpus_0006_migration_runs
Revises: corpus_0005_entities
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0006_migration_runs"
down_revision = "corpus_0005_entities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add ``migration_runs``; tie receipts and the ledger row to a run (expand only)."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "migration_runs",
        sa.Column("run_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("code_digest", sa.Text(), nullable=True),
        sa.Column("outcome_sha256", sa.Text(), nullable=True),
        sa.Column("plan", postgresql.JSONB(), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('dry_run', 'apply', 'undo', 'rerun')", name="migration_runs_kind_check"
        ),
        sa.CheckConstraint(
            "status IN ('ok', 'refused', 'failed')", name="migration_runs_status_check"
        ),
        schema=schema,
    )
    op.create_index(
        "migration_runs_natural_idx", "migration_runs", ["natural_id", "run_id"], schema=schema
    )
    # Nullable because the columns are added to tables that may already hold rows; the framework
    # always sets them.
    op.add_column(
        "migration_receipts",
        sa.Column(
            "run_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.migration_runs.run_id"),
            nullable=True,
        ),
        schema=schema,
    )
    op.add_column(
        "migration_receipts", sa.Column("seq", sa.Integer(), nullable=True), schema=schema
    )
    op.create_index(
        "migration_receipts_run_idx", "migration_receipts", ["run_id", "seq"], schema=schema
    )
    op.add_column(
        "corpus_migrations", sa.Column("to_version", sa.Text(), nullable=True), schema=schema
    )
    op.add_column(
        "corpus_migrations",
        sa.Column(
            "last_run_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.migration_runs.run_id"),
            nullable=True,
        ),
        schema=schema,
    )
