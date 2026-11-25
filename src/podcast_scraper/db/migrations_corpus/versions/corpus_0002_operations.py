"""corpus: migrations and receipts, runs, provenance, jobs and leases, assets

Revision ID: corpus_0002_operations
Revises: corpus_0001_bookkeeping
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0002_operations"
down_revision = "corpus_0001_bookkeeping"
branch_labels = None
depends_on = None


def _id(name: str) -> sa.Column:
    return sa.Column(name, sa.BigInteger(), sa.Identity(), primary_key=True)


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )


def upgrade() -> None:
    """Create the operations tables; ``ingestion_runs.source_id`` gets its key in corpus_0003."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "corpus_meta",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", postgresql.JSONB(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        schema=schema,
    )
    op.create_table(
        "corpus_migrations",
        _id("migration_id"),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        _created_at(),
        sa.UniqueConstraint("natural_id", name="corpus_migrations_natural_id_key"),
        sa.CheckConstraint(
            "status IN ('dry_run', 'applied', 'undone')", name="corpus_migrations_status_check"
        ),
        schema=schema,
    )
    op.create_table(
        "migration_receipts",
        _id("receipt_id"),
        sa.Column(
            "migration_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.corpus_migrations.migration_id"),
            nullable=False,
        ),
        sa.Column("target_table", sa.Text(), nullable=False),
        sa.Column("target_key", sa.Text(), nullable=False),
        sa.Column("before", postgresql.JSONB(), nullable=True),
        sa.Column("after", postgresql.JSONB(), nullable=True),
        _created_at(),
        schema=schema,
    )
    op.create_index(
        "migration_receipts_migration_idx", "migration_receipts", ["migration_id"], schema=schema
    )
    op.create_table(
        "ingestion_runs",
        _id("run_id"),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("source_id", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'running'")),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        _created_at(),
        sa.UniqueConstraint("natural_id", name="ingestion_runs_natural_id_key"),
        schema=schema,
    )
    op.create_index(
        "ingestion_runs_created_idx",
        "ingestion_runs",
        [sa.text("created_at DESC")],
        schema=schema,
    )
    op.create_table(
        "run_item_outcomes",
        _id("outcome_id"),
        sa.Column(
            "run_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.ingestion_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("item_id", sa.BigInteger(), nullable=True),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("detail", postgresql.JSONB(), nullable=True),
        _created_at(),
        schema=schema,
    )
    op.create_index("run_item_outcomes_run_idx", "run_item_outcomes", ["run_id"], schema=schema)
    op.create_index("run_item_outcomes_item_idx", "run_item_outcomes", ["item_id"], schema=schema)
    op.create_table(
        "processing_stages",
        _id("processing_stage_id"),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("method_version", sa.Text(), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("prompt_hash", sa.Text(), nullable=True),
        sa.Column("git_sha", sa.Text(), nullable=True),
        sa.Column(
            "run_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.ingestion_runs.run_id"),
            nullable=True,
        ),
        _created_at(),
        schema=schema,
    )
    op.create_index("processing_stages_run_idx", "processing_stages", ["run_id"], schema=schema)
    op.create_table(
        "enrichment_runs",
        _id("enrichment_run_id"),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("enricher_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'running'")),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        _created_at(),
        sa.UniqueConstraint("natural_id", name="enrichment_runs_natural_id_key"),
        schema=schema,
    )
    op.create_index(
        "enrichment_runs_created_idx",
        "enrichment_runs",
        [sa.text("created_at DESC")],
        schema=schema,
    )
    op.create_table(
        "jobs",
        _id("job_id"),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'queued'")),
        sa.Column("cursor", postgresql.JSONB(), nullable=True),
        _created_at(),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("natural_id", name="jobs_natural_id_key"),
        schema=schema,
    )
    op.create_index("jobs_created_idx", "jobs", [sa.text("created_at DESC")], schema=schema)
    op.create_table(
        "job_leases",
        _id("lease_id"),
        sa.Column("resource", sa.Text(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("holder", sa.Text(), nullable=False),
        sa.Column(
            "job_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.jobs.job_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "heartbeat_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        _created_at(),
        sa.UniqueConstraint("resource", "holder", name="job_leases_resource_holder_key"),
        sa.CheckConstraint("mode IN ('exclusive', 'shared')", name="job_leases_mode_check"),
        schema=schema,
    )
    op.create_index("job_leases_expires_idx", "job_leases", ["expires_at"], schema=schema)
    op.create_index("job_leases_job_idx", "job_leases", ["job_id"], schema=schema)
    op.create_table(
        "assets",
        _id("asset_id"),
        sa.Column("owner_kind", sa.Text(), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        _created_at(),
        sa.UniqueConstraint("owner_kind", "owner_id", "role", name="assets_owner_role_key"),
        schema=schema,
    )
