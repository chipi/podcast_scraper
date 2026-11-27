"""corpus: the news re-check's state per item and its window per source (CRFC-006 G7, W9)

Revision ID: corpus_0015_news_recheck
Revises: corpus_0014_projections
"""

import sqlalchemy as sa
from alembic import context, op

revision = "corpus_0015_news_recheck"
down_revision = "corpus_0014_projections"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``news_recheck_sources`` (a source's re-check window in days; a source with no row
    has a window of 0, so nothing is re-checked until the operator sets one: W9 ships dark) and
    ``news_recheck`` (an item's checks done, next due, last status, consecutive gone answers and
    the validators its page last sent). They sit in the corpus history, as ``audio_fingerprints``
    does for the podcast module: no module has a history of its own yet."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "news_recheck_sources",
        sa.Column("source_id", sa.BigInteger(), nullable=False),
        sa.Column("window_days", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("source_id", name="news_recheck_sources_pkey"),
        sa.ForeignKeyConstraint(
            ["source_id"],
            [f"{schema}.sources.source_id"],
            name="news_recheck_sources_source_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("window_days >= 0", name="news_recheck_sources_window_check"),
        schema=schema,
    )
    op.create_table(
        "news_recheck",
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("first_fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("checks_done", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("next_due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status", sa.Text(), nullable=True),
        sa.Column("gone_in_a_row", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("done", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("etag", sa.Text(), nullable=True),
        sa.Column("last_modified", sa.Text(), nullable=True),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("item_id", name="news_recheck_pkey"),
        sa.ForeignKeyConstraint(
            ["item_id"],
            [f"{schema}.content_items.item_id"],
            name="news_recheck_item_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "checks_done >= 0 AND gone_in_a_row >= 0", name="news_recheck_counts_check"
        ),
        sa.CheckConstraint("done = (next_due_at IS NULL)", name="news_recheck_due_check"),
        schema=schema,
    )
    op.create_index(
        "news_recheck_due_idx",
        "news_recheck",
        ["next_due_at"],
        schema=schema,
        postgresql_where=sa.text("NOT done"),
    )
