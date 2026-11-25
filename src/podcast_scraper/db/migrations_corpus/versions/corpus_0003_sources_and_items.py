"""corpus: sources, content_items and item_cards

Revision ID: corpus_0003_sources_and_items
Revises: corpus_0002_operations
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0003_sources_and_items"
down_revision = "corpus_0002_operations"
branch_labels = None
depends_on = None

_FLAGS = ("has_transcript", "has_gi", "has_kg", "has_moments", "withheld", "unusable")


def _id(name: str) -> sa.Column:
    return sa.Column(name, sa.BigInteger(), sa.Identity(), primary_key=True)


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )


def _updated_at() -> sa.Column:
    return sa.Column(
        "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )


def _flags() -> list[sa.Column]:
    return [
        sa.Column(name, sa.Boolean(), nullable=False, server_default=sa.text("false"))
        for name in _FLAGS
    ]


def upgrade() -> None:
    """Create the three tables and their indexes, then the keys the operations tables wait for."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "sources",
        _id("source_id"),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("raw_feed_id", sa.Text(), nullable=True),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("publisher_id", sa.BigInteger(), nullable=True),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("licence", sa.Text(), nullable=True),
        sa.Column(
            "artwork_asset_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.assets.asset_id", ondelete="SET NULL"),
            nullable=True,
        ),
        _created_at(),
        _updated_at(),
        sa.UniqueConstraint("natural_id", name="sources_natural_id_key"),
        sa.UniqueConstraint("raw_feed_id", name="sources_raw_feed_id_key"),
        schema=schema,
    )
    op.create_index("sources_artwork_idx", "sources", ["artwork_asset_id"], schema=schema)
    op.create_table(
        "content_items",
        _id("item_id"),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column(
            "source_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.sources.source_id"),
            nullable=False,
        ),
        sa.Column("external_item_id", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("editorial_type", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("filed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_s", sa.Integer(), nullable=True),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("served", sa.SmallInteger(), nullable=False, server_default=sa.text("0")),
        *_flags(),
        sa.Column(
            "type_attrs", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        _created_at(),
        _updated_at(),
        sa.UniqueConstraint("natural_id", name="content_items_natural_id_key"),
        sa.UniqueConstraint("slug", name="content_items_slug_key"),
        sa.UniqueConstraint(
            "source_id", "external_item_id", name="content_items_source_external_key"
        ),
        sa.CheckConstraint("served IN (0, 1, 2)", name="content_items_served_check"),
        schema=schema,
    )
    op.create_index(
        "content_items_published_idx",
        "content_items",
        [sa.text("published_at DESC"), "item_id"],
        schema=schema,
        postgresql_where=sa.text("served > 0"),
    )
    op.create_index(
        "content_items_source_published_idx",
        "content_items",
        ["source_id", sa.text("published_at DESC")],
        schema=schema,
        postgresql_where=sa.text("served > 0"),
    )
    op.create_index(
        "content_items_type_published_idx",
        "content_items",
        ["content_type", sa.text("published_at DESC")],
        schema=schema,
        postgresql_where=sa.text("served > 0"),
    )
    op.create_index(
        "content_items_title_trgm_idx",
        "content_items",
        ["title"],
        schema=schema,
        postgresql_using="gin",
        postgresql_ops={"title": "gin_trgm_ops"},
    )
    op.create_table(
        "item_cards",
        sa.Column(
            "item_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.content_items.item_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_title", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_s", sa.Integer(), nullable=True),
        sa.Column(
            "artwork_asset_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.assets.asset_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("lede", sa.Text(), nullable=True),
        sa.Column(
            "summary_bullets",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "top_topics", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        *_flags(),
        _created_at(),
        _updated_at(),
        schema=schema,
    )
    op.create_index("item_cards_artwork_idx", "item_cards", ["artwork_asset_id"], schema=schema)
    op.create_foreign_key(
        "ingestion_runs_source_id_fkey",
        "ingestion_runs",
        "sources",
        ["source_id"],
        ["source_id"],
        source_schema=schema,
        referent_schema=schema,
    )
    op.create_foreign_key(
        "run_item_outcomes_item_id_fkey",
        "run_item_outcomes",
        "content_items",
        ["item_id"],
        ["item_id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
    op.create_index("ingestion_runs_source_idx", "ingestion_runs", ["source_id"], schema=schema)
