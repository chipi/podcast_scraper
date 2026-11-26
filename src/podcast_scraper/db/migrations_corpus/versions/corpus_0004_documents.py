"""corpus: documents and the hash-partitioned segments

Revision ID: corpus_0004_documents
Revises: corpus_0003_sources_and_items
"""

import sqlalchemy as sa
from alembic import context, op

revision = "corpus_0004_documents"
down_revision = "corpus_0003_sources_and_items"
branch_labels = None
depends_on = None

_PARTITIONS = 16


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


def upgrade() -> None:
    """Create ``documents``, then ``segments`` and its sixteen partitions."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "documents",
        _id("document_id"),
        sa.Column(
            "item_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.content_items.item_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("variant", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("supersedes_id", sa.BigInteger(), nullable=True),
        sa.Column("change_kind", sa.Text(), nullable=True),
        sa.Column("body_sha256", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extraction", sa.Text(), nullable=True),
        sa.Column("extractor_version", sa.Text(), nullable=True),
        _created_at(),
        _updated_at(),
        sa.UniqueConstraint(
            "item_id",
            "kind",
            "variant",
            "version",
            name="documents_item_kind_variant_version_key",
        ),
        sa.UniqueConstraint("supersedes_id", name="documents_supersedes_key"),
        sa.UniqueConstraint("item_id", "document_id", name="documents_item_document_key"),
        sa.UniqueConstraint(
            "item_id", "kind", "variant", "document_id", name="documents_series_document_key"
        ),
        # A version supersedes one of its own item, kind and variant, never another series.
        sa.ForeignKeyConstraint(
            ["item_id", "kind", "variant", "supersedes_id"],
            [
                f"{schema}.documents.item_id",
                f"{schema}.documents.kind",
                f"{schema}.documents.variant",
                f"{schema}.documents.document_id",
            ],
            name="documents_supersedes_fkey",
        ),
        sa.CheckConstraint("kind IN ('transcript', 'article')", name="documents_kind_check"),
        sa.CheckConstraint(
            "variant IN ('raw', 'cleaned', 'adfree', 'en')", name="documents_variant_check"
        ),
        sa.CheckConstraint("version >= 1", name="documents_version_check"),
        schema=schema,
    )
    op.create_table(
        "segments",
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=True),
        sa.Column("end_ms", sa.Integer(), nullable=True),
        sa.Column("speaker_id", sa.BigInteger(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("item_id", "document_id", "seq", name="segments_pkey"),
        # The partition key must be the document's own item, so the pair is the reference.
        sa.ForeignKeyConstraint(
            ["item_id", "document_id"],
            [f"{schema}.documents.item_id", f"{schema}.documents.document_id"],
            name="segments_document_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "start_ms IS NULL OR end_ms IS NULL OR end_ms >= start_ms", name="segments_span_check"
        ),
        schema=schema,
        postgresql_partition_by="HASH (item_id)",
    )
    for n in range(_PARTITIONS):
        op.execute(
            f"CREATE TABLE {schema}.segments_p{n:02d} PARTITION OF {schema}.segments "
            f"FOR VALUES WITH (MODULUS {_PARTITIONS}, REMAINDER {n}) "
            "WITH (autovacuum_vacuum_scale_factor = 0.01, autovacuum_analyze_scale_factor = 0.005)"
        )
