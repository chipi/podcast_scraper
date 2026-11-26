"""corpus: search chunks, embeddings, and the corpus revision counter

Revision ID: corpus_0007_search_chunks
Revises: corpus_0006_migration_runs
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0007_search_chunks"
down_revision = "corpus_0006_migration_runs"
branch_labels = None
depends_on = None

# The search index's doc types today (search/indexer.py, two_tier_indexer.py).
_KINDS = (
    "'insight', 'quote', 'transcript', 'summary', 'summary_short', 'kg_topic', 'kg_entity', "
    "'episode_title', 'episode_description'"
)


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )


def upgrade() -> None:
    """Create ``search_chunks``, ``embeddings`` and ``corpus_revision`` with its one row."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "search_chunks",
        sa.Column("chunk_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("start_seq", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_seq", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column(
            "speakers",
            postgresql.ARRAY(sa.BigInteger()),
            nullable=False,
            server_default=sa.text("'{}'::bigint[]"),
        ),
        _created_at(),
        sa.UniqueConstraint("natural_id", name="search_chunks_natural_id_key"),
        sa.ForeignKeyConstraint(
            ["item_id", "document_id"],
            [f"{schema}.documents.item_id", f"{schema}.documents.document_id"],
            name="search_chunks_document_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "start_seq >= 0 AND start_offset >= 0 AND end_offset >= 0",
            name="search_chunks_bounds_check",
        ),
        # A chunk is a non-empty span: its end is after its start in (seq, offset) order.
        sa.CheckConstraint(
            "(end_seq, end_offset) > (start_seq, start_offset)", name="search_chunks_span_check"
        ),
        schema=schema,
    )
    op.create_index(
        "search_chunks_document_idx",
        "search_chunks",
        ["item_id", "document_id", "start_seq"],
        schema=schema,
    )
    op.create_index(
        "search_chunks_speakers_idx",
        "search_chunks",
        ["speakers"],
        schema=schema,
        postgresql_using="gin",
    )
    # Keyed by the object's natural id: most kinds have no table yet. The vector is real[] until
    # E09 picks its type (halfvec, binary with a re-rank, or a search node); pgvector is not here.
    op.create_table(
        "embeddings",
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("object_id", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.Text(), nullable=False),
        sa.Column("dims", sa.SmallInteger(), nullable=False),
        sa.Column("vector", postgresql.ARRAY(sa.REAL()), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("kind", "object_id", "model", name="embeddings_pkey"),
        sa.CheckConstraint(f"kind IN ({_KINDS})", name="embeddings_kind_check"),
        sa.CheckConstraint(
            "dims >= 1 AND array_ndims(vector) = 1 AND cardinality(vector) = dims",
            name="embeddings_dims_check",
        ),
        schema=schema,
    )
    op.create_index("embeddings_model_idx", "embeddings", ["model"], schema=schema)
    # DECISIONS rule 7: one counter every content transaction bumps, last in the lock order.
    op.create_table(
        "corpus_revision",
        sa.Column("id", sa.SmallInteger(), primary_key=True, server_default=sa.text("1")),
        sa.Column("revision", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("id = 1", name="corpus_revision_one_row_check"),
        sa.CheckConstraint("revision >= 0", name="corpus_revision_revision_check"),
        schema=schema,
    )
    revision_table = sa.table(
        "corpus_revision",
        sa.column("id", sa.SmallInteger()),
        sa.column("revision", sa.BigInteger()),
        schema=schema,
    )
    op.execute(revision_table.insert().values(id=1, revision=0))
