"""corpus: the audio dedup gate's fingerprint index as a table

Revision ID: corpus_0012_audio_fingerprints
Revises: corpus_0011_foreign_key_indexes
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0012_audio_fingerprints"
down_revision = "corpus_0011_foreign_key_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``audio_fingerprints``: per source, the sha256 of each episode's audio bytes, the
    episode it was transcribed for and the identities skipped as copies of it. It replaces each
    feed's ``.podcast_scraper/audio-fingerprints.json`` (the operator, 2026-11-26)."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "audio_fingerprints",
        sa.Column("source_id", sa.BigInteger(), nullable=False),
        sa.Column("digest", sa.Text(), nullable=False),
        sa.Column("identity", sa.Text(), nullable=False),
        sa.Column("item_id", sa.BigInteger(), nullable=True),
        sa.Column("episode_title", sa.Text(), nullable=True),
        sa.Column("transcript_path", sa.Text(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "duplicates",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.PrimaryKeyConstraint("source_id", "digest", name="audio_fingerprints_pkey"),
        sa.ForeignKeyConstraint(
            ["source_id"],
            [f"{schema}.sources.source_id"],
            name="audio_fingerprints_source_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            [f"{schema}.content_items.item_id"],
            name="audio_fingerprints_item_fkey",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint("digest ~ '^[0-9a-f]{64}$'", name="audio_fingerprints_digest_check"),
        schema=schema,
    )
    op.create_index("audio_fingerprints_item_idx", "audio_fingerprints", ["item_id"], schema=schema)
