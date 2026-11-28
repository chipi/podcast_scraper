"""corpus: the GI view's remaining insight and quote fields (r-intents, item_insights)

Revision ID: corpus_0017_insight_fields
Revises: corpus_0016_rankings
"""

import sqlalchemy as sa
from alembic import context, op

revision = "corpus_0017_insight_fields"
down_revision = "corpus_0016_rankings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add what Common's GI view serves and the rows did not hold, each nullable and filled by the
    GI writers: ``insights.confidence`` (the node's own, beside ``grounding_confidence``),
    ``insights.speaker_voice_type`` (the unattributed fallback never shows a commercial read),
    ``translated_from`` on ``insights`` and ``evidence`` (the source language of a translated
    text), and ``evidence.char_start`` / ``char_end`` (the quote's span in its document's body,
    the same span its segment range names)."""
    schema = context.config.attributes["plane"]
    op.add_column("insights", sa.Column("confidence", sa.REAL(), nullable=True), schema=schema)
    op.add_column(
        "insights", sa.Column("speaker_voice_type", sa.Text(), nullable=True), schema=schema
    )
    op.add_column("insights", sa.Column("translated_from", sa.Text(), nullable=True), schema=schema)
    op.add_column("evidence", sa.Column("translated_from", sa.Text(), nullable=True), schema=schema)
    op.add_column("evidence", sa.Column("char_start", sa.Integer(), nullable=True), schema=schema)
    op.add_column("evidence", sa.Column("char_end", sa.Integer(), nullable=True), schema=schema)
    op.create_check_constraint(
        "evidence_char_check",
        "evidence",
        "(char_start IS NULL) = (char_end IS NULL) "
        "AND (char_start IS NULL OR (char_start >= 0 AND char_end > char_start))",
        schema=schema,
    )
