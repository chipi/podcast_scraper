"""corpus: the source columns each item states, so a source row is derived from its items

Revision ID: corpus_0008_source_attrs
Revises: corpus_0007_search_chunks
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0008_source_attrs"
down_revision = "corpus_0007_search_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add ``content_items.source_attrs``: the ``sources`` columns the item's own metadata states
    (``corpus_rows.source_row`` without the source's keys). A source row is the merge of its
    served items' ``source_attrs`` in ``corpus_rows.newest_key`` order, whatever order the items
    were written in, and follows an item that stops being served."""
    schema = context.config.attributes["plane"]
    op.add_column(
        "content_items",
        sa.Column(
            "source_attrs",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        schema=schema,
    )
