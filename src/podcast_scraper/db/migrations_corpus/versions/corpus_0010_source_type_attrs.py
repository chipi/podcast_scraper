"""corpus: a source's feed fields the catalog row carries and no column holds

Revision ID: corpus_0010_source_type_attrs
Revises: corpus_0009_intelligence
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0010_source_type_attrs"
down_revision = "corpus_0009_intelligence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add ``sources.type_attrs``: the feed fields a ``CatalogEpisodeRow`` reads (the feed's RSS
    URL, description, authors, last update and image) that no ``sources`` column holds, so a
    catalog row built from rows equals the one built from files."""
    schema = context.config.attributes["plane"]
    op.add_column(
        "sources",
        sa.Column(
            "type_attrs",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        schema=schema,
    )
