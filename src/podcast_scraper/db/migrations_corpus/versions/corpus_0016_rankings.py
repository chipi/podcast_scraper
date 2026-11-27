"""corpus: person_rankings and topic_speakers, topic_relations under a projection build, and the
ABOUT edge's confidence (CRFC-005 §5)

Revision ID: corpus_0016_rankings
Revises: corpus_0015_news_recheck
"""

import sqlalchemy as sa
from alembic import context, op

revision = "corpus_0016_rankings"
down_revision = "corpus_0015_news_recheck"
branch_labels = None
depends_on = None


def _build_fk(table: str, schema: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["build_id"],
        [f"{schema}.projection_builds.build_id"],
        name=f"{table}_build_fkey",
        ondelete="CASCADE",
    )


def _entity_fk(table: str, column: str, schema: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        [column], [f"{schema}.entities.entity_id"], name=f"{table}_{column}_fkey"
    )


def upgrade() -> None:
    """Create ``person_rankings`` (persons/top: quote-backed insights per canonical person) and
    ``topic_speakers`` (a topic's perspectives, counts and rank), both under a projection build.
    Tie ``topic_relations.build_id`` to ``projection_builds`` as well: nothing writes the table
    yet, and its rows are a projection's. Add the nullable ``insight_mentions.confidence``, the
    ABOUT edge's confidence that persons/top ranks a person's topics by; the GI writers fill it."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "person_rankings",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("items", sa.Integer(), nullable=False),
        sa.Column("insights", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("build_id", "entity_id", name="person_rankings_pkey"),
        _build_fk("person_rankings", schema),
        _entity_fk("person_rankings", "entity_id", schema),
        sa.UniqueConstraint("build_id", "rank", name="person_rankings_rank_key"),
        schema=schema,
    )
    op.create_index("person_rankings_entity_idx", "person_rankings", ["entity_id"], schema=schema)
    op.create_table(
        "topic_speakers",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("topic_id", sa.BigInteger(), nullable=False),
        sa.Column("person_id", sa.BigInteger(), nullable=False),
        sa.Column("items", sa.Integer(), nullable=False),
        sa.Column("insights", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("build_id", "topic_id", "person_id", name="topic_speakers_pkey"),
        _build_fk("topic_speakers", schema),
        _entity_fk("topic_speakers", "topic_id", schema),
        _entity_fk("topic_speakers", "person_id", schema),
        sa.UniqueConstraint("build_id", "topic_id", "rank", name="topic_speakers_rank_key"),
        schema=schema,
    )
    op.create_index("topic_speakers_topic_idx", "topic_speakers", ["topic_id"], schema=schema)
    op.create_index("topic_speakers_person_idx", "topic_speakers", ["person_id"], schema=schema)
    op.create_foreign_key(
        "topic_relations_build_fkey",
        "topic_relations",
        "projection_builds",
        ["build_id"],
        ["build_id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="CASCADE",
    )
    op.add_column(
        "insight_mentions", sa.Column("confidence", sa.REAL(), nullable=True), schema=schema
    )
