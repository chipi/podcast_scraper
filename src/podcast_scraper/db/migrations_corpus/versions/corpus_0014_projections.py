"""corpus: the projections, one published build after each batch (CRFC-005 §5)

Revision ID: corpus_0014_projections
Revises: corpus_0013_groupings
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0014_projections"
down_revision = "corpus_0013_groupings"
branch_labels = None
depends_on = None

_BUILD_STATES = "'built', 'published', 'retired', 'refused'"


def _jsonb(name: str) -> sa.Column:
    return sa.Column(
        name, postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
    )


def _build_fk(table: str, schema: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["build_id"],
        [f"{schema}.projection_builds.build_id"],
        name=f"{table}_build_fkey",
        ondelete="CASCADE",
    )


def _entity_fk(table: str, schema: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["entity_id"], [f"{schema}.entities.entity_id"], name=f"{table}_entity_fkey"
    )


def upgrade() -> None:
    """Create ``projection_builds`` (its published build is the graph's ``graph_build_id``),
    ``entity_item_counts``, ``entity_weeks`` and ``cluster_reach``; drop ``clusters.reach``, which
    ``cluster_reach`` replaces so a published cluster build is never written after publishing."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "projection_builds",
        sa.Column("build_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        _jsonb("inputs"),
        _jsonb("stats"),
        sa.Column(
            "built_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"state IN ({_BUILD_STATES})", name="projection_builds_state_check"),
        schema=schema,
    )
    op.create_index(
        "projection_builds_one_published_idx",
        "projection_builds",
        [sa.text("(true)")],
        unique=True,
        postgresql_where=sa.text("state = 'published'"),
        schema=schema,
    )
    op.create_table(
        "entity_item_counts",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("items", sa.Integer(), nullable=False),
        sa.Column("insights", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("build_id", "entity_id", name="entity_item_counts_pkey"),
        _build_fk("entity_item_counts", schema),
        _entity_fk("entity_item_counts", schema),
        schema=schema,
    )
    op.create_index(
        "entity_item_counts_entity_idx", "entity_item_counts", ["entity_id"], schema=schema
    )
    op.create_table(
        "entity_weeks",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("week", sa.Date(), nullable=False),
        sa.Column("items", sa.Integer(), nullable=False),
        sa.Column("insights", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("build_id", "entity_id", "week", name="entity_weeks_pkey"),
        _build_fk("entity_weeks", schema),
        _entity_fk("entity_weeks", schema),
        sa.CheckConstraint("extract(isodow from week) = 1", name="entity_weeks_monday_check"),
        schema=schema,
    )
    op.create_index(
        "entity_weeks_kind_week_idx", "entity_weeks", ["build_id", "kind", "week"], schema=schema
    )
    op.create_index("entity_weeks_entity_idx", "entity_weeks", ["entity_id"], schema=schema)
    op.create_table(
        "cluster_reach",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("cluster_build_id", sa.BigInteger(), nullable=False),
        sa.Column("cluster_id", sa.BigInteger(), nullable=False),
        sa.Column("reach", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("build_id", "cluster_id", name="cluster_reach_pkey"),
        _build_fk("cluster_reach", schema),
        sa.ForeignKeyConstraint(
            ["cluster_build_id", "cluster_id"],
            [f"{schema}.clusters.build_id", f"{schema}.clusters.cluster_id"],
            name="cluster_reach_cluster_fkey",
            ondelete="CASCADE",
        ),
        schema=schema,
    )
    op.create_index(
        "cluster_reach_cluster_idx",
        "cluster_reach",
        ["cluster_build_id", "cluster_id"],
        schema=schema,
    )
    op.drop_column("clusters", "reach", schema=schema)
