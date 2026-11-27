"""corpus: groupings and enrichment outputs (CRFC-005 §3.7)

Revision ID: corpus_0013_groupings
Revises: corpus_0012_audio_fingerprints
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0013_groupings"
down_revision = "corpus_0012_audio_fingerprints"
branch_labels = None
depends_on = None

_GROUP_KINDS = "'theme', 'storyline', 'super_theme', 'insight'"
_BUILD_STATES = "'built', 'published', 'retired', 'refused'"


def _jsonb(name: str) -> sa.Column:
    return sa.Column(
        name, postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
    )


def _tz(name: str, *, now: bool = False) -> sa.Column:
    if now:
        return sa.Column(
            name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        )
    return sa.Column(name, sa.DateTime(timezone=True), nullable=True)


def _builds(schema: str) -> None:
    op.create_table(
        "cluster_builds",
        sa.Column("build_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("rule_version", sa.Text(), nullable=False),
        _jsonb("inputs"),
        sa.Column("min_link_weight", sa.Integer(), nullable=True),
        _jsonb("registry_state"),
        _jsonb("stats"),
        _tz("built_at", now=True),
        sa.Column("state", sa.Text(), nullable=False),
        _tz("published_at"),
        sa.CheckConstraint(f"state IN ({_BUILD_STATES})", name="cluster_builds_state_check"),
        schema=schema,
    )
    op.create_index(
        "cluster_builds_one_published_idx",
        "cluster_builds",
        [sa.text("(true)")],
        unique=True,
        postgresql_where=sa.text("state = 'published'"),
        schema=schema,
    )
    op.create_table(
        "enrichment_builds",
        sa.Column("build_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("enricher_id", sa.Text(), nullable=False),
        sa.Column("enricher_version", sa.Text(), nullable=True),
        _tz("computed_at"),
        _tz("built_at", now=True),
        sa.Column("state", sa.Text(), nullable=False),
        _tz("published_at"),
        sa.CheckConstraint(f"state IN ({_BUILD_STATES})", name="enrichment_builds_state_check"),
        schema=schema,
    )
    op.create_index(
        "enrichment_builds_one_published_idx",
        "enrichment_builds",
        ["enricher_id"],
        unique=True,
        postgresql_where=sa.text("state = 'published'"),
        schema=schema,
    )


def _groups(schema: str) -> None:
    op.create_table(
        "cluster_ids",
        sa.Column("cluster_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.UniqueConstraint("natural_id", name="cluster_ids_natural_id_key"),
        sa.CheckConstraint(f"kind IN ({_GROUP_KINDS})", name="cluster_ids_kind_check"),
        schema=schema,
    )
    op.create_table(
        "clusters",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("cluster_id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("parent_id", sa.BigInteger(), nullable=True),
        sa.Column("member_count", sa.Integer(), nullable=True),
        sa.Column("anchor_entity_id", sa.BigInteger(), nullable=True),
        sa.Column("reach", sa.Integer(), nullable=True),
        sa.Column("state", sa.Text(), nullable=True),
        sa.Column(
            "aliases",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        _tz("created_at"),
        _tz("last_active_at"),
        _jsonb("detail"),
        sa.PrimaryKeyConstraint("build_id", "cluster_id", name="clusters_pkey"),
        sa.ForeignKeyConstraint(
            ["build_id"],
            [f"{schema}.cluster_builds.build_id"],
            name="clusters_build_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["cluster_id"], [f"{schema}.cluster_ids.cluster_id"], name="clusters_cluster_fkey"
        ),
        sa.ForeignKeyConstraint(
            ["build_id", "parent_id"],
            [f"{schema}.clusters.build_id", f"{schema}.clusters.cluster_id"],
            name="clusters_parent_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["anchor_entity_id"],
            [f"{schema}.entities.entity_id"],
            name="clusters_anchor_entity_fkey",
        ),
        sa.CheckConstraint(f"kind IN ({_GROUP_KINDS})", name="clusters_kind_check"),
        sa.CheckConstraint(
            "state IS NULL OR state IN ('active', 'quiet')", name="clusters_state_check"
        ),
        schema=schema,
    )
    for name, columns in (
        ("clusters_cluster_idx", ["cluster_id"]),
        ("clusters_build_kind_idx", ["build_id", "kind"]),
        ("clusters_parent_idx", ["build_id", "parent_id"]),
        ("clusters_anchor_entity_idx", ["anchor_entity_id"]),
    ):
        op.create_index(name, "clusters", columns, schema=schema)
    op.create_table(
        "cluster_members",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("cluster_id", sa.BigInteger(), nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=True),
        sa.Column("insight_id", sa.BigInteger(), nullable=True),
        sa.Column("strength", sa.REAL(), nullable=True),
        sa.Column("lift_to_cluster", sa.REAL(), nullable=True),
        sa.Column("is_anchor", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.ForeignKeyConstraint(
            ["build_id", "cluster_id"],
            [f"{schema}.clusters.build_id", f"{schema}.clusters.cluster_id"],
            name="cluster_members_cluster_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"], [f"{schema}.entities.entity_id"], name="cluster_members_entity_fkey"
        ),
        sa.ForeignKeyConstraint(
            ["insight_id"],
            [f"{schema}.insights.insight_id"],
            name="cluster_members_insight_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "(entity_id IS NULL) <> (insight_id IS NULL)",
            name="cluster_members_one_member_check",
        ),
        schema=schema,
    )
    for name, columns in (
        ("cluster_members_cluster_idx", ["build_id", "cluster_id"]),
        ("cluster_members_entity_idx", ["entity_id"]),
        ("cluster_members_insight_idx", ["insight_id"]),
    ):
        op.create_index(name, "cluster_members", columns, schema=schema)
    for member in ("entity_id", "insight_id"):
        op.create_index(
            f"cluster_members_{member.split('_')[0]}_key",
            "cluster_members",
            ["build_id", "cluster_id", member],
            unique=True,
            postgresql_where=sa.text(f"{member} IS NOT NULL"),
            schema=schema,
        )


def _history(schema: str) -> None:
    op.create_table(
        "cluster_id_history",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("old_natural_id", sa.Text(), nullable=False),
        sa.Column("new_natural_id", sa.Text(), nullable=True),
        sa.Column("change", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("build_id", "old_natural_id", name="cluster_id_history_pkey"),
        sa.ForeignKeyConstraint(
            ["build_id"],
            [f"{schema}.cluster_builds.build_id"],
            name="cluster_id_history_build_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "(change = 'successor') = (new_natural_id IS NOT NULL) "
            "AND change IN ('successor', 'gone')",
            name="cluster_id_history_change_check",
        ),
        schema=schema,
    )
    op.create_table(
        "cluster_events",
        sa.Column("event_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("changes", postgresql.JSONB(), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        schema=schema,
    )
    op.create_index("cluster_events_kind_at_idx", "cluster_events", ["kind", "at"], schema=schema)
    op.create_table(
        "cluster_links",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("a", sa.BigInteger(), nullable=False),
        sa.Column("b", sa.BigInteger(), nullable=False),
        sa.Column("weight", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("build_id", "kind", "a", "b", name="cluster_links_pkey"),
        sa.ForeignKeyConstraint(
            ["build_id", "a"],
            [f"{schema}.clusters.build_id", f"{schema}.clusters.cluster_id"],
            name="cluster_links_a_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["build_id", "b"],
            [f"{schema}.clusters.build_id", f"{schema}.clusters.cluster_id"],
            name="cluster_links_b_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("a < b", name="cluster_links_order_check"),
        schema=schema,
    )
    op.create_index("cluster_links_a_idx", "cluster_links", ["build_id", "a"], schema=schema)
    op.create_index("cluster_links_b_idx", "cluster_links", ["build_id", "b"], schema=schema)


def _enrichment(schema: str) -> None:
    op.create_table(
        "enrichment_outputs",
        sa.Column("output_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("enricher_id", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("item_id", sa.BigInteger(), nullable=True),
        sa.Column("entity_id", sa.BigInteger(), nullable=True),
        sa.Column("output", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["build_id"],
            [f"{schema}.enrichment_builds.build_id"],
            name="enrichment_outputs_build_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            [f"{schema}.content_items.item_id"],
            name="enrichment_outputs_item_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"], [f"{schema}.entities.entity_id"], name="enrichment_outputs_entity_fkey"
        ),
        sa.UniqueConstraint(
            "build_id",
            "scope",
            "item_id",
            "entity_id",
            name="enrichment_outputs_key",
            postgresql_nulls_not_distinct=True,
        ),
        schema=schema,
    )
    for name, columns in (
        ("enrichment_outputs_lookup_idx", ["enricher_id", "scope", "item_id"]),
        ("enrichment_outputs_item_idx", ["item_id"]),
        ("enrichment_outputs_entity_idx", ["entity_id"]),
    ):
        op.create_index(name, "enrichment_outputs", columns, schema=schema)
    op.create_table(
        "entity_pairs",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("enricher_id", sa.Text(), nullable=False),
        sa.Column("a", sa.BigInteger(), nullable=False),
        sa.Column("b", sa.BigInteger(), nullable=False),
        sa.Column("score", sa.REAL(), nullable=False),
        _jsonb("detail"),
        sa.PrimaryKeyConstraint("build_id", "a", "b", name="entity_pairs_pkey"),
        sa.ForeignKeyConstraint(
            ["build_id"],
            [f"{schema}.enrichment_builds.build_id"],
            name="entity_pairs_build_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["a"], [f"{schema}.entities.entity_id"], name="entity_pairs_a_fkey"
        ),
        sa.ForeignKeyConstraint(
            ["b"], [f"{schema}.entities.entity_id"], name="entity_pairs_b_fkey"
        ),
        sa.CheckConstraint("a <> b", name="entity_pairs_distinct_check"),
        schema=schema,
    )
    op.create_index("entity_pairs_a_idx", "entity_pairs", ["a", "enricher_id"], schema=schema)
    op.create_index("entity_pairs_b_idx", "entity_pairs", ["b", "enricher_id"], schema=schema)


def upgrade() -> None:
    """Create the clustering build's tables (``cluster_builds``, durable ``cluster_ids``,
    ``clusters``, ``cluster_members``, ``cluster_id_history``, ``cluster_events``,
    ``cluster_links``) and the enrichers' (``enrichment_builds``, ``enrichment_outputs``,
    ``entity_pairs``), each foreign key under an index that starts with its columns."""
    schema = context.config.attributes["plane"]
    _builds(schema)
    _groups(schema)
    _history(schema)
    _enrichment(schema)
