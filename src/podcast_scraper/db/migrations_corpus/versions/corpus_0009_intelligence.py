"""corpus: insights, evidence and the graph's edges (CRFC-005 3.4)

Revision ID: corpus_0009_intelligence
Revises: corpus_0008_source_attrs
"""

import sqlalchemy as sa
from alembic import context, op

revision = "corpus_0009_intelligence"
down_revision = "corpus_0008_source_attrs"
branch_labels = None
depends_on = None

# gi.v4.schema.json: the Insight's ``insight_type`` enum, its ``text_origin`` const, and the
# Insight edge types the mentions keep as their ``kind``.
_INSIGHT_TYPES = "'claim', 'recommendation', 'observation', 'question', 'unknown'"
_MENTION_KINDS = "'ABOUT', 'MENTIONS_PERSON', 'MENTIONS_ORG', 'MENTIONS_OBJECT'"
# kg/speaker_coherence.py KNOWN_ROLES.
_PERSON_ROLES = "'host', 'guest', 'mentioned'"
# GI routing (AP-11): what a surface may show is "not drop, surfaceable".
_ROUTING_TAGS = "'surface', 'connect', 'drop'"

# CRFC-005 section 6: the default 0.2 waits for 32M dead rows on 160M at 100x.
_EDGE_STORAGE = {"autovacuum_vacuum_scale_factor": 0.01, "autovacuum_analyze_scale_factor": 0.005}


def upgrade() -> None:
    """Create the insights, their evidence and supports, and the entity edges of an item."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "insights",
        sa.Column("insight_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("insight_type", sa.Text(), nullable=False),
        # 0 is the first insight in the order the GI view shows: salience descending, ties in file
        # order. A reader wanting another order (cil_queries: salience, position_hint, id) has the
        # columns for it.
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("salience", sa.REAL(), nullable=True),
        sa.Column("routing_tag", sa.Text(), nullable=False),
        sa.Column("surfaceable", sa.Boolean(), nullable=True),
        sa.Column("grounded", sa.Boolean(), nullable=True),
        sa.Column("tier", sa.SmallInteger(), nullable=True),
        sa.Column("grounding_confidence", sa.REAL(), nullable=True),
        sa.Column("position_hint", sa.REAL(), nullable=True),
        sa.Column("text_origin", sa.Text(), nullable=False),
        sa.Column("visible", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("text", sa.Text(), nullable=False),
        sa.UniqueConstraint("natural_id", name="insights_natural_id_key"),
        sa.ForeignKeyConstraint(
            ["item_id", "document_id"],
            [f"{schema}.documents.item_id", f"{schema}.documents.document_id"],
            name="insights_document_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(f"insight_type IN ({_INSIGHT_TYPES})", name="insights_type_check"),
        sa.CheckConstraint("text_origin IN ('authored')", name="insights_text_origin_check"),
        sa.CheckConstraint("rank >= 0", name="insights_rank_check"),
        sa.CheckConstraint(f"routing_tag IN ({_ROUTING_TAGS})", name="insights_routing_tag_check"),
        sa.CheckConstraint("tier IS NULL OR tier BETWEEN 0 AND 3", name="insights_tier_check"),
        sa.CheckConstraint(
            "(grounding_confidence IS NULL OR grounding_confidence BETWEEN 0 AND 1) "
            "AND (position_hint IS NULL OR position_hint BETWEEN 0 AND 1)",
            name="insights_unit_check",
        ),
        schema=schema,
    )
    op.create_index("insights_item_rank_idx", "insights", ["item_id", "rank"], schema=schema)
    op.create_table(
        "evidence",
        sa.Column("evidence_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("speaker_id", sa.BigInteger(), nullable=True),
        sa.Column("start_seq", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_seq", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=True),
        sa.Column("end_ms", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.UniqueConstraint("natural_id", name="evidence_natural_id_key"),
        sa.ForeignKeyConstraint(
            ["item_id", "document_id"],
            [f"{schema}.documents.item_id", f"{schema}.documents.document_id"],
            name="evidence_document_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["speaker_id"],
            [f"{schema}.entities.entity_id"],
            name="evidence_speaker_fkey",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "start_seq >= 0 AND start_offset >= 0 AND end_offset >= 0",
            name="evidence_bounds_check",
        ),
        sa.CheckConstraint(
            "(end_seq, end_offset) > (start_seq, start_offset)", name="evidence_span_check"
        ),
        sa.CheckConstraint(
            "(start_ms IS NULL) = (end_ms IS NULL) "
            "AND (start_ms IS NULL OR (start_ms >= 0 AND end_ms >= start_ms))",
            name="evidence_ms_check",
        ),
        schema=schema,
        postgresql_with=_EDGE_STORAGE,
    )
    op.create_index("evidence_item_idx", "evidence", ["item_id"], schema=schema)
    op.create_index(
        "evidence_speaker_idx",
        "evidence",
        ["speaker_id", "item_id"],
        schema=schema,
        postgresql_where=sa.text("speaker_id IS NOT NULL"),
    )
    op.create_table(
        "insight_support",
        sa.Column("insight_id", sa.BigInteger(), nullable=False),
        sa.Column("evidence_id", sa.BigInteger(), nullable=False),
        sa.Column("ord", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("insight_id", "evidence_id", name="insight_support_pkey"),
        sa.ForeignKeyConstraint(
            ["insight_id"],
            [f"{schema}.insights.insight_id"],
            name="insight_support_insight_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["evidence_id"],
            [f"{schema}.evidence.evidence_id"],
            name="insight_support_evidence_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("ord >= 0", name="insight_support_ord_check"),
        schema=schema,
        postgresql_with=_EDGE_STORAGE,
    )
    op.create_index(
        "insight_support_evidence_idx", "insight_support", ["evidence_id"], schema=schema
    )
    op.create_table(
        "insight_mentions",
        sa.Column("insight_id", sa.BigInteger(), nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("insight_id", "kind", "entity_id", name="insight_mentions_pkey"),
        sa.ForeignKeyConstraint(
            ["insight_id"],
            [f"{schema}.insights.insight_id"],
            name="insight_mentions_insight_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"], [f"{schema}.entities.entity_id"], name="insight_mentions_entity_fkey"
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            [f"{schema}.content_items.item_id"],
            name="insight_mentions_item_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(f"kind IN ({_MENTION_KINDS})", name="insight_mentions_kind_check"),
        sa.CheckConstraint("rank >= 0", name="insight_mentions_rank_check"),
        schema=schema,
        postgresql_with=_EDGE_STORAGE,
    )
    # A topic's top insights are one range of this index; ``item_id`` rides along for the page.
    op.create_index(
        "insight_mentions_entity_rank_idx",
        "insight_mentions",
        ["entity_id", "kind", "rank", "insight_id"],
        schema=schema,
        postgresql_include=["item_id"],
    )
    op.create_index("insight_mentions_item_idx", "insight_mentions", ["item_id"], schema=schema)
    op.create_table(
        "insight_attribution",
        sa.Column("insight_id", sa.BigInteger(), nullable=False),
        sa.Column("person_id", sa.BigInteger(), nullable=False),
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("insight_id", "person_id", name="insight_attribution_pkey"),
        sa.ForeignKeyConstraint(
            ["insight_id"],
            [f"{schema}.insights.insight_id"],
            name="insight_attribution_insight_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            [f"{schema}.entities.entity_id"],
            name="insight_attribution_person_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            [f"{schema}.content_items.item_id"],
            name="insight_attribution_item_fkey",
            ondelete="CASCADE",
        ),
        schema=schema,
        postgresql_with=_EDGE_STORAGE,
    )
    op.create_index(
        "insight_attribution_person_idx",
        "insight_attribution",
        ["person_id", "item_id"],
        schema=schema,
    )
    op.create_table(
        "item_entities",
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("item_id", "entity_id", "role", name="item_entities_pkey"),
        sa.ForeignKeyConstraint(
            ["item_id"],
            [f"{schema}.content_items.item_id"],
            name="item_entities_item_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"], [f"{schema}.entities.entity_id"], name="item_entities_entity_fkey"
        ),
        schema=schema,
        postgresql_with=_EDGE_STORAGE,
    )
    op.create_index(
        "item_entities_entity_published_idx",
        "item_entities",
        ["entity_id", sa.text("published_at DESC"), "item_id"],
        schema=schema,
    )
    op.create_index(
        "item_entities_entity_type_published_idx",
        "item_entities",
        ["entity_id", "content_type", sa.text("published_at DESC"), "item_id"],
        schema=schema,
    )
    op.create_table(
        "person_roles",
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("person_id", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("item_id", "person_id", "role", name="person_roles_pkey"),
        sa.ForeignKeyConstraint(
            ["item_id"],
            [f"{schema}.content_items.item_id"],
            name="person_roles_item_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["person_id"], [f"{schema}.entities.entity_id"], name="person_roles_person_fkey"
        ),
        sa.CheckConstraint(f"role IN ({_PERSON_ROLES})", name="person_roles_role_check"),
        schema=schema,
    )
    op.create_index(
        "person_roles_person_idx", "person_roles", ["person_id", "item_id"], schema=schema
    )
    # No build table exists until the clusters revision, so ``build_id`` is a plain number here.
    op.create_table(
        "topic_relations",
        sa.Column("topic_a", sa.BigInteger(), nullable=False),
        sa.Column("topic_b", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("weight", sa.REAL(), nullable=False),
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint(
            "build_id", "topic_a", "topic_b", "kind", name="topic_relations_pkey"
        ),
        sa.ForeignKeyConstraint(
            ["topic_a"], [f"{schema}.entities.entity_id"], name="topic_relations_topic_a_fkey"
        ),
        sa.ForeignKeyConstraint(
            ["topic_b"], [f"{schema}.entities.entity_id"], name="topic_relations_topic_b_fkey"
        ),
        sa.CheckConstraint("topic_a <> topic_b", name="topic_relations_distinct_check"),
        schema=schema,
    )
    op.create_index(
        "topic_relations_a_idx",
        "topic_relations",
        ["topic_a", "kind", sa.text("weight DESC"), "topic_b"],
        schema=schema,
    )
    op.create_index(
        "topic_relations_b_idx",
        "topic_relations",
        ["topic_b", "kind", sa.text("weight DESC"), "topic_a"],
        schema=schema,
    )
