"""corpus: entities, their identifiers, and the identity builds that group them

Revision ID: corpus_0005_entities
Revises: corpus_0004_documents
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "corpus_0005_entities"
down_revision = "corpus_0004_documents"
branch_labels = None
depends_on = None

_SCOPES = "'person', 'organisation', 'topic', 'object'"


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


def _jsonb(name: str) -> sa.Column:
    return sa.Column(
        name, postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
    )


def upgrade() -> None:
    """Create the entities, the identity builds, and the speaker key on ``segments``."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "entities",
        _id("entity_id"),
        sa.Column("natural_id", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("normalized_label", sa.Text(), nullable=False),
        sa.Column("canonical_id", sa.BigInteger(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=True),
        _jsonb("flags"),
        _created_at(),
        _updated_at(),
        sa.UniqueConstraint("natural_id", name="entities_natural_id_key"),
        # Deferred so a canonical row can name itself; the insert trigger fills that in.
        sa.ForeignKeyConstraint(
            ["canonical_id"],
            [f"{schema}.entities.entity_id"],
            name="entities_canonical_fkey",
            deferrable=True,
            initially="DEFERRED",
        ),
        schema=schema,
    )
    # A canonical row's own id is unknown to its inserter, so NULL means "me".
    op.execute(
        f"CREATE FUNCTION {schema}.entities_canonical_default() RETURNS trigger "
        "LANGUAGE plpgsql AS $$ BEGIN "
        "IF NEW.canonical_id IS NULL THEN NEW.canonical_id := NEW.entity_id; END IF; "
        "RETURN NEW; END $$"
    )
    op.execute(
        f"CREATE TRIGGER entities_canonical_default BEFORE INSERT ON {schema}.entities "
        f"FOR EACH ROW EXECUTE FUNCTION {schema}.entities_canonical_default()"
    )
    op.create_index("entities_canonical_idx", "entities", ["canonical_id"], schema=schema)
    op.create_index(
        "entities_type_label_idx", "entities", ["type", "normalized_label"], schema=schema
    )
    op.create_index(
        "entities_label_trgm_idx",
        "entities",
        ["normalized_label"],
        schema=schema,
        postgresql_using="gin",
        postgresql_ops={"normalized_label": "gin_trgm_ops"},
    )
    op.create_table(
        "entity_identifiers",
        sa.Column(
            "entity_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.entities.entity_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scheme", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("entity_id", "scheme", name="entity_identifiers_pkey"),
        sa.UniqueConstraint("scheme", "value", name="entity_identifiers_scheme_value_key"),
        schema=schema,
    )
    op.create_table(
        "identity_builds",
        _id("build_id"),
        sa.Column("rule_version", sa.Text(), nullable=False),
        _jsonb("inputs"),
        sa.Column(
            "built_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "state IN ('built', 'published', 'retired', 'refused')",
            name="identity_builds_state_check",
        ),
        schema=schema,
    )
    # One pointer to the served build; undo publishes the previous build again.
    op.create_index(
        "identity_builds_one_published_idx",
        "identity_builds",
        [sa.text("(true)")],
        unique=True,
        schema=schema,
        postgresql_where=sa.text("state = 'published'"),
    )
    op.create_table(
        "identity_changes",
        sa.Column(
            "decision_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "build_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.identity_builds.build_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("subject_a", sa.Text(), nullable=False),
        sa.Column("subject_b", sa.Text(), nullable=True),
        sa.Column("basis", sa.Text(), nullable=False),
        sa.Column("scorer", sa.Text(), nullable=True),
        sa.Column("scorer_version", sa.Text(), nullable=True),
        sa.Column("score", sa.REAL(), nullable=True),
        _jsonb("evidence"),
        sa.Column("decided_by", sa.Text(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'accepted'")),
        sa.CheckConstraint(f"scope IN ({_SCOPES})", name="identity_changes_scope_check"),
        sa.CheckConstraint(
            "action IN ('merge', 'reject', 'confirm')", name="identity_changes_action_check"
        ),
        sa.CheckConstraint(
            "basis IN ('rule', 'scorer', 'human')", name="identity_changes_basis_check"
        ),
        sa.CheckConstraint("status IN ('accepted')", name="identity_changes_status_check"),
        schema=schema,
    )
    op.create_index("identity_changes_build_idx", "identity_changes", ["build_id"], schema=schema)
    op.create_table(
        "identity_clusters",
        sa.Column(
            "build_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.identity_builds.build_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cluster_id", sa.BigInteger(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column(
            "canonical_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.entities.entity_id"),
            nullable=False,
        ),
        sa.Column("display_name", sa.Text(), nullable=True),
        _jsonb("display_evidence"),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("build_id", "cluster_id", name="identity_clusters_pkey"),
        sa.UniqueConstraint(
            "build_id", "cluster_id", "scope", name="identity_clusters_build_cluster_scope_key"
        ),
        sa.CheckConstraint(f"scope IN ({_SCOPES})", name="identity_clusters_scope_check"),
        sa.CheckConstraint("member_count >= 1", name="identity_clusters_member_count_check"),
        schema=schema,
    )
    op.create_index(
        "identity_clusters_canonical_idx", "identity_clusters", ["canonical_id"], schema=schema
    )
    op.create_table(
        "identity_members",
        sa.Column("build_id", sa.BigInteger(), nullable=False),
        sa.Column("cluster_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "member_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{schema}.entities.entity_id"),
            nullable=False,
        ),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column(
            "decision_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(f"{schema}.identity_changes.decision_id", ondelete="SET NULL"),
            nullable=True,
        ),
        # One written id is in one cluster per build.
        sa.PrimaryKeyConstraint("build_id", "member_id", name="identity_members_pkey"),
        # The scope is in the key, so a member is always of its cluster's scope.
        sa.ForeignKeyConstraint(
            ["build_id", "cluster_id", "scope"],
            [
                f"{schema}.identity_clusters.build_id",
                f"{schema}.identity_clusters.cluster_id",
                f"{schema}.identity_clusters.scope",
            ],
            name="identity_members_cluster_fkey",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(f"scope IN ({_SCOPES})", name="identity_members_scope_check"),
        schema=schema,
    )
    op.create_index("identity_members_member_idx", "identity_members", ["member_id"], schema=schema)
    op.create_foreign_key(
        "segments_speaker_fkey",
        "segments",
        "entities",
        ["speaker_id"],
        ["entity_id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
