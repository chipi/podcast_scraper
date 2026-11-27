"""Kernel tables (ADR-169): defined WITHOUT a schema, mapped onto the plane at connection time.

Never add ``schema=`` to a ``metadata`` table: the engine's ``schema_translate_map={None: plane}``
and Alembic's per-plane run depend on every kernel table having none. The exceptions are the
``shared`` and ``corpus`` schemas at the end, each with its own metadata and its own history.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    func,
    Identity,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    REAL,
    SmallInteger,
    Table,
    Text,
    text,
    true,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("user_id", Text, primary_key=True),
    Column("email", Text, nullable=False, server_default=text("''")),
    Column("name", Text, nullable=False, server_default=text("''")),
    # NULL, not '': profiles older than the handle have none, and UNIQUE must not collide them.
    Column("username", Text, nullable=True),
    Column("image", Text, nullable=True),
    Column("provider", Text, nullable=False),
    Column("subject", Text, nullable=False),
    Column("disabled", Boolean, nullable=False, server_default=text("false")),
    Column("role", Text, nullable=False, server_default=text("'listener'")),
    Column("mcp_access", Boolean, nullable=False, server_default=text("true")),
    # NULL until backfilled at the next sign-in; UNIQUE, so one pseudonym is one account.
    Column("analytics_id", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("username", name="users_username_key"),
    UniqueConstraint("analytics_id", name="users_analytics_id_key"),
    CheckConstraint("role IN ('listener', 'creator', 'admin')", name="users_role_check"),
    UniqueConstraint("provider", "subject", name="users_provider_subject_key"),
)

# One row per revision a ``db upgrade`` applied; written by ``migrate.upgrade``, never by the app.
schema_history = Table(
    "schema_history",
    metadata,
    Column("id", Integer, Identity(), primary_key=True),
    Column("revision", Text, nullable=False),
    Column("applied_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("release", Text, nullable=True),
    # Which variable ``release`` came from: CL_RELEASE (the deploy) or PODCAST_RELEASE (image ref).
    Column("release_source", Text, nullable=True),
    Column("duration_ms", Integer, nullable=True),
)


def _user_fk() -> Column[str]:
    """``user_id`` naming an account; the row goes with the account (one cascade, #2273)."""
    return Column(
        "user_id",
        Text,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )


def _updated_at() -> Column[datetime]:
    return Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now())


# Who may sign in (#2190). At most one row; none means the env policy applies.
access_policy = Table(
    "access_policy",
    metadata,
    Column("id", SmallInteger, primary_key=True, server_default=text("1")),
    Column("mode", Text, nullable=False),
    Column("allowed_emails", ARRAY(Text), nullable=False, server_default=text("'{}'")),
    Column("allowed_domains", ARRAY(Text), nullable=False, server_default=text("'{}'")),
    _updated_at(),
    CheckConstraint("id = 1", name="access_policy_single_row"),
    CheckConstraint("mode IN ('allowlist', 'open')", name="access_policy_mode_check"),
)

# The consent matrix and schedules, as the store merges them onto its defaults (#1414). The
# unsubscribe handle is its own column so the one-click link is one indexed lookup.
comms = Table(
    "comms",
    metadata,
    _user_fk(),
    Column("settings", JSONB, nullable=False),
    Column("unsubscribe_ref", Text, nullable=True),
    _updated_at(),
    UniqueConstraint("unsubscribe_ref", name="comms_unsubscribe_ref_key"),
)

# Web-Push subscriptions, one per device (#1415). ``seq`` keeps the order the list had: a
# re-subscribe moves the endpoint to the end.
push_subscriptions = Table(
    "push_subscriptions",
    metadata,
    Column(
        "user_id",
        Text,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("endpoint", Text, primary_key=True),
    Column("subscription", JSONB, nullable=False),
    Column("seq", BigInteger, Identity(), nullable=False),
)

# The client's free-form preferences object (USERPREFS-1); the server never interprets it.
user_preferences = Table(
    "user_preferences",
    metadata,
    _user_fk(),
    Column("payload", JSONB, nullable=False),
    _updated_at(),
)

# Sign in with Apple's refresh token, kept only so deletion can revoke it.
apple_tokens = Table(
    "apple_tokens",
    metadata,
    _user_fk(),
    Column("refresh_token", Text, nullable=False),
    _updated_at(),
)

# Graph usage events (append-only). The account's events go with the account, as their file did;
# NULL is a signed-out session (the route's ``anon`` bucket, which is not an account).
graph_events = Table(
    "graph_events",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column(
        "user_id",
        Text,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("event", JSONB, nullable=False),
    Index("graph_events_user_idx", "user_id", "id"),
)

# The delivery outbox (#1415, RFC-110 §2). No foreign key: a sign-in envelope is written before its
# account exists. ``user_id`` and ``recipient_email`` are copied out of the envelope so account
# deletion can match them.
outbox = Table(
    "outbox",
    metadata,
    Column("envelope_id", Text, primary_key=True),
    Column("envelope", JSONB, nullable=False),
    Column("channel", Text, nullable=True),
    Column("user_id", Text, nullable=True),
    Column("recipient_email", Text, nullable=True),
    Column("status", Text, nullable=False),
    Column("detail", Text, nullable=True),
    _updated_at(),
    CheckConstraint(
        "status IN ('pending', 'delivered', 'bounced', 'complaint', 'suppressed', 'failed')",
        name="outbox_status_check",
    ),
    Index(
        "outbox_pending_idx",
        "channel",
        "updated_at",
        postgresql_where=text("status = 'pending'"),
    ),
    Index("outbox_user_idx", "user_id"),
    Index("outbox_recipient_idx", "recipient_email"),
)

# Consumed magic-link token ids: each link signs in once (app_magic_link).
magic_link_used = Table(
    "magic_link_used",
    metadata,
    Column("token_id", Text, primary_key=True),
    Column("used_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# When a link was last sent to an address, keyed by a hash of it, never the address.
magic_link_requests = Table(
    "magic_link_requests",
    metadata,
    Column("email_hash", Text, primary_key=True),
    Column("requested_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# Operator and security events (#1071). No foreign key: the trail outlives the accounts it names.
audit_log = Table(
    "audit_log",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("ts", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("record", JSONB, nullable=False),
)


#: The ``shared`` schema: the few rows both planes read and write (the operator, 2026-11-05). Its
#: tables name their schema, so the planes' ``schema_translate_map`` leaves them alone.
SHARED_SCHEMA = "shared"
shared_metadata = MetaData(schema=SHARED_SCHEMA)

# Each client app's released version, set from the operator viewer and read by the phones through
# the player api (#2296: one value, not one per plane).
app_releases = Table(
    "app_releases",
    shared_metadata,
    Column("app", Text, primary_key=True),
    Column("version", Text, nullable=False),
    _updated_at(),
    CheckConstraint("app ~ '^[a-z][a-z0-9_]{0,31}$'", name="app_releases_app_check"),
    CheckConstraint(
        "version ~ '^[0-9]{1,4}(\\.[0-9]{1,4}){0,3}$'", name="app_releases_version_check"
    ),
)


#: The ``corpus`` schema: the platform's content tables, read by every plane and the pipeline
#: (ADR-184). Its tables name their schema, like ``shared``'s. ``schema_history`` is the kernel's
#: table definition, written there by ``migrate``.
CORPUS_SCHEMA = "corpus"
corpus_metadata = MetaData(schema=CORPUS_SCHEMA)


def _created_at() -> Column[datetime]:
    return Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now())


def _surrogate(name: str) -> Column[int]:
    return Column(name, BigInteger, Identity(), primary_key=True)


# Which migration ran, its dry run included, and the rows it changed: the undo reads the receipts.
corpus_migrations = Table(
    "corpus_migrations",
    corpus_metadata,
    _surrogate("migration_id"),
    Column("natural_id", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    _created_at(),
    Column("to_version", Text, nullable=True),
    Column("last_run_id", BigInteger, ForeignKey("migration_runs.run_id"), nullable=True),
    UniqueConstraint("natural_id", name="corpus_migrations_natural_id_key"),
    # 'dry_run' is allowed and unused: a dry run is a migration_runs row (RFC-135 §2).
    CheckConstraint(
        "status IN ('dry_run', 'applied', 'undone')", name="corpus_migrations_status_check"
    ),
    Index("corpus_migrations_last_run_idx", "last_run_id"),
)

# Every attempt of a migration on rows: dry run, apply, undo, rerun (RFC-135 §2).
migration_runs = Table(
    "migration_runs",
    corpus_metadata,
    _surrogate("run_id"),
    Column("natural_id", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("code_digest", Text, nullable=True),
    Column("outcome_sha256", Text, nullable=True),
    Column("plan", JSONB, nullable=True),
    Column("result", JSONB, nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    CheckConstraint(
        "kind IN ('dry_run', 'apply', 'undo', 'rerun')", name="migration_runs_kind_check"
    ),
    CheckConstraint("status IN ('ok', 'refused', 'failed')", name="migration_runs_status_check"),
    Index("migration_runs_natural_idx", "natural_id", "run_id"),
)

migration_receipts = Table(
    "migration_receipts",
    corpus_metadata,
    _surrogate("receipt_id"),
    Column(
        "migration_id", BigInteger, ForeignKey("corpus_migrations.migration_id"), nullable=False
    ),
    Column("target_table", Text, nullable=False),
    Column("target_key", Text, nullable=False),
    Column("before", JSONB, nullable=True),
    Column("after", JSONB, nullable=True),
    _created_at(),
    Column("run_id", BigInteger, ForeignKey("migration_runs.run_id"), nullable=True),
    Column("seq", Integer, nullable=True),
    Index("migration_receipts_migration_idx", "migration_id"),
    Index("migration_receipts_run_idx", "run_id", "seq"),
)

corpus_meta = Table(
    "corpus_meta",
    corpus_metadata,
    Column("key", Text, primary_key=True),
    Column("value", JSONB, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# ``source_id``'s foreign key is added in corpus_0003, once ``sources`` exists.
ingestion_runs = Table(
    "ingestion_runs",
    corpus_metadata,
    _surrogate("run_id"),
    Column("natural_id", Text, nullable=False),
    Column(
        "source_id",
        BigInteger,
        ForeignKey("sources.source_id", name="ingestion_runs_source_id_fkey"),
        nullable=True,
    ),
    Column("status", Text, nullable=False, server_default=text("'running'")),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    _created_at(),
    UniqueConstraint("natural_id", name="ingestion_runs_natural_id_key"),
    Index("ingestion_runs_created_idx", text("created_at DESC")),
    Index("ingestion_runs_source_idx", "source_id"),
)

run_item_outcomes = Table(
    "run_item_outcomes",
    corpus_metadata,
    _surrogate("outcome_id"),
    Column(
        "run_id",
        BigInteger,
        ForeignKey("ingestion_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "item_id",
        BigInteger,
        ForeignKey(
            "content_items.item_id", name="run_item_outcomes_item_id_fkey", ondelete="SET NULL"
        ),
        nullable=True,
    ),
    Column("outcome", Text, nullable=False),
    Column("detail", JSONB, nullable=True),
    _created_at(),
    Index("run_item_outcomes_run_idx", "run_id"),
    Index("run_item_outcomes_item_idx", "item_id"),
)

# What produced a row (stage, method version, model, prompt hash, git sha, run).
processing_stages = Table(
    "processing_stages",
    corpus_metadata,
    _surrogate("processing_stage_id"),
    Column("stage", Text, nullable=False),
    Column("method_version", Text, nullable=True),
    Column("model", Text, nullable=True),
    Column("prompt_hash", Text, nullable=True),
    Column("git_sha", Text, nullable=True),
    Column("run_id", BigInteger, ForeignKey("ingestion_runs.run_id"), nullable=True),
    _created_at(),
    Index("processing_stages_run_idx", "run_id"),
)

enrichment_runs = Table(
    "enrichment_runs",
    corpus_metadata,
    _surrogate("enrichment_run_id"),
    Column("natural_id", Text, nullable=False),
    Column("enricher_id", Text, nullable=False),
    Column("status", Text, nullable=False, server_default=text("'running'")),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    _created_at(),
    UniqueConstraint("natural_id", name="enrichment_runs_natural_id_key"),
    Index("enrichment_runs_created_idx", text("created_at DESC")),
)

# The job's cursor lives on its row, so a restarted job resumes from it.
jobs = Table(
    "jobs",
    corpus_metadata,
    _surrogate("job_id"),
    Column("natural_id", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("status", Text, nullable=False, server_default=text("'queued'")),
    Column("cursor", JSONB, nullable=True),
    _created_at(),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("natural_id", name="jobs_natural_id_key"),
    Index("jobs_created_idx", text("created_at DESC")),
)

# A writer re-checks its lease in every chunk: ``holder = :me AND expires_at > now()``.
job_leases = Table(
    "job_leases",
    corpus_metadata,
    _surrogate("lease_id"),
    Column("resource", Text, nullable=False),
    Column("mode", Text, nullable=False),
    Column("holder", Text, nullable=False),
    Column("job_id", BigInteger, ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=True),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("heartbeat_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    _created_at(),
    UniqueConstraint("resource", "holder", name="job_leases_resource_holder_key"),
    CheckConstraint("mode IN ('exclusive', 'shared')", name="job_leases_mode_check"),
    Index("job_leases_expires_idx", "expires_at"),
    Index("job_leases_job_idx", "job_id"),
)

assets = Table(
    "assets",
    corpus_metadata,
    _surrogate("asset_id"),
    Column("owner_kind", Text, nullable=False),
    Column("owner_id", BigInteger, nullable=False),
    Column("role", Text, nullable=False),
    Column("path", Text, nullable=False),
    Column("content_type", Text, nullable=False),
    _created_at(),
    UniqueConstraint("owner_kind", "owner_id", "role", name="assets_owner_role_key"),
)

sources = Table(
    "sources",
    corpus_metadata,
    _surrogate("source_id"),
    Column("natural_id", Text, nullable=False),
    Column("raw_feed_id", Text, nullable=True),
    Column("content_type", Text, nullable=False),
    Column("title", Text, nullable=False),
    # No foreign key until the entities table exists.
    Column("publisher_id", BigInteger, nullable=True),
    Column("language", Text, nullable=True),
    Column("category", Text, nullable=True),
    Column("licence", Text, nullable=True),
    Column(
        "artwork_asset_id",
        BigInteger,
        ForeignKey("assets.asset_id", ondelete="SET NULL"),
        nullable=True,
    ),
    _created_at(),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    # corpus_0010: the feed fields the catalog row reads that no column holds.
    Column("type_attrs", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    UniqueConstraint("natural_id", name="sources_natural_id_key"),
    UniqueConstraint("raw_feed_id", name="sources_raw_feed_id_key"),
    Index("sources_artwork_idx", "artwork_asset_id"),
)


def _flag(name: str) -> Column[bool]:
    return Column(name, Boolean, nullable=False, server_default=text("false"))


def _flags() -> list[Column[bool]]:
    return [
        _flag(n)
        for n in ("has_transcript", "has_gi", "has_kg", "has_moments", "withheld", "unusable")
    ]


# ``served`` (0 not served, 1 served, 2 served and in the newest run) replaces the read-time rule
# for the newest copy of an item; a superseded copy is not a row.
content_items = Table(
    "content_items",
    corpus_metadata,
    _surrogate("item_id"),
    Column("natural_id", Text, nullable=False),
    Column("slug", Text, nullable=False),
    Column("source_id", BigInteger, ForeignKey("sources.source_id"), nullable=False),
    Column("external_item_id", Text, nullable=False),
    Column("content_type", Text, nullable=False),
    Column("editorial_type", Text, nullable=True),
    Column("title", Text, nullable=False),
    # Each time is nullable: unknown stays null, never a guess.
    Column("published_at", DateTime(timezone=True), nullable=True),
    Column("filed_at", DateTime(timezone=True), nullable=True),
    Column("period_start", DateTime(timezone=True), nullable=True),
    Column("period_end", DateTime(timezone=True), nullable=True),
    Column("observed_at", DateTime(timezone=True), nullable=True),
    Column("ingested_at", DateTime(timezone=True), nullable=True),
    Column("duration_s", Integer, nullable=True),
    Column("language", Text, nullable=True),
    Column("served", SmallInteger, nullable=False, server_default=text("0")),
    *_flags(),
    Column("type_attrs", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    _created_at(),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    # corpus_0008: the source columns this item states; the source row is merged from them.
    Column("source_attrs", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    UniqueConstraint("natural_id", name="content_items_natural_id_key"),
    UniqueConstraint("slug", name="content_items_slug_key"),
    UniqueConstraint("source_id", "external_item_id", name="content_items_source_external_key"),
    CheckConstraint("served IN (0, 1, 2)", name="content_items_served_check"),
    Index(
        "content_items_published_idx",
        text("published_at DESC"),
        "item_id",
        postgresql_where=text("served > 0"),
    ),
    Index(
        "content_items_source_published_idx",
        "source_id",
        text("published_at DESC"),
        postgresql_where=text("served > 0"),
    ),
    Index(
        "content_items_type_published_idx",
        "content_type",
        text("published_at DESC"),
        postgresql_where=text("served > 0"),
    ),
    Index(
        "content_items_title_trgm_idx",
        "title",
        postgresql_using="gin",
        postgresql_ops={"title": "gin_trgm_ops"},
    ),
)

# One row per item, written in the item's transaction: the card payload a list reads.
item_cards = Table(
    "item_cards",
    corpus_metadata,
    Column(
        "item_id",
        BigInteger,
        ForeignKey("content_items.item_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("title", Text, nullable=False),
    Column("source_title", Text, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=True),
    Column("duration_s", Integer, nullable=True),
    Column(
        "artwork_asset_id",
        BigInteger,
        ForeignKey("assets.asset_id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("lede", Text, nullable=True),
    Column("summary_bullets", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("top_topics", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    *_flags(),
    _created_at(),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("item_cards_artwork_idx", "artwork_asset_id"),
)

# A document is one version of one kind and variant of an item; the served one is the one no row
# supersedes.
documents = Table(
    "documents",
    corpus_metadata,
    _surrogate("document_id"),
    Column(
        "item_id",
        BigInteger,
        ForeignKey("content_items.item_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("kind", Text, nullable=False),
    Column("variant", Text, nullable=False),
    Column("version", Integer, nullable=False, server_default=text("1")),
    Column("supersedes_id", BigInteger, nullable=True),
    Column("change_kind", Text, nullable=True),
    Column("body_sha256", Text, nullable=True),
    Column("fetched_at", DateTime(timezone=True), nullable=True),
    Column("extraction", Text, nullable=True),
    Column("extractor_version", Text, nullable=True),
    _created_at(),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint(
        "item_id", "kind", "variant", "version", name="documents_item_kind_variant_version_key"
    ),
    UniqueConstraint("supersedes_id", name="documents_supersedes_key"),
    UniqueConstraint("item_id", "document_id", name="documents_item_document_key"),
    UniqueConstraint(
        "item_id", "kind", "variant", "document_id", name="documents_series_document_key"
    ),
    ForeignKeyConstraint(
        ["item_id", "kind", "variant", "supersedes_id"],
        [
            "documents.item_id",
            "documents.kind",
            "documents.variant",
            "documents.document_id",
        ],
        name="documents_supersedes_fkey",
    ),
    CheckConstraint("kind IN ('transcript', 'article')", name="documents_kind_check"),
    CheckConstraint(
        "variant IN ('raw', 'cleaned', 'adfree', 'en')", name="documents_variant_check"
    ),
    CheckConstraint("version >= 1", name="documents_version_check"),
    Index("documents_supersedes_idx", "item_id", "kind", "variant", "supersedes_id"),
)

_SCOPES = "'person', 'organisation', 'topic', 'object'"

# Every written id is a row; ``canonical_id`` is the id its identity resolves to (itself for a
# canonical row, filled by the ``entities_canonical_default`` insert trigger that corpus_0005
# creates and the metadata cannot describe). Deferred, so a merge can repoint a group in one commit.
entities = Table(
    "entities",
    corpus_metadata,
    _surrogate("entity_id"),
    Column("natural_id", Text, nullable=False),
    Column("type", Text, nullable=False),
    Column("label", Text, nullable=False),
    Column("normalized_label", Text, nullable=False),
    Column("canonical_id", BigInteger, nullable=False),
    Column("scope", Text, nullable=True),
    Column("flags", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    _created_at(),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("natural_id", name="entities_natural_id_key"),
    ForeignKeyConstraint(
        ["canonical_id"],
        ["entities.entity_id"],
        name="entities_canonical_fkey",
        deferrable=True,
        initially="DEFERRED",
    ),
    Index("entities_canonical_idx", "canonical_id"),
    Index("entities_type_label_idx", "type", "normalized_label"),
    Index(
        "entities_label_trgm_idx",
        "normalized_label",
        postgresql_using="gin",
        postgresql_ops={"normalized_label": "gin_trgm_ops"},
    ),
)

entity_identifiers = Table(
    "entity_identifiers",
    corpus_metadata,
    Column(
        "entity_id",
        BigInteger,
        ForeignKey("entities.entity_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("scheme", Text, nullable=False),
    Column("value", Text, nullable=False),
    PrimaryKeyConstraint("entity_id", "scheme", name="entity_identifiers_pkey"),
    UniqueConstraint("scheme", "value", name="entity_identifiers_scheme_value_key"),
)

identity_builds = Table(
    "identity_builds",
    corpus_metadata,
    _surrogate("build_id"),
    Column("rule_version", Text, nullable=False),
    Column("inputs", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("built_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("state", Text, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=True),
    CheckConstraint(
        "state IN ('built', 'published', 'retired', 'refused')", name="identity_builds_state_check"
    ),
    # One pointer to the served build; undo publishes the previous build again.
    Index(
        "identity_builds_one_published_idx",
        text("(true)"),
        unique=True,
        postgresql_where=text("state = 'published'"),
    ),
)

# The decision rows of an identity build: what was decided, on what basis, by whom.
identity_changes = Table(
    "identity_changes",
    corpus_metadata,
    Column(
        "decision_id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "build_id",
        BigInteger,
        ForeignKey("identity_builds.build_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("scope", Text, nullable=False),
    Column("action", Text, nullable=False),
    Column("subject_a", Text, nullable=False),
    Column("subject_b", Text, nullable=True),
    Column("basis", Text, nullable=False),
    Column("scorer", Text, nullable=True),
    Column("scorer_version", Text, nullable=True),
    Column("score", REAL, nullable=True),
    Column("evidence", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("decided_by", Text, nullable=False),
    Column("decided_at", DateTime(timezone=True), nullable=False),
    Column("status", Text, nullable=False, server_default=text("'accepted'")),
    CheckConstraint(f"scope IN ({_SCOPES})", name="identity_changes_scope_check"),
    CheckConstraint(
        "action IN ('merge', 'reject', 'confirm')", name="identity_changes_action_check"
    ),
    CheckConstraint("basis IN ('rule', 'scorer', 'human')", name="identity_changes_basis_check"),
    CheckConstraint("status IN ('accepted')", name="identity_changes_status_check"),
    Index("identity_changes_build_idx", "build_id"),
)

# ``cluster_id`` is assigned by the build, durable across builds while the canonical is unchanged.
identity_clusters = Table(
    "identity_clusters",
    corpus_metadata,
    Column(
        "build_id",
        BigInteger,
        ForeignKey("identity_builds.build_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("cluster_id", BigInteger, nullable=False),
    Column("scope", Text, nullable=False),
    Column("canonical_id", BigInteger, ForeignKey("entities.entity_id"), nullable=False),
    Column("display_name", Text, nullable=True),
    Column("display_evidence", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("member_count", Integer, nullable=False),
    PrimaryKeyConstraint("build_id", "cluster_id", name="identity_clusters_pkey"),
    UniqueConstraint(
        "build_id", "cluster_id", "scope", name="identity_clusters_build_cluster_scope_key"
    ),
    CheckConstraint(f"scope IN ({_SCOPES})", name="identity_clusters_scope_check"),
    CheckConstraint("member_count >= 1", name="identity_clusters_member_count_check"),
    Index("identity_clusters_canonical_idx", "canonical_id"),
)

identity_members = Table(
    "identity_members",
    corpus_metadata,
    Column("build_id", BigInteger, nullable=False),
    Column("cluster_id", BigInteger, nullable=False),
    Column("member_id", BigInteger, ForeignKey("entities.entity_id"), nullable=False),
    Column("scope", Text, nullable=False),
    Column(
        "decision_id",
        UUID(as_uuid=True),
        ForeignKey("identity_changes.decision_id", ondelete="SET NULL"),
        nullable=True,
    ),
    # One written id is in one cluster per build.
    PrimaryKeyConstraint("build_id", "member_id", name="identity_members_pkey"),
    ForeignKeyConstraint(
        ["build_id", "cluster_id", "scope"],
        ["identity_clusters.build_id", "identity_clusters.cluster_id", "identity_clusters.scope"],
        name="identity_members_cluster_fkey",
        ondelete="CASCADE",
    ),
    CheckConstraint(f"scope IN ({_SCOPES})", name="identity_members_scope_check"),
    Index("identity_members_member_idx", "member_id"),
    Index("identity_members_cluster_idx", "build_id", "cluster_id", "scope"),
    Index("identity_members_decision_idx", "decision_id"),
)

# Keyed by document, not variant, so an old version keeps the text its quotes were cut from. The
# sixteen hash partitions (segments_p00..p15) are made by corpus_0004, not listed in the metadata.
segments = Table(
    "segments",
    corpus_metadata,
    Column("item_id", BigInteger, nullable=False),
    Column("document_id", BigInteger, nullable=False),
    Column("seq", Integer, nullable=False),
    Column("start_ms", Integer, nullable=True),
    Column("end_ms", Integer, nullable=True),
    Column("speaker_id", BigInteger, nullable=True),
    Column("text", Text, nullable=False),
    PrimaryKeyConstraint("item_id", "document_id", "seq", name="segments_pkey"),
    ForeignKeyConstraint(
        ["item_id", "document_id"],
        ["documents.item_id", "documents.document_id"],
        name="segments_document_fkey",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["speaker_id"], ["entities.entity_id"], name="segments_speaker_fkey", ondelete="SET NULL"
    ),
    CheckConstraint(
        "start_ms IS NULL OR end_ms IS NULL OR end_ms >= start_ms", name="segments_span_check"
    ),
    Index("segments_speaker_idx", "speaker_id"),
    postgresql_partition_by="HASH (item_id)",
)

# A transcript span the search index cut by tokens, so it can cross voices: ``speakers`` holds
# every speaker of its range. Offsets are characters into the cuts of ``document_cut``.
search_chunks = Table(
    "search_chunks",
    corpus_metadata,
    _surrogate("chunk_id"),
    Column("natural_id", Text, nullable=False),
    Column("item_id", BigInteger, nullable=False),
    Column("document_id", BigInteger, nullable=False),
    Column("start_seq", Integer, nullable=False),
    Column("start_offset", Integer, nullable=False),
    Column("end_seq", Integer, nullable=False),
    Column("end_offset", Integer, nullable=False),
    Column("speakers", ARRAY(BigInteger), nullable=False, server_default=text("'{}'::bigint[]")),
    _created_at(),
    UniqueConstraint("natural_id", name="search_chunks_natural_id_key"),
    ForeignKeyConstraint(
        ["item_id", "document_id"],
        ["documents.item_id", "documents.document_id"],
        name="search_chunks_document_fkey",
        ondelete="CASCADE",
    ),
    CheckConstraint(
        "start_seq >= 0 AND start_offset >= 0 AND end_offset >= 0",
        name="search_chunks_bounds_check",
    ),
    CheckConstraint(
        "(end_seq, end_offset) > (start_seq, start_offset)", name="search_chunks_span_check"
    ),
    Index("search_chunks_document_idx", "item_id", "document_id", "start_seq"),
    Index("search_chunks_speakers_idx", "speakers", postgresql_using="gin"),
)

_EMBEDDING_KINDS = (
    "'insight', 'quote', 'transcript', 'summary', 'summary_short', 'kg_topic', 'kg_entity', "
    "'episode_title', 'episode_description'"
)

# Keyed by the object's natural id (most kinds have no table yet); ``text_hash`` lets a re-index
# embed only what changed. ``vector`` is real[] until E09 chooses its type.
embeddings = Table(
    "embeddings",
    corpus_metadata,
    Column("kind", Text, nullable=False),
    Column("object_id", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("text_hash", Text, nullable=False),
    Column("dims", SmallInteger, nullable=False),
    Column("vector", ARRAY(REAL), nullable=False),
    _created_at(),
    PrimaryKeyConstraint("kind", "object_id", "model", name="embeddings_pkey"),
    CheckConstraint(f"kind IN ({_EMBEDDING_KINDS})", name="embeddings_kind_check"),
    CheckConstraint(
        "dims >= 1 AND array_ndims(vector) = 1 AND cardinality(vector) = dims",
        name="embeddings_dims_check",
    ),
    Index("embeddings_model_idx", "model"),
)

# One row (id 1), bumped last in every content transaction (DECISIONS rule 7); corpus_0007 seeds it.
corpus_revision = Table(
    "corpus_revision",
    corpus_metadata,
    Column("id", SmallInteger, primary_key=True, server_default=text("1")),
    Column("revision", BigInteger, nullable=False, server_default=text("0")),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("id = 1", name="corpus_revision_one_row_check"),
    CheckConstraint("revision >= 0", name="corpus_revision_revision_check"),
)

# CRFC-005 3.4: an item's insights (of one document version), the quotes that support them, and the
# entity edges. The edge tables carry the autovacuum settings of section 6.
_EDGE_STORAGE = {"autovacuum_vacuum_scale_factor": 0.01, "autovacuum_analyze_scale_factor": 0.005}
_INSIGHT_TYPES = "'claim', 'recommendation', 'observation', 'question', 'unknown'"
_MENTION_KINDS = "'ABOUT', 'MENTIONS_PERSON', 'MENTIONS_ORG', 'MENTIONS_OBJECT'"
_PERSON_ROLES = "'host', 'guest', 'mentioned'"
_ROUTING_TAGS = "'surface', 'connect', 'drop'"

insights = Table(
    "insights",
    corpus_metadata,
    _surrogate("insight_id"),
    Column("natural_id", Text, nullable=False),
    Column("item_id", BigInteger, nullable=False),
    Column("document_id", BigInteger, nullable=False),
    Column("insight_type", Text, nullable=False),
    Column("rank", Integer, nullable=False),
    Column("salience", REAL, nullable=True),
    Column("routing_tag", Text, nullable=False),
    Column("surfaceable", Boolean, nullable=True),
    Column("grounded", Boolean, nullable=True),
    Column("tier", SmallInteger, nullable=True),
    Column("grounding_confidence", REAL, nullable=True),
    Column("position_hint", REAL, nullable=True),
    Column("text_origin", Text, nullable=False),
    Column("visible", Boolean, nullable=False, server_default=true()),
    Column("text", Text, nullable=False),
    UniqueConstraint("natural_id", name="insights_natural_id_key"),
    ForeignKeyConstraint(
        ["item_id", "document_id"],
        ["documents.item_id", "documents.document_id"],
        name="insights_document_fkey",
        ondelete="CASCADE",
    ),
    CheckConstraint(f"insight_type IN ({_INSIGHT_TYPES})", name="insights_type_check"),
    CheckConstraint("text_origin IN ('authored')", name="insights_text_origin_check"),
    CheckConstraint("rank >= 0", name="insights_rank_check"),
    CheckConstraint(f"routing_tag IN ({_ROUTING_TAGS})", name="insights_routing_tag_check"),
    CheckConstraint("tier IS NULL OR tier BETWEEN 0 AND 3", name="insights_tier_check"),
    CheckConstraint(
        "(grounding_confidence IS NULL OR grounding_confidence BETWEEN 0 AND 1) "
        "AND (position_hint IS NULL OR position_hint BETWEEN 0 AND 1)",
        name="insights_unit_check",
    ),
    Index("insights_item_rank_idx", "item_id", "rank"),
    Index("insights_document_idx", "item_id", "document_id"),
)

evidence = Table(
    "evidence",
    corpus_metadata,
    _surrogate("evidence_id"),
    Column("natural_id", Text, nullable=False),
    Column("item_id", BigInteger, nullable=False),
    Column("document_id", BigInteger, nullable=False),
    Column("speaker_id", BigInteger, nullable=True),
    Column("start_seq", Integer, nullable=False),
    Column("start_offset", Integer, nullable=False),
    Column("end_seq", Integer, nullable=False),
    Column("end_offset", Integer, nullable=False),
    Column("start_ms", Integer, nullable=True),
    Column("end_ms", Integer, nullable=True),
    Column("text", Text, nullable=False),
    UniqueConstraint("natural_id", name="evidence_natural_id_key"),
    ForeignKeyConstraint(
        ["item_id", "document_id"],
        ["documents.item_id", "documents.document_id"],
        name="evidence_document_fkey",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["speaker_id"], ["entities.entity_id"], name="evidence_speaker_fkey", ondelete="SET NULL"
    ),
    CheckConstraint(
        "start_seq >= 0 AND start_offset >= 0 AND end_offset >= 0", name="evidence_bounds_check"
    ),
    CheckConstraint(
        "(end_seq, end_offset) > (start_seq, start_offset)", name="evidence_span_check"
    ),
    CheckConstraint(
        "(start_ms IS NULL) = (end_ms IS NULL) "
        "AND (start_ms IS NULL OR (start_ms >= 0 AND end_ms >= start_ms))",
        name="evidence_ms_check",
    ),
    Index("evidence_item_idx", "item_id"),
    Index("evidence_document_idx", "item_id", "document_id"),
    Index(
        "evidence_speaker_idx",
        "speaker_id",
        "item_id",
        postgresql_where=text("speaker_id IS NOT NULL"),
    ),
    postgresql_with=_EDGE_STORAGE,
)

insight_support = Table(
    "insight_support",
    corpus_metadata,
    Column("insight_id", BigInteger, nullable=False),
    Column("evidence_id", BigInteger, nullable=False),
    Column("ord", Integer, nullable=False),
    PrimaryKeyConstraint("insight_id", "evidence_id", name="insight_support_pkey"),
    ForeignKeyConstraint(
        ["insight_id"],
        ["insights.insight_id"],
        name="insight_support_insight_fkey",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["evidence_id"],
        ["evidence.evidence_id"],
        name="insight_support_evidence_fkey",
        ondelete="CASCADE",
    ),
    CheckConstraint("ord >= 0", name="insight_support_ord_check"),
    Index("insight_support_evidence_idx", "evidence_id"),
    postgresql_with=_EDGE_STORAGE,
)

insight_mentions = Table(
    "insight_mentions",
    corpus_metadata,
    Column("insight_id", BigInteger, nullable=False),
    Column("entity_id", BigInteger, nullable=False),
    Column("kind", Text, nullable=False),
    Column("item_id", BigInteger, nullable=False),
    Column("rank", Integer, nullable=False),
    PrimaryKeyConstraint("insight_id", "kind", "entity_id", name="insight_mentions_pkey"),
    ForeignKeyConstraint(
        ["insight_id"],
        ["insights.insight_id"],
        name="insight_mentions_insight_fkey",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["entity_id"], ["entities.entity_id"], name="insight_mentions_entity_fkey"
    ),
    ForeignKeyConstraint(
        ["item_id"],
        ["content_items.item_id"],
        name="insight_mentions_item_fkey",
        ondelete="CASCADE",
    ),
    CheckConstraint(f"kind IN ({_MENTION_KINDS})", name="insight_mentions_kind_check"),
    CheckConstraint("rank >= 0", name="insight_mentions_rank_check"),
    Index(
        "insight_mentions_entity_rank_idx",
        "entity_id",
        "kind",
        "rank",
        "insight_id",
        postgresql_include=["item_id"],
    ),
    Index("insight_mentions_item_idx", "item_id"),
    postgresql_with=_EDGE_STORAGE,
)

insight_attribution = Table(
    "insight_attribution",
    corpus_metadata,
    Column("insight_id", BigInteger, nullable=False),
    Column("person_id", BigInteger, nullable=False),
    Column("item_id", BigInteger, nullable=False),
    PrimaryKeyConstraint("insight_id", "person_id", name="insight_attribution_pkey"),
    ForeignKeyConstraint(
        ["insight_id"],
        ["insights.insight_id"],
        name="insight_attribution_insight_fkey",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["person_id"], ["entities.entity_id"], name="insight_attribution_person_fkey"
    ),
    ForeignKeyConstraint(
        ["item_id"],
        ["content_items.item_id"],
        name="insight_attribution_item_fkey",
        ondelete="CASCADE",
    ),
    Index("insight_attribution_person_idx", "person_id", "item_id"),
    Index("insight_attribution_item_idx", "item_id"),
    postgresql_with=_EDGE_STORAGE,
)

# ``published_at`` and ``content_type`` are copied from the item so an entity's newest items, of
# every type or of one, are an index range; the item's writer keeps them equal.
item_entities = Table(
    "item_entities",
    corpus_metadata,
    Column("item_id", BigInteger, nullable=False),
    Column("entity_id", BigInteger, nullable=False),
    Column("role", Text, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=True),
    Column("content_type", Text, nullable=False),
    PrimaryKeyConstraint("item_id", "entity_id", "role", name="item_entities_pkey"),
    ForeignKeyConstraint(
        ["item_id"], ["content_items.item_id"], name="item_entities_item_fkey", ondelete="CASCADE"
    ),
    ForeignKeyConstraint(["entity_id"], ["entities.entity_id"], name="item_entities_entity_fkey"),
    Index("item_entities_entity_published_idx", "entity_id", text("published_at DESC"), "item_id"),
    Index(
        "item_entities_entity_type_published_idx",
        "entity_id",
        "content_type",
        text("published_at DESC"),
        "item_id",
    ),
    postgresql_with=_EDGE_STORAGE,
)

person_roles = Table(
    "person_roles",
    corpus_metadata,
    Column("item_id", BigInteger, nullable=False),
    Column("person_id", BigInteger, nullable=False),
    Column("role", Text, nullable=False),
    PrimaryKeyConstraint("item_id", "person_id", "role", name="person_roles_pkey"),
    ForeignKeyConstraint(
        ["item_id"], ["content_items.item_id"], name="person_roles_item_fkey", ondelete="CASCADE"
    ),
    ForeignKeyConstraint(["person_id"], ["entities.entity_id"], name="person_roles_person_fkey"),
    CheckConstraint(f"role IN ({_PERSON_ROLES})", name="person_roles_role_check"),
    Index("person_roles_person_idx", "person_id", "item_id"),
)

# No build table exists until the clusters revision, so ``build_id`` is a plain number here.
topic_relations = Table(
    "topic_relations",
    corpus_metadata,
    Column("topic_a", BigInteger, nullable=False),
    Column("topic_b", BigInteger, nullable=False),
    Column("kind", Text, nullable=False),
    Column("weight", REAL, nullable=False),
    Column("build_id", BigInteger, nullable=False),
    PrimaryKeyConstraint("build_id", "topic_a", "topic_b", "kind", name="topic_relations_pkey"),
    ForeignKeyConstraint(["topic_a"], ["entities.entity_id"], name="topic_relations_topic_a_fkey"),
    ForeignKeyConstraint(["topic_b"], ["entities.entity_id"], name="topic_relations_topic_b_fkey"),
    CheckConstraint("topic_a <> topic_b", name="topic_relations_distinct_check"),
    Index("topic_relations_a_idx", "topic_a", "kind", text("weight DESC"), "topic_b"),
    Index("topic_relations_b_idx", "topic_b", "kind", text("weight DESC"), "topic_a"),
)

# The audio dedup gate's index (#1656), one row per digest a source's episode was transcribed from
# (operator, 2026-11-26: a corpus table, not a file beside the rows). ``identity`` is the episode's
# GUID, else its media URL, as the gate keys it; ``item_id`` is that item when it is a row.
# ``duplicates`` are the identities the gate skipped as copies of it, which the nightly's
# ``episode_selection=unprocessed`` must not offer again.
audio_fingerprints = Table(
    "audio_fingerprints",
    corpus_metadata,
    Column("source_id", BigInteger, nullable=False),
    Column("digest", Text, nullable=False),
    Column("identity", Text, nullable=False),
    Column("item_id", BigInteger, nullable=True),
    Column("episode_title", Text, nullable=True),
    Column("transcript_path", Text, nullable=True),
    Column("recorded_at", DateTime(timezone=True), nullable=True),
    Column("duplicates", ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")),
    PrimaryKeyConstraint("source_id", "digest", name="audio_fingerprints_pkey"),
    ForeignKeyConstraint(
        ["source_id"],
        ["sources.source_id"],
        name="audio_fingerprints_source_fkey",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["item_id"],
        ["content_items.item_id"],
        name="audio_fingerprints_item_fkey",
        ondelete="SET NULL",
    ),
    CheckConstraint("digest ~ '^[0-9a-f]{64}$'", name="audio_fingerprints_digest_check"),
    Index("audio_fingerprints_item_idx", "item_id"),
)
