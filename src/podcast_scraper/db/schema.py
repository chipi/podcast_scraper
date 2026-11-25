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
    func,
    Identity,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    Table,
    Text,
    text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

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
    UniqueConstraint("natural_id", name="corpus_migrations_natural_id_key"),
    CheckConstraint(
        "status IN ('dry_run', 'applied', 'undone')", name="corpus_migrations_status_check"
    ),
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
    Index("migration_receipts_migration_idx", "migration_id"),
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
