"""Kernel tables (ADR-169): defined WITHOUT a schema, mapped onto the plane at connection time.

Never add ``schema=`` to a ``metadata`` table: the engine's ``schema_translate_map={None: plane}``
and Alembic's per-plane run depend on every kernel table having none. The one exception is the
``shared`` schema at the end, with its own metadata and its own history.
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
