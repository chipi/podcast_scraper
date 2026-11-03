"""Kernel tables (ADR-169): defined WITHOUT a schema, mapped onto the plane at connection time.

Never add ``schema=`` here: the engine's ``schema_translate_map={None: plane}`` and Alembic's
per-plane run depend on every kernel table having none.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    func,
    MetaData,
    Table,
    Text,
    text,
    UniqueConstraint,
)

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
