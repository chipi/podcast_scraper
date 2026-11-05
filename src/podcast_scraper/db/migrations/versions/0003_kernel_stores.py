"""kernel stores: access policy, comms, push, preferences, Apple tokens, graph events, outbox,
magic-link markers, audit log

Revision ID: 0003_kernel_stores
Revises: 0002_schema_history
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0003_kernel_stores"
down_revision = "0002_schema_history"
branch_labels = None
depends_on = None


def _updated_at() -> sa.Column:
    return sa.Column(
        "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )


def _account(schema: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(["user_id"], [f"{schema}.users.user_id"], ondelete="CASCADE")


def upgrade() -> None:
    """Create the tables of every remaining kernel store."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "access_policy",
        sa.Column("id", sa.SmallInteger(), primary_key=True, server_default=sa.text("1")),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column(
            "allowed_emails",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "allowed_domains",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        _updated_at(),
        sa.CheckConstraint("id = 1", name="access_policy_single_row"),
        sa.CheckConstraint("mode IN ('allowlist', 'open')", name="access_policy_mode_check"),
        schema=schema,
    )
    op.create_table(
        "comms",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("settings", postgresql.JSONB(), nullable=False),
        sa.Column("unsubscribe_ref", sa.Text(), nullable=True),
        _updated_at(),
        _account(schema),
        sa.UniqueConstraint("unsubscribe_ref", name="comms_unsubscribe_ref_key"),
        schema=schema,
    )
    op.create_table(
        "push_subscriptions",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("endpoint", sa.Text(), primary_key=True),
        sa.Column("subscription", postgresql.JSONB(), nullable=False),
        sa.Column("seq", sa.BigInteger(), sa.Identity(), nullable=False),
        _account(schema),
        schema=schema,
    )
    op.create_table(
        "user_preferences",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        _updated_at(),
        _account(schema),
        schema=schema,
    )
    op.create_table(
        "apple_tokens",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("refresh_token", sa.Text(), nullable=False),
        _updated_at(),
        _account(schema),
        schema=schema,
    )
    op.create_table(
        "graph_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("event", postgresql.JSONB(), nullable=False),
        _account(schema),
        schema=schema,
    )
    op.create_index("graph_events_user_idx", "graph_events", ["user_id", "id"], schema=schema)
    op.create_table(
        "outbox",
        sa.Column("envelope_id", sa.Text(), primary_key=True),
        sa.Column("envelope", postgresql.JSONB(), nullable=False),
        sa.Column("channel", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Text(), nullable=True),
        sa.Column("recipient_email", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        _updated_at(),
        sa.CheckConstraint(
            "status IN ('pending', 'delivered', 'bounced', 'complaint', 'suppressed', 'failed')",
            name="outbox_status_check",
        ),
        schema=schema,
    )
    op.create_index(
        "outbox_pending_idx",
        "outbox",
        ["channel", "updated_at"],
        schema=schema,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index("outbox_user_idx", "outbox", ["user_id"], schema=schema)
    op.create_index("outbox_recipient_idx", "outbox", ["recipient_email"], schema=schema)
    op.create_table(
        "magic_link_used",
        sa.Column("token_id", sa.Text(), primary_key=True),
        sa.Column(
            "used_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        schema=schema,
    )
    op.create_table(
        "magic_link_requests",
        sa.Column("email_hash", sa.Text(), primary_key=True),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        schema=schema,
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("record", postgresql.JSONB(), nullable=False),
        schema=schema,
    )
