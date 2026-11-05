"""users

Revision ID: 0001_users
Revises:
"""

import sqlalchemy as sa
from alembic import context, op

revision = "0001_users"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the ``users`` table."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "users",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("email", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("name", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("username", sa.Text(), nullable=True),
        sa.Column("image", sa.Text(), nullable=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("disabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("role", sa.Text(), nullable=False, server_default=sa.text("'listener'")),
        sa.Column("mcp_access", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("analytics_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("username", name="users_username_key"),
        sa.UniqueConstraint("analytics_id", name="users_analytics_id_key"),
        sa.CheckConstraint("role IN ('listener', 'creator', 'admin')", name="users_role_check"),
        sa.UniqueConstraint("provider", "subject", name="users_provider_subject_key"),
        schema=schema,
    )
