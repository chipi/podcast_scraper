"""shared: app_releases, and the schema's own schema_history

Revision ID: shared_0001_app_releases
Revises:
"""

import sqlalchemy as sa
from alembic import context, op

revision = "shared_0001_app_releases"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``app_releases`` and ``schema_history`` in the shared schema."""
    schema = context.config.attributes["plane"]
    op.create_table(
        "app_releases",
        sa.Column("app", sa.Text(), primary_key=True),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("app ~ '^[a-z][a-z0-9_]{0,31}$'", name="app_releases_app_check"),
        sa.CheckConstraint(
            "version ~ '^[0-9]{1,4}(\\.[0-9]{1,4}){0,3}$'", name="app_releases_version_check"
        ),
        schema=schema,
    )
    op.create_table(
        "schema_history",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column(
            "applied_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("release", sa.Text(), nullable=True),
        sa.Column("release_source", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        schema=schema,
    )
