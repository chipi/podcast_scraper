"""demo items

Revision ID: demo_0001_items
Revises:
"""

import sqlalchemy as sa
from alembic import context, op

revision = "demo_0001_items"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = context.config.attributes["plane"]
    op.create_table(
        "demo_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], [f"{schema}.users.user_id"], ondelete="CASCADE"),
        schema=schema,
    )


def downgrade() -> None:
    op.drop_table("demo_items", schema=context.config.attributes["plane"])
