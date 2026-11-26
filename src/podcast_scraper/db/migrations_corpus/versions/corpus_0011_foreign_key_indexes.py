"""corpus: an index under every foreign key that had none

Revision ID: corpus_0011_foreign_key_indexes
Revises: corpus_0010_source_type_attrs
"""

from alembic import context, op

revision = "corpus_0011_foreign_key_indexes"
down_revision = "corpus_0010_source_type_attrs"
branch_labels = None
depends_on = None

#: (index, table, columns): each foreign key's own columns, in its order.
_INDEXES = (
    ("corpus_migrations_last_run_idx", "corpus_migrations", ["last_run_id"]),
    ("documents_supersedes_idx", "documents", ["item_id", "kind", "variant", "supersedes_id"]),
    ("identity_members_cluster_idx", "identity_members", ["build_id", "cluster_id", "scope"]),
    ("identity_members_decision_idx", "identity_members", ["decision_id"]),
    ("segments_speaker_idx", "segments", ["speaker_id"]),
    ("insights_document_idx", "insights", ["item_id", "document_id"]),
    ("evidence_document_idx", "evidence", ["item_id", "document_id"]),
    ("insight_attribution_item_idx", "insight_attribution", ["item_id"]),
)


def upgrade() -> None:
    """Index the columns of the eight foreign keys no index started with. A delete of a parent row
    (an entity merged away, a document or an item removed) checks or cascades into its children by
    these columns, and without an index each delete scans the child table: every partition of
    ``segments`` for one entity. On ``segments`` the index is created on the partitioned table and
    so on each partition."""
    schema = context.config.attributes["plane"]
    for name, table, columns in _INDEXES:
        op.create_index(name, table, columns, schema=schema)
