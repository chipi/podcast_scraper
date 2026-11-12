"""Offline corpus migration helpers: the KG document format transforms public keeps."""

from podcast_scraper.migrations.kg_document_migrations import (
    migrate_kg_document,
    migrate_kg_document_v2,
)

__all__ = ["migrate_kg_document", "migrate_kg_document_v2"]
