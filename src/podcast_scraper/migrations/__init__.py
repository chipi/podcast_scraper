"""Offline corpus migration helpers (RFC-072, etc.)."""

from podcast_scraper.migrations.rfc072 import migrate_gil_document, migrate_kg_document

__all__ = ["migrate_gil_document", "migrate_kg_document"]
