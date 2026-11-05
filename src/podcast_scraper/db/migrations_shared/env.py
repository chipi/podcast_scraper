"""Alembic environment: the ``shared`` schema's history (ADR-169); see ``migrations._env``."""

from podcast_scraper.db.migrations._env import run
from podcast_scraper.db.schema import shared_metadata

run(shared_metadata)
