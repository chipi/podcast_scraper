"""Alembic environment: the ``corpus`` schema's history (ADR-184); see ``migrations._env``."""

from podcast_scraper.db.migrations._env import run
from podcast_scraper.db.schema import corpus_metadata

run(corpus_metadata)
