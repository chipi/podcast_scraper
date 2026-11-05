"""Alembic environment: one kernel history, run once per plane schema (ADR-169); see ``_env``."""

from podcast_scraper.db.migrations._env import run
from podcast_scraper.db.schema import metadata

run(metadata)
