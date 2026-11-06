"""The demo module's Alembic environment: the kernel's runner with the module's metadata."""

from podcast_scraper.db.migrations._env import run
from tests.unit.podcast_scraper.db.demo_module import metadata

run(metadata)
