"""The player as an extension (ADR-158): what it adds to the platform's lifecycle.

Moves to the private Player package at the cutover. Its routers, startup hooks and jobs join this
module as each seam is cut.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.extensions import Extension
from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_user_store import User


def _account_created(data_dir: Path, user: User, provider: str) -> None:
    app_user_state.append_account_created(data_dir, user.user_id, provider)


EXTENSION = Extension(name="player", account_created=(_account_created,))
