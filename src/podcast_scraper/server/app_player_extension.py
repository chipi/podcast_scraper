"""The player as an extension (ADR-158): what it adds to the platform's lifecycle.

Moves to the private Player package at the cutover. Its routers, startup hooks and jobs join this
module as each seam is cut. Imports stay inside the functions (see ``app_mcp_extension``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from podcast_scraper.extensions import Extension

if TYPE_CHECKING:
    from podcast_scraper.server.app_user_store import User


def _account_created(data_dir: Path, user: User, provider: str) -> None:
    from podcast_scraper.server import app_user_state

    app_user_state.append_account_created(data_dir, user.user_id, provider)


EXTENSION = Extension(name="player", account_created=(_account_created,))
