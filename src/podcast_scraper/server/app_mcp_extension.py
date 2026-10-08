"""The MCP sign-in surface as an extension (ADR-158): token management, the OAuth authorization
server, the internal verify seam, and what account deletion must remove for it.

Moves to the private Common package (``identity``) at the cutover; the platform then has no MCP
routes and nothing MCP-shaped to delete. Imports stay inside the functions: this module is loaded
wherever extensions are, including the pipeline, which has no web stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, TYPE_CHECKING

from podcast_scraper.extensions import Extension, RouterMount

if TYPE_CHECKING:
    from podcast_scraper.server.app_user_store import User


def _routers() -> Sequence[RouterMount]:
    from podcast_scraper.server.routes import app_mcp, internal_mcp, mcp_oauth

    return (
        RouterMount(app_mcp.router, "app"),
        RouterMount(mcp_oauth.router, "app"),
        # Service-to-service, token-gated, tailnet-only (#1471).
        RouterMount(internal_mcp.router, "internal"),
        # RFC 8414 discovery at the app root.
        RouterMount(mcp_oauth.wellknown_router, "root"),
    )


def _forget_user(data_dir: Path, user: User) -> dict[str, int]:
    """The personal-token index and the OAuth server's grants, consents and last-use records."""
    from podcast_scraper.server import app_mcp_tokens, app_oauth_server

    return {
        "mcp_token_index": app_mcp_tokens.forget_user(data_dir, user.user_id),
        "mcp_oauth_records": app_oauth_server.forget_user(data_dir, user.user_id),
    }


EXTENSION = Extension(name="mcp", routers=_routers, account_deleted=(_forget_user,))
