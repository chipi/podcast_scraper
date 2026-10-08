"""The MCP sign-in surface as an extension (ADR-158): token management, the OAuth authorization
server, the internal verify seam, and what account deletion must remove for it.

Moves to the private Common package (``identity``) at the cutover; the platform then has no MCP
routes and nothing MCP-shaped to delete.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.extensions import Extension, RouterMount
from podcast_scraper.server import app_mcp_tokens, app_oauth_server
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes import app_mcp, internal_mcp, mcp_oauth


def _forget_user(data_dir: Path, user: User) -> dict[str, int]:
    """The personal-token index and the OAuth server's grants, consents and last-use records."""
    return {
        "mcp_token_index": app_mcp_tokens.forget_user(data_dir, user.user_id),
        "mcp_oauth_records": app_oauth_server.forget_user(data_dir, user.user_id),
    }


EXTENSION = Extension(
    name="mcp",
    routers=(
        RouterMount(app_mcp.router, "app"),
        RouterMount(mcp_oauth.router, "app"),
        # Service-to-service, token-gated, tailnet-only (#1471).
        RouterMount(internal_mcp.router, "internal"),
        # RFC 8414 discovery at the app root.
        RouterMount(mcp_oauth.wellknown_router, "root"),
    ),
    account_deleted=(_forget_user,),
)
