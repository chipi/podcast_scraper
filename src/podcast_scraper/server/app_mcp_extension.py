"""Common identity as an extension (ADR-158): the Google and Apple sign-in providers, and the MCP
sign-in surface (token management, the OAuth authorization server, the internal verify seam, and
what account deletion must remove for it).

Moves to the private Common package (``identity``) at the cutover; the platform then has no MCP
routes and nothing MCP-shaped to delete. Imports stay inside the functions: this module is loaded
wherever extensions are, including the pipeline, which has no web stack.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

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


def _internal_token(app: Any) -> None:
    """Shared token for the internal MCP verify seam (RFC-112 §4, #1471): the MCP server process
    authenticates with it over the tailnet. Empty → ``/internal/mcp/verify`` 503s (disabled)."""
    app.state.internal_mcp_token = os.environ.get("INTERNAL_MCP_TOKEN", "")


def _google() -> Any:
    from podcast_scraper.server.app_oauth_providers import GoogleProvider

    return GoogleProvider.from_env()


def _apple() -> Any:
    from podcast_scraper.server.app_oauth_providers import AppleProvider

    return AppleProvider.from_env()


EXTENSION = Extension(
    name="mcp",
    routers=_routers,
    account_deleted=(_forget_user,),
    app_configured=(_internal_token,),
    oauth_providers={"google": _google, "apple": _apple},
)
