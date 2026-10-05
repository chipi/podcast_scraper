"""The public player never serves the admin surface (operator decision 2026-10-05).

Admin — user management, access policy, operator overrides, ranking config, the graph-event
readout — is tailnet-only. The player nginx blocks those paths with a regex location that wins
over the ``/api/app/`` proxy prefix. This test reads the SHIPPED config and runs its regex over a
table of real route paths, so a broadened rule that eats a consumer route fails here as surely as
a narrowed one that lets an admin route through. (The slow docker test in
tests/integration/server/test_player_nginx_runtime.py proves the same at runtime.)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CONF = Path(__file__).resolve().parents[4] / "web" / "learning-player" / "nginx.conf"


def _block_regex() -> "re.Pattern[str]":
    text = _CONF.read_text(encoding="utf-8")
    found = re.findall(r"location ~ (\S+) \{\s*return 404;\s*\}", text)
    assert len(found) == 1, f"expected exactly one admin-blocking location, found {found}"
    return re.compile(found[0])


@pytest.mark.parametrize(
    "path",
    [
        "/api/app/admin/users",
        "/api/app/admin/users/u123",
        "/api/app/admin/access-policy",
        "/api/app/admin/overrides",
        "/api/app/admin/overrides/feed",
        "/api/app/admin",
        "/api/app/ranking-config",
        "/api/app/graph-events/summary",
        "/api/app/graph-events/sessions",
        "/api/app/graph-events/session/abc",
    ],
)
def test_every_admin_path_is_blocked(path: str) -> None:
    assert _block_regex().search(path), path


@pytest.mark.parametrize(
    "path",
    [
        "/api/app/graph-events",  # the player's OWN event ingestion (POST) — must stay served
        "/api/app/me",
        "/api/app/auth/login",
        "/api/app/episodes",
        "/api/app/discover",
        "/api/app/administrator-notes",  # a hypothetical consumer route sharing the prefix text
        "/api/app/ranking-configs",
    ],
)
def test_consumer_paths_are_not_blocked(path: str) -> None:
    assert not _block_regex().search(path), path


def test_the_block_comes_before_the_consumer_proxy() -> None:
    """Order is not what makes nginx prefer a regex location, but the rule and its reason are
    meant to be read together, next to the proxy they carve out of."""
    text = _CONF.read_text(encoding="utf-8")
    assert text.index("return 404;") < text.index("location /api/app/ {")
