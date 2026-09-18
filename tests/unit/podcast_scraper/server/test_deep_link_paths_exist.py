"""Every deep link the server hands out must be a path the app can actually open.

The bug this guards (operator 2026-09-18): eight producers emitted ``/player/<slug>`` — in-app
notifications, both digest sections, the personal digest, the daily recap, auto-picks, PKM export
and collection items. The SPA has no ``/player`` route; it is ``/episode/:slug``. The router's
catch-all (``/:pathMatch(.*)*``) redirects anything unmatched to Home, so tapping a "new episode"
notification marked it read, closed the panel and dropped the user on Home with nothing to explain
it — and every episode link in a digest email did the same, silently.

Nothing failed, which is why it survived: a redirect-to-Home is indistinguishable from a no-op, and
no test asserted that a produced path corresponds to a real route. So this reads the ROUTER as the
source of truth rather than hard-coding a list — a renamed route makes the producers fail here
instead of in somebody's inbox.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[4]
_ROUTER = _REPO / "web" / "learning-player" / "src" / "router" / "index.ts"
_SERVER = _REPO / "src" / "podcast_scraper" / "server"

#: A deep link literal in the server sources: `"/foo/{bar}"`, `f"/foo/{bar}?t=1"`, `](/foo/{bar})`.
_DEEP_LINK = re.compile(r'["(\[]\s*(/[a-z][a-z0-9-]*)/')


def _router_roots() -> set[str]:
    """Top-level path segments the SPA router defines (``/episode``, ``/podcast``, …)."""
    src = _ROUTER.read_text(encoding="utf-8")
    roots = set()
    for raw in re.findall(r"path:\s*'([^']+)'", src):
        if not raw.startswith("/") or raw.startswith("/:"):
            continue  # catch-all and relative children are not link targets
        roots.add("/" + raw.lstrip("/").split("/")[0])
    return roots


def test_router_file_is_readable() -> None:
    """If this moves, the guard below would silently pass against an empty set."""
    assert _ROUTER.is_file(), f"router not found at {_ROUTER}"
    assert "/episode/:slug" in _ROUTER.read_text(encoding="utf-8")


def test_every_server_deep_link_points_at_a_real_route() -> None:
    roots = _router_roots()
    assert "/episode" in roots, f"expected the player route in {sorted(roots)}"

    offenders: list[str] = []
    for path in sorted(_SERVER.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "deep_link" not in line and "deep =" not in line and "](/" not in line:
                continue
            for root in _DEEP_LINK.findall(line):
                # `/api/...` is a server route, not a client one; those are not deep links.
                if root.startswith("/api"):
                    continue
                if root not in roots:
                    rel = path.relative_to(_REPO)
                    offenders.append(f"{rel}:{lineno} -> {root} (line: {line.strip()})")

    assert not offenders, (
        "These deep links name a path the SPA router does not define, so the catch-all sends the "
        "user to Home instead — silently, because a redirect is not an error:\n  "
        + "\n  ".join(offenders)
        + f"\n\nRoutes the app actually has: {sorted(roots)}"
    )


def test_the_retired_player_path_is_gone() -> None:
    """`/player/<slug>` specifically: it is the path that shipped, so name it."""
    hits = [
        f"{p.relative_to(_REPO)}"
        for p in _SERVER.rglob("*.py")
        if "/player/" in p.read_text(encoding="utf-8")
    ]
    assert not hits, f"'/player/' is not a route; use '/episode/'. Still present in: {hits}"


def test_the_delivery_golden_fixtures_use_real_routes() -> None:
    """The delivery goldens are a CONTRACT with another repo, so they have to be right too.

    Fixing the producers alone left these carrying ``/player/``: they are the envelopes the homelab
    delivery worker (``agentic-ai-homelab/infra/delivery``) renders its Jinja email templates
    against, per RFC-110. A golden that documents a dead path teaches the other side of the seam to
    emit one, and the first guard here only scanned ``server/**.py`` — which is exactly how these
    were missed (operator asked "what else should we check", 2026-09-18).
    """
    fixtures = sorted((_REPO / "tests" / "fixtures" / "delivery").glob("*.golden.json"))
    assert fixtures, "no delivery goldens found — this guard would pass vacuously"

    roots = _router_roots()
    offenders: list[str] = []
    for path in fixtures:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "deep_link" not in line:
                continue
            for root in _DEEP_LINK.findall(line):
                if root.startswith("/api"):
                    continue
                if root not in roots:
                    offenders.append(f"{path.relative_to(_REPO)}:{lineno} -> {root}")

    assert not offenders, (
        "Delivery goldens name paths the app does not define. These are the contract another repo "
        "renders emails from:\n  " + "\n  ".join(offenders) + f"\n\nReal routes: {sorted(roots)}"
    )
