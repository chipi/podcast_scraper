"""An app-layer-public media route is only public if the EDGE agrees.

The defect this exists to stop has now shipped three times, identically:

1. ``serve_avatar`` (#2109) was made unauthenticated so an ``<img src>`` could fetch it — an
   ``<img>`` cannot send an ``Authorization`` header, and the native shell carries its session in
   exactly that header. The edge was not told, so every avatar came back as coming-soon HTML with
   a **200**, the browser failed to decode HTML as an image, and ``ProfileAvatar`` fell back to
   initials. Fixed at the edge on 2026-09-17.
2. ``person_photo`` — same reasoning in its own docstring, same omission. Measured on prod
   2026-09-27: inside ``compose-api-1`` the route returns ``200 image/jpeg 31392``; through the
   public edge the identical path returns ``200 text/html 1278``. 712 hosted portraits, never once
   seen by anyone.
3. ``org_logo`` — same again, found only because we went looking after (2).

A 200 of HTML is the worst available answer here: it is not an error, so nothing logs, nothing
retries, and no alert fires. The failure presents as a *design* choice ("this app shows initials"),
which is why it survived two enrichment waves.

So the invariant is structural, not behavioural: **if a route serves a file and does NOT depend on
``get_current_user``, the player edge must carry a path exemption for it.**

The edge configuration is deployment tooling and is not in this repository. The two decisions meet
in ``config/deploy_contract.json`` instead: this file keeps its ``player_edge_public_file_paths``
EXACTLY equal to what the routes actually do, and the deployment side tests its edge against that
list. A route made public here fails this file until it is added to the contract, and adding it is
what tells the edge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVER = ROOT / "src" / "podcast_scraper" / "server"
ROUTES_DIR = SERVER / "routes"
DEPLOY_CONTRACT = ROOT / "config" / "deploy_contract.json"
_CONTRACT_KEY = "player_edge_public_file_paths"

# `@router.get("<path>")` … `async def name(` … up to the next decorator or EOF.
_ROUTE = re.compile(
    r"@router\.(?P<verb>get)\(\s*\"(?P<path>/[^\"]*)\"(?P<decor>[^)]*)\)\s*"
    r"(?:async\s+)?def\s+(?P<name>\w+)\((?P<sig>.*?)\)\s*->\s*(?P<ret>[^:]+):",
    re.DOTALL,
)


def _player_route_modules() -> list[str]:
    """The modules mounted at ``/api/app`` — i.e. the ones the PLAYER edge actually fronts.

    Read from ``_APP_ROUTES`` in ``app.py`` rather than matched on an ``app_*`` filename, because
    the mount is the thing that matters and the naming is only a convention. The operator plane
    (``corpus_media``, ``jobs``, …) mounts at ``/api`` on the tailnet and public-operator surfaces
    and never passes through ``player.caddy``, so its file routes are correctly out of scope here.
    """
    src = (SERVER / "app.py").read_text(encoding="utf-8")
    block = src[src.index("_APP_ROUTES = (") :]
    block = block[: block.index(")")]
    return [f"{name}.py" for name in re.findall(r"^\s*(\w+),", block, re.MULTILINE)]


def _file_serving_public_routes() -> list[tuple[str, str, str]]:
    """``(module, function, route path)`` for every GET that returns a file without auth."""
    found: list[tuple[str, str, str]] = []
    player_modules = set(_player_route_modules())
    for py in sorted(ROUTES_DIR.glob("*.py")):
        if py.name not in player_modules:
            continue
        src = py.read_text(encoding="utf-8")
        for m in _ROUTE.finditer(src):
            if "FileResponse" not in m.group("ret"):
                continue
            # A route that still demands a session is not public and needs no exemption.
            if "get_current_user" in m.group("sig"):
                continue
            found.append((py.name, m.group("name"), m.group("path")))
    return found


def _contract_paths() -> list[str]:
    paths: list[str] = json.loads(DEPLOY_CONTRACT.read_text(encoding="utf-8"))[_CONTRACT_KEY]
    return paths


def test_the_scan_finds_the_routes_it_is_meant_to_police() -> None:
    """Guard the guard: a regex that silently matches nothing would make this file always pass."""
    assert _player_route_modules(), "could not read _APP_ROUTES out of app.py"
    routes = _file_serving_public_routes()
    names = {name for _, name, _ in routes}
    assert {"serve_avatar", "person_photo", "org_logo"} <= names, (
        "the three known public media routes were not all found — the route regex has drifted "
        f"from the source, so this file is no longer checking anything. Found: {sorted(names)}"
    )


@pytest.mark.parametrize(
    ("module", "func", "route"),
    _file_serving_public_routes(),
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_public_file_route_is_reachable_through_the_player_edge(
    module: str, func: str, route: str
) -> None:
    declared = _contract_paths()
    assert f"/api/app{route}" in declared, (
        f"{module}::{func} serves a file and does NOT require a session — it was made public so an "
        f"<img src> could fetch it. But config/deploy_contract.json does not list /api/app{route} "
        f"under {_CONTRACT_KEY}, so the edge is never told: the coming-soon gate will answer it "
        f"with HTML and a 200, and the image will silently render as initials on web AND "
        f"native.\n\nAdd it to the contract. The deployment's edge configuration is tested "
        f"against that list.\n\n"
        f"Currently declared: {declared}"
    )


def test_the_contract_lists_no_route_that_is_not_public() -> None:
    """The other direction: a stale entry would keep an edge exemption open for a route that now
    requires a session, or no longer exists."""
    actual = {f"/api/app{route}" for _, _, route in _file_serving_public_routes()}
    stale = sorted(set(_contract_paths()) - actual)
    assert not stale, (
        f"config/deploy_contract.json lists {stale} under {_CONTRACT_KEY}, but no unauthenticated "
        f"file-serving player route matches. Remove them from the contract."
    )
