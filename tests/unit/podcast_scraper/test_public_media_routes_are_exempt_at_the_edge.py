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
``get_current_user``, the player edge must carry a path exemption for it.** Reading both files is
the only place these two decisions — made in different languages, in different directories, by
different reflexes — are put next to each other.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVER = ROOT / "src" / "podcast_scraper" / "server"
ROUTES_DIR = SERVER / "routes"
PLAYER_CADDY = ROOT / "infra" / "caddy" / "player.caddy"

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


def _edge_exempt_paths() -> list[str]:
    """Every path listed on a `path` matcher line in the player vhost."""
    paths: list[str] = []
    for line in PLAYER_CADDY.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped.startswith("path "):
            continue
        paths.extend(stripped.split()[1:])
    return paths


def _matches(edge_pattern: str, route_path: str) -> bool:
    """Does a Caddy `path` glob cover this route, once FastAPI's `{param}` is a path segment?

    Caddy's `*` does not span `/`, and neither does a FastAPI path parameter, so `{person_id}` and
    `*` line up one-for-one. Anchored at both ends, because Caddy's path matcher is exact unless
    the pattern itself ends in a wildcard.
    """
    concrete = re.sub(r"\{[^}]+\}", "*", route_path)
    pattern = "^" + "[^/]+".join(re.escape(part) for part in edge_pattern.split("*")) + "$"
    return re.match(pattern, concrete) is not None


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
    exemptions = _edge_exempt_paths()
    assert any(_matches(p, f"/api/app{route}") for p in exemptions), (
        f"{module}::{func} serves a file and does NOT require a session — it was made public so an "
        f"<img src> could fetch it. But the player edge has no exemption covering "
        f"/api/app{route}, so the coming-soon gate will answer it with HTML and a 200, and the "
        f"image will silently render as initials on web AND native.\n\n"
        f"Add a GET-only `path` matcher for it in infra/caddy/player.caddy beside @entity_media, "
        f"then re-run infra/caddy/validate.sh.\n\n"
        f"Current exemptions: {exemptions}"
    )
