"""SPA static serving with per-entity OG-tag injection (#2036).

Replaces the bare ``StaticFiles(html=True)`` catch-all with three behaviours:

  1. Real files (JS/CSS/fonts/icons/…) serve exactly as before.
  2. A client-routed document that maps to no file falls back to ``index.html`` — the SPA
     history-mode fallback the bare mount lacked (a hard refresh on ``/topic/x`` used to 404 at the
     backend), so deep links now load.
  3. For a shareable ENTITY document (``/topic|/person|/episode|/podcast|/storyline/{id}``), the
     ``index.html`` is served with ``og:*`` / ``twitter:*`` tags injected into the head, pointing
     ``og:image`` at the ``/og/{kind}/{id}.png`` card — so a shared LINK unfurls as the card.

A genuinely missing ASSET (a path whose last segment has a file extension) still 404s — only
extension-less document routes fall back, so a broken ``/assets/x.js`` never masquerades as HTML.
"""

from __future__ import annotations

import asyncio
import html
from pathlib import Path
from typing import Any
from urllib.parse import quote

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import HTMLResponse, Response
from starlette.staticfiles import StaticFiles

# Paths that must hard-404 rather than fall back to the SPA shell — a mistyped API/internal/OG path
# returning 200 HTML would mask the error at the client's JSON parse.
_NON_SPA_PREFIXES = ("/api/", "/internal/", "/og/")

# SPA document route prefix → OG card kind. Organization is intentionally absent — it has no
# standalone page in the player (overlay-only), so there is no org link to unfurl.
_PATH_KIND = {
    "topic": "topic",
    "person": "person",
    "episode": "episode",
    "podcast": "show",
    "storyline": "storyline",
}


class SpaStaticFiles(StaticFiles):
    """Static files + SPA fallback + OG injection for entity documents."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._index_cache: str | None = None

    def _index_html(self) -> str | None:
        if self._index_cache is None:
            directory = self.directory
            if directory is None:
                return None
            try:
                self._index_cache = (Path(directory) / "index.html").read_text(encoding="utf-8")
            except OSError:
                return None  # missing index (misdeploy) → caller 404s rather than 500s
        return self._index_cache

    async def get_response(self, path: str, scope: Any) -> Response:
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            response = None
        if response is not None and response.status_code != 404:
            return response

        req_path = scope.get("path", "/")
        # API / internal / OG paths hard-404 (never the SPA shell); a missing asset (has a file
        # extension) also stays a 404 — only extension-less document routes fall back.
        if req_path.startswith(_NON_SPA_PREFIXES) or "." in req_path.rsplit("/", 1)[-1]:
            raise StarletteHTTPException(status_code=404)
        index = self._index_html()
        if index is None:
            raise StarletteHTTPException(status_code=404)
        return HTMLResponse(await self._document_for(index, req_path, scope))

    async def _document_for(self, index: str, req_path: str, scope: Any) -> str:
        """index.html, with OG tags injected for a shareable entity path (else the shell as-is)."""
        target = self._entity_target(req_path)
        if target is None:
            return index
        kind, ident = target
        root = self._corpus_root(scope)
        if root is None:
            return index
        from podcast_scraper.server.og.build import build_og_meta

        try:
            # Off the event loop — the meta build reads KG artifacts (same reason the /og route uses
            # a thread); text-only (build_og_meta skips artwork bytes) so it stays cheap.
            meta = await asyncio.to_thread(build_og_meta, root, kind, ident)
        except Exception:  # noqa: BLE001 - never break the document on a corpus quirk
            meta = None
        if meta is None:
            return index
        origin = self._origin(scope)
        image = f"{origin}/og/{kind}/{quote(ident, safe='')}.png"
        return self._inject(index, meta.title, meta.description, image, f"{origin}{req_path}")

    @staticmethod
    def _entity_target(req_path: str) -> tuple[str, str] | None:
        # scope["path"] is already percent-decoded by the ASGI server — do NOT unquote again.
        parts = req_path.strip("/").split("/")
        if len(parts) != 2 or parts[0] not in _PATH_KIND:
            return None
        ident = parts[1].strip()
        return (_PATH_KIND[parts[0]], ident) if ident else None

    @staticmethod
    def _corpus_root(scope: Any) -> Path | None:
        app = scope.get("app")
        anchor = getattr(getattr(app, "state", None), "output_dir", None)
        return Path(anchor) if anchor else None

    @staticmethod
    def _origin(scope: Any) -> str:
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        host = headers.get(b"host", b"").decode() or "closelistening.app"
        proto = headers.get(b"x-forwarded-proto", b"").decode() or scope.get("scheme", "https")
        # Defend the injected og:image URL against a crafted Host/proto (CR/LF, stray whitespace) —
        # take only the first token before any newline.
        host = host.splitlines()[0].strip() or "closelistening.app"
        proto = (proto.splitlines()[0].strip() or "https") if proto else "https"
        return f"{proto}://{host}"

    @staticmethod
    def _inject(index: str, title: str, description: str, image: str, page_url: str) -> str:
        t = html.escape(title, quote=True)
        d = html.escape(description, quote=True)
        i = html.escape(image, quote=True)
        u = html.escape(page_url, quote=True)
        tags = (
            f'<meta property="og:type" content="article">'
            f'<meta property="og:title" content="{t}">'
            f'<meta property="og:description" content="{d}">'
            f'<meta property="og:image" content="{i}">'
            f'<meta property="og:image:width" content="1080">'
            f'<meta property="og:image:height" content="1440">'
            f'<meta property="og:url" content="{u}">'
            f'<meta name="twitter:card" content="summary_large_image">'
            f'<meta name="twitter:title" content="{t}">'
            f'<meta name="twitter:description" content="{d}">'
            f'<meta name="twitter:image" content="{i}">'
        )
        if "</head>" in index:
            return index.replace("</head>", tags + "</head>", 1)
        return tags + index
