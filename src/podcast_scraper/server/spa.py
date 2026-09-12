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

import html
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import HTMLResponse, Response
from starlette.staticfiles import StaticFiles

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

    def _index_html(self) -> str:
        if self._index_cache is None:
            # StaticFiles always has a directory here — we only ever construct it with one.
            directory = self.directory
            assert directory is not None
            self._index_cache = (Path(directory) / "index.html").read_text(encoding="utf-8")
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
        last = req_path.rsplit("/", 1)[-1]
        # A missing asset (has a file extension) stays a 404 — only document routes fall back.
        if "." in last:
            raise StarletteHTTPException(status_code=404)

        document = self._document_for(req_path, scope)
        return HTMLResponse(document)

    def _document_for(self, req_path: str, scope: Any) -> str:
        """index.html, with OG tags injected for a shareable entity path (else the shell as-is)."""
        index = self._index_html()
        target = self._entity_target(req_path)
        if target is None:
            return index
        kind, ident = target
        root = self._corpus_root(scope)
        if root is None:
            return index
        from podcast_scraper.server.og.build import build_og_meta

        try:
            meta = build_og_meta(root, kind, ident)
        except Exception:  # noqa: BLE001 - never break the document on a corpus quirk
            meta = None
        if meta is None:
            return index
        image = f"{self._origin(scope)}/og/{kind}/{quote(ident, safe='')}.png"
        return self._inject(index, meta.title, meta.description, image)

    @staticmethod
    def _entity_target(req_path: str) -> tuple[str, str] | None:
        parts = req_path.strip("/").split("/")
        if len(parts) != 2 or parts[0] not in _PATH_KIND:
            return None
        ident = unquote(parts[1]).strip()
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
    def _inject(index: str, title: str, description: str, image: str) -> str:
        t = html.escape(title, quote=True)
        d = html.escape(description, quote=True)
        i = html.escape(image, quote=True)
        tags = (
            f'<meta property="og:type" content="article">'
            f'<meta property="og:title" content="{t}">'
            f'<meta property="og:description" content="{d}">'
            f'<meta property="og:image" content="{i}">'
            f'<meta name="twitter:card" content="summary_large_image">'
            f'<meta name="twitter:title" content="{t}">'
            f'<meta name="twitter:description" content="{d}">'
            f'<meta name="twitter:image" content="{i}">'
        )
        if "</head>" in index:
            return index.replace("</head>", tags + "</head>", 1)
        return tags + index
