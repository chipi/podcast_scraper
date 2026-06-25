"""Consumer artwork serving + URL helpers (#1078 follow-up, RFC-099).

The consumer app serves the podcast art **we already downloaded at ingest** (into the
corpus-art store) — it never re-fetches graphics from the origin host at request time. This
is the deliberate counterpart to *bridge-never-rehost* for audio: audio is the large,
licensed product we stream from origin every play; cover art is a small promotional asset
that every player caches, so we store it once and serve our copy, cached hard.

Two sizes, both derived from the local original (we downscale, never upscale):

- ``large`` — the original bytes. Podcast cover art is ≥1400² at the source (Apple spec), so
  the original is big enough for the Player's hero zone.
- ``thumb`` — a ≤320px downscale for list density, generated on first request and cached
  under ``corpus-art/derived/thumb/`` (content-addressed → immutable). Falls back to the
  original if Pillow/resize is unavailable or the source can't be decoded.

URLs are content-addressed (sha256 store), so responses carry ``immutable`` cache headers
and the browser + PWA service worker keep them on-device after the first fetch.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import quote

from podcast_scraper.utils.corpus_artwork import CORPUS_ART_REL_PREFIX
from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

logger = logging.getLogger(__name__)

THUMB_MAX_PX = 320
_ART_PREFIX = f"{CORPUS_ART_REL_PREFIX}/"


def artwork_url(relpath: str | None, size: str = "large") -> str | None:
    """Build the consumer artwork URL for a corpus-relative art path, or ``None``.

    ``size`` is ``thumb`` (lists/cards) or ``large`` (player). Returns ``None`` when there is
    no local art, so callers fall back to the remote feed image URL.
    """
    if not relpath or not str(relpath).strip():
        return None
    return f"/api/app/artwork?ref={quote(str(relpath), safe='')}&size={size}"


def safe_artwork_target(corpus_root: Path, relpath: str) -> str | None:
    """Resolve ``relpath`` to an absolute path inside the corpus-art store, or ``None``.

    ``None`` means the path is not under the allowlisted artwork subtree (caller → 400).
    """
    norm = (relpath or "").strip().replace("\\", "/").lstrip("/")
    if norm == CORPUS_ART_REL_PREFIX or not norm.startswith(_ART_PREFIX):
        return None
    safe = safe_relpath_under_corpus_root(corpus_root, norm)
    return os.path.normpath(safe) if safe else None


def _thumb_target(corpus_root: Path, original_abs: str) -> str:
    """Derived-cache path for the thumbnail of ``original_abs`` (always under derived/thumb)."""
    stem = os.path.splitext(os.path.basename(original_abs))[0]
    dst = corpus_root / CORPUS_ART_REL_PREFIX / "derived" / "thumb" / f"{stem}.jpg"
    return os.path.normpath(str(dst))


def ensure_thumbnail(corpus_root: Path, original_abs: str) -> tuple[str, str]:
    """Return ``(path, media_type)`` for the thumbnail; generate+cache it on first request.

    Falls back to ``(original_abs, guessed_type)`` when Pillow is missing, the image can't be
    decoded, or the derived cache can't be written (e.g. a read-only corpus).
    """
    dst = _thumb_target(corpus_root, original_abs)
    if os.path.isfile(dst):
        return dst, "image/jpeg"
    try:
        from PIL import Image

        with Image.open(original_abs) as im:
            img = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            img.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            img.save(dst, format="JPEG", quality=85, optimize=True)
        return dst, "image/jpeg"
    except Exception as exc:  # noqa: BLE001 - any decode/write failure → serve the original
        logger.debug("thumbnail generation failed for %s: %s", original_abs, exc)
        import mimetypes

        media_type, _ = mimetypes.guess_type(os.path.basename(original_abs))
        return original_abs, media_type or "application/octet-stream"
