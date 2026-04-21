"""Download podcast feed/episode artwork into the corpus tree.

Files live under ``<corpus>/.podcast_scraper/corpus-art/sha256/…`` and are served by
``GET /api/corpus/binary`` (allowlisted subtree only).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from podcast_scraper.rss.downloader import http_get
from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

logger = logging.getLogger(__name__)

# Relative to corpus / output root (POSIX).
CORPUS_ART_REL_PREFIX = ".podcast_scraper/corpus-art"

# Podcast cover images; reject HTML error pages and huge responses.
_MAX_ARTWORK_BYTES = 8 * 1024 * 1024


def _guess_extension(content_type: str, url: str) -> str:
    ct = (content_type or "").lower().split(";")[0].strip()
    if "jpeg" in ct or ct == "image/jpg":
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".bin"


def _is_probably_image(body_head: bytes, content_type: str) -> bool:
    if len(body_head) < 3:
        return False
    if body_head.startswith(b"\xff\xd8\xff"):
        return True
    if body_head.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if len(body_head) >= 12 and body_head.startswith(b"RIFF") and body_head[8:12] == b"WEBP":
        return True
    if body_head.startswith(b"GIF87a") or body_head.startswith(b"GIF89a"):
        return True
    ct = (content_type or "").lower()
    return ct.startswith("image/")


def download_podcast_artwork(
    url: str,
    corpus_root: Path,
    *,
    user_agent: str,
    timeout: int,
    max_bytes: int = _MAX_ARTWORK_BYTES,
) -> Optional[str]:
    """Download image bytes into the corpus art store; return POSIX relpath or None."""

    normalized = (url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return None

    body, ctype = http_get(normalized, user_agent, timeout)
    if not body or len(body) > max_bytes:
        if body and len(body) > max_bytes:
            logger.debug("Artwork too large (%d bytes), skipping", len(body))
        return None

    if not _is_probably_image(body[: min(len(body), 32)], ctype or ""):
        logger.debug("Response does not look like an image, skipping artwork")
        return None

    digest = hashlib.sha256(body).hexdigest()
    ext = _guess_extension(ctype or "", normalized)
    rel_dir = f"{CORPUS_ART_REL_PREFIX}/sha256/{digest[:2]}/{digest[2:4]}"
    fname = f"{digest}{ext}"
    rel_posix = f"{rel_dir}/{fname}".replace("\\", "/")
    dest_str = safe_relpath_under_corpus_root(corpus_root, rel_posix)
    if not dest_str:
        return None
    dest_parent = os.path.dirname(dest_str)
    try:
        os.makedirs(dest_parent, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create artwork dir %s: %s", dest_parent, exc)
        return None

    if os.path.isfile(dest_str) and os.path.getsize(dest_str) > 0:
        return rel_posix

    try:
        with open(dest_str, "wb") as fh:
            fh.write(body)
    except OSError as exc:
        logger.warning("Could not write artwork %s: %s", dest_str, exc)
        return None
    return rel_posix
