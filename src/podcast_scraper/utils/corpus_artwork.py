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

    if not (os.path.isfile(dest_str) and os.path.getsize(dest_str) > 0):
        try:
            with open(dest_str, "wb") as fh:
                fh.write(body)
        except OSError as exc:
            logger.warning("Could not write artwork %s: %s", dest_str, exc)
            return None
    # The serving API mounts the corpus READ-ONLY, so a downscale it cannot find is never made
    # there: it serves the full image instead (measured on prod, 200 KB for a 320px slot). The
    # writer has write access — make both now. Best effort: a missing one only costs bytes.
    write_thumbnail(Path(corpus_root), dest_str)
    write_medium(Path(corpus_root), dest_str)
    return rel_posix


#: Longest edge of a list/card thumbnail (served by ``GET /api/app/artwork?size=thumb``).
THUMB_MAX_PX = 320

#: Longest edge of the player-sized image (``size=medium``): the hero is ~400 CSS px wide, so
#: 1024 covers a 2.5x screen. Originals run to 3000², ~36 MB once a phone decodes them, and
#: Android's WebView went blank under that load (2026-10-08, Pixel 8, build 1.0.2).
MEDIUM_MAX_PX = 1024


def _derived_path(corpus_root: Path, original_abs: str, kind: str) -> Path:
    stem = os.path.splitext(os.path.basename(original_abs))[0]
    return Path(corpus_root) / CORPUS_ART_REL_PREFIX / "derived" / kind / f"{stem}.jpg"


def _write_derived(corpus_root: Path, original_abs: str, kind: str, max_px: int) -> bool:
    dst = _derived_path(corpus_root, original_abs, kind)
    if dst.is_file():
        return True
    try:
        from PIL import Image

        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(original_abs) as im:
            img = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            img.thumbnail((max_px, max_px))
            tmp = dst.with_name(dst.name + ".tmp")
            img.save(tmp, format="JPEG", quality=85, optimize=True)
        os.replace(tmp, dst)
        return True
    except Exception as exc:  # noqa: BLE001 - undecodable / unwritable -> the original is served
        logger.debug("%s not written for %s: %s", kind, original_abs, exc)
        return False


def thumbnail_path(corpus_root: Path, original_abs: str) -> Path:
    """Where the thumbnail of *original_abs* lives: ``corpus-art/derived/thumb/<stem>.jpg``."""
    return _derived_path(corpus_root, original_abs, "thumb")


def write_thumbnail(corpus_root: Path, original_abs: str) -> bool:
    """Make the thumbnail for *original_abs* if it is missing. ``True`` when it exists after."""
    return _write_derived(corpus_root, original_abs, "thumb", THUMB_MAX_PX)


def medium_path(corpus_root: Path, original_abs: str) -> Path:
    """Where the player-sized copy lives: ``corpus-art/derived/medium/<stem>.jpg``."""
    return _derived_path(corpus_root, original_abs, "medium")


def write_medium(corpus_root: Path, original_abs: str) -> bool:
    """Make the player-sized copy if it is missing (never upscaled). ``True`` when it exists."""
    return _write_derived(corpus_root, original_abs, "medium", MEDIUM_MAX_PX)
