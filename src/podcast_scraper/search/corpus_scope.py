"""Corpus-wide episode identity helpers (GitHub #505 / RFC-063).

Composite ``(feed_id, episode_id)`` scope keys avoid fingerprint and vector row
collisions when multiple feeds share a corpus parent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from podcast_scraper.utils import filesystem


def normalize_feed_id(feed_id: Any) -> Optional[str]:
    """Return stripped feed id string, or None if missing."""
    if isinstance(feed_id, str) and feed_id.strip():
        return feed_id.strip()
    return None


def index_fingerprint_scope_key(feed_id: Optional[str], episode_id: str) -> str:
    """Stable key for ``episode_fingerprints.json`` (one row per scoped episode)."""
    fn = normalize_feed_id(feed_id)
    if fn:
        return f"{fn}\x1f{episode_id}"
    return episode_id


def vector_doc_scope_tag(feed_id: Optional[str], episode_id: str) -> str:
    """Segment embedded in FAISS doc ids for uniqueness across feeds."""
    if not feed_id:
        return episode_id
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in feed_id)[:120]
    return f"{safe}__{episode_id}"


def gi_map_lookup_key_from_vector_meta(meta: dict) -> str:
    """Key into the GI path map built from metadata (matches fingerprint scope)."""
    ep = meta.get("episode_id")
    if not isinstance(ep, str) or not ep:
        return ""
    return index_fingerprint_scope_key(normalize_feed_id(meta.get("feed_id")), ep)


def episode_root_from_metadata_path(metadata_path: Path) -> Path:
    """Episode workspace root: parent of ``metadata/`` (RFC-063 §5)."""
    return metadata_path.parent.parent.resolve()


def discover_metadata_files(output_root: Path) -> List[Path]:
    """List episode metadata files (flat output or corpus parent with ``feeds/``).

    Hybrid layout: if ``feeds/`` exists **and** top-level ``metadata/`` exists, both are
    included (GitHub #505 follow-up).
    """
    patterns = ("*.metadata.json", "*.metadata.yaml", "*.metadata.yml")
    found: List[Path] = []

    def _extend_from_meta_dir(meta_dir: Path) -> None:
        if not meta_dir.is_dir():
            return
        for pat in patterns:
            found.extend(meta_dir.glob(pat))

    if (output_root / "feeds").is_dir():
        for meta_dir in output_root.rglob("metadata"):
            if meta_dir.is_dir():
                _extend_from_meta_dir(meta_dir)
        # Corpus parent may also hold legacy or auxiliary metadata beside feeds/
        _extend_from_meta_dir(output_root / filesystem.METADATA_SUBDIR)
    else:
        _extend_from_meta_dir(output_root / filesystem.METADATA_SUBDIR)

    return sorted({p.resolve() for p in found})
