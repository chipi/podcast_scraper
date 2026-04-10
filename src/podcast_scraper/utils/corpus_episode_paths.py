"""Resolve per-episode GI/KG artifact paths from a corpus root (multi-feed aware).

Walks episode metadata (``discover_metadata_files``) and maps to sibling ``.gi.json`` /
``.kg.json`` paths. Falls back to scanning artifact files when metadata is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from podcast_scraper.search.corpus_scope import discover_metadata_files, normalize_feed_id
from podcast_scraper.utils.path_validation import safe_resolve_directory

ArtifactKind = Literal["gi", "kg"]


def _load_loose_metadata(meta_path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = meta_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        lower = meta_path.name.lower()
        if lower.endswith((".yaml", ".yml")):
            import yaml

            raw = yaml.safe_load(text)
            return raw if isinstance(raw, dict) else None
        blob = json.loads(text)
        return blob if isinstance(blob, dict) else None
    except Exception:
        return None


def _feed_and_episode_ids(doc: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    if not doc:
        return None, None
    feed = doc.get("feed")
    episode = doc.get("episode")
    fid: Any = feed.get("feed_id") if isinstance(feed, dict) else None
    eid: Any = episode.get("episode_id") if isinstance(episode, dict) else None
    return normalize_feed_id(fid), eid.strip() if isinstance(eid, str) and eid.strip() else None


def _artifact_path_from_metadata_str(metadata_path_str: str, kind: ArtifactKind) -> Path:
    mp = metadata_path_str
    suffix = ".gi.json" if kind == "gi" else ".kg.json"
    if mp.endswith(".metadata.json"):
        return Path(mp.replace(".metadata.json", suffix))
    if mp.endswith(".metadata.yaml"):
        return Path(mp.replace(".metadata.yaml", suffix))
    if mp.endswith(".metadata.yml"):
        return Path(mp.replace(".metadata.yml", suffix))
    return Path(f"{os.path.splitext(mp)[0]}{suffix}")


def _sibling_metadata_candidates(artifact_path: Path, kind: ArtifactKind) -> List[Path]:
    parent = artifact_path.parent
    name = artifact_path.name
    ext = ".gi.json" if kind == "gi" else ".kg.json"
    if not name.endswith(ext):
        return []
    base = name[: -len(ext)]
    return [
        parent / f"{base}.metadata.json",
        parent / f"{base}.metadata.yaml",
        parent / f"{base}.metadata.yml",
    ]


def _feed_id_from_sibling_metadata(artifact_path: Path, kind: ArtifactKind) -> Optional[str]:
    for meta in _sibling_metadata_candidates(artifact_path, kind):
        doc = _load_loose_metadata(meta)
        fid, _eid = _feed_and_episode_ids(doc)
        if fid:
            return fid
    return None


def _episode_id_from_artifact_file(path: Path) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    eid = data.get("episode_id")
    return eid if isinstance(eid, str) else None


def _fallback_rglob_candidates(
    output_root: Path,
    episode_id: str,
    want_feed: Optional[str],
    kind: ArtifactKind,
) -> List[Path]:
    pattern = "**/*.gi.json" if kind == "gi" else "**/*.kg.json"
    found: List[Path] = []
    for path in sorted(output_root.glob(pattern)):
        if not path.is_file():
            continue
        if _episode_id_from_artifact_file(path) != episode_id:
            continue
        if want_feed is not None:
            sf = _feed_id_from_sibling_metadata(path, kind)
            if normalize_feed_id(sf) != want_feed:
                continue
        found.append(path.resolve())
    return found


def list_artifact_paths_for_episode(
    output_root: Path,
    episode_id: str,
    *,
    feed_id: Optional[str] = None,
    kind: ArtifactKind,
) -> List[Path]:
    """Return resolved paths to ``.gi.json`` or ``.kg.json`` for ``episode_id``.

    Uses episode metadata under the corpus (including ``feeds/`` layout). When
    ``feed_id`` is set, only that feed's episode matches. When multiple matches
    exist and ``feed_id`` is omitted, returns all matches (caller may treat as
    ambiguous for single-path APIs).

    Args:
        output_root: Pipeline or corpus parent directory.
        episode_id: RSS-aligned episode id from metadata / artifact.
        feed_id: Optional feed scope (normalized string equality).
        kind: ``gi`` or ``kg``.

    Returns:
        Sorted unique paths that exist on disk.
    """
    want_feed = normalize_feed_id(feed_id)
    out: List[Path] = []
    for meta_path in discover_metadata_files(output_root):
        doc = _load_loose_metadata(meta_path)
        fid, eid = _feed_and_episode_ids(doc)
        if eid != episode_id:
            continue
        if want_feed is not None and fid != want_feed:
            continue
        rel = _artifact_path_from_metadata_str(str(meta_path), kind)
        if rel.is_file():
            out.append(rel.resolve())
    uniq_sorted = sorted({p.resolve() for p in out})
    if uniq_sorted:
        return uniq_sorted
    return _fallback_rglob_candidates(output_root, episode_id, want_feed, kind)


def pick_single_artifact_path(
    paths: List[Path],
) -> Optional[Path]:
    """Return the only path if unambiguous, else ``None``."""
    if len(paths) == 1:
        return paths[0]
    return None


def corpus_search_parent_hint(listed_root: Path) -> List[str]:
    """If ``listed_root`` is not where the unified FAISS index lives, suggest parent.

    Returns human-readable hint strings (empty when none).
    """
    try:
        from podcast_scraper.search.faiss_store import VECTORS_FILE
    except ImportError:
        return []

    root = safe_resolve_directory(listed_root)
    if root is None:
        return []
    # CodeQL py/path-injection sanitiser: normpath then startswith(os.sep).
    root_normed = os.path.normpath(str(root))
    if not root_normed.startswith(os.sep):
        return []

    vec_str = os.path.normpath(os.path.join(root_normed, "search", VECTORS_FILE))
    if vec_str.startswith(os.sep) and os.path.isfile(vec_str):
        return []

    hints: List[str] = []
    for ancestor in root.parents:
        ancestor_normed = os.path.normpath(str(ancestor))
        if not ancestor_normed.startswith(os.sep):
            continue
        idx_str = os.path.normpath(os.path.join(ancestor_normed, "search", VECTORS_FILE))
        if not idx_str.startswith(os.sep):
            continue
        if not os.path.isfile(idx_str):
            continue
        try:
            root.relative_to(ancestor)
        except ValueError:
            break
        if ancestor != root:
            hints.append(
                f"Unified semantic index is under {ancestor}. Set corpus root to that directory "
                "for search and index stats (multi-feed layout)."
            )
        break
    return hints
