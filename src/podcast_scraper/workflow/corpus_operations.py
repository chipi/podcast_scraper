"""Multi-feed corpus artifacts and status (GitHub #506 / RFC-063 §7)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from podcast_scraper import __version__
from podcast_scraper.utils import filesystem

if TYPE_CHECKING:
    from podcast_scraper import config

logger = logging.getLogger(__name__)

CORPUS_MANIFEST_FILE = "corpus_manifest.json"
CORPUS_RUN_SUMMARY_FILE = "corpus_run_summary.json"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_iso_now() -> str:
    """Public UTC timestamp for per-feed batch rows (ISO 8601 with Z)."""
    return _utc_iso()


@dataclass
class MultiFeedFeedResult:
    """One feed's outcome from a multi-feed batch."""

    feed_url: str
    ok: bool
    error: Optional[str]
    episodes_processed: int
    finished_at: Optional[str] = None


def write_corpus_manifest(
    corpus_parent: str,
    feed_results: List[MultiFeedFeedResult],
) -> None:
    """Write ``corpus_manifest.json`` at the corpus parent (GitHub #506)."""
    parent = Path(filesystem.validate_and_normalize_output_dir(corpus_parent))
    feeds_out: List[Dict[str, Any]] = []
    for fr in feed_results:
        sub = filesystem.feed_workspace_dirname(fr.feed_url)
        feeds_out.append(
            {
                "feed_url": fr.feed_url,
                "stable_feed_dir": sub,
                "last_run_finished_at": fr.finished_at or _utc_iso(),
                "ok": fr.ok,
                "error": fr.error,
                "episodes_processed": fr.episodes_processed,
            }
        )
    doc = {
        "schema_version": "1.0.0",
        "tool_version": __version__,
        "corpus_parent": str(parent),
        "updated_at": _utc_iso(),
        "feeds": feeds_out,
    }
    path = parent / CORPUS_MANIFEST_FILE
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote corpus manifest: %s", path)


def build_corpus_run_summary_document(
    corpus_parent_resolved: str,
    feed_results: List[MultiFeedFeedResult],
    *,
    overall_ok: bool,
    batch_finished_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the ``corpus_run_summary.json`` payload (normative shape; GitHub #506)."""
    fin = batch_finished_at or _utc_iso()
    feeds_json: List[Dict[str, Any]] = []
    for fr in feed_results:
        row: Dict[str, Any] = {
            "feed_url": fr.feed_url,
            "ok": fr.ok,
            "error": fr.error,
            "episodes_processed": fr.episodes_processed,
        }
        if fr.finished_at:
            row["finished_at"] = fr.finished_at
        feeds_json.append(row)
    return {
        "schema_version": "1.0.0",
        "corpus_parent": corpus_parent_resolved,
        "finished_at": fin,
        "overall_ok": overall_ok,
        "feeds": feeds_json,
    }


def write_corpus_run_summary(
    corpus_parent: str,
    feed_results: List[MultiFeedFeedResult],
    *,
    overall_ok: bool,
) -> Dict[str, Any]:
    """Write machine-readable batch summary JSON (GitHub #506). Returns the written document."""
    parent = Path(filesystem.validate_and_normalize_output_dir(corpus_parent))
    doc = build_corpus_run_summary_document(str(parent), feed_results, overall_ok=overall_ok)
    path = parent / CORPUS_RUN_SUMMARY_FILE
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote corpus run summary: %s", path)
    logger.info(
        "corpus_multi_feed_summary: %s", json.dumps(doc, ensure_ascii=False, sort_keys=True)
    )
    return doc


def log_multi_feed_summary_structured(feed_results: List[MultiFeedFeedResult]) -> None:
    """Emit a single structured log line for automation (GitHub #506)."""
    rows: List[Dict[str, Any]] = []
    for fr in feed_results:
        r: Dict[str, Any] = {
            "feed_url": fr.feed_url,
            "ok": fr.ok,
            "error": fr.error,
            "episodes_processed": fr.episodes_processed,
        }
        if fr.finished_at:
            r["finished_at"] = fr.finished_at
        rows.append(r)
    payload = {"feeds": rows}
    logger.info("multi_feed_batch: %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))


def collect_corpus_status(corpus_parent: str) -> Dict[str, Any]:
    """Inspect a corpus tree without network (GitHub #506)."""
    parent = Path(filesystem.validate_and_normalize_output_dir(corpus_parent))
    manifest_path = parent / CORPUS_MANIFEST_FILE
    manifest: Optional[Dict[str, Any]] = None
    if manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = raw if isinstance(raw, dict) else None
        except json.JSONDecodeError:
            manifest = None

    feeds_dir = parent / "feeds"
    per_feed: List[Dict[str, Any]] = []
    if feeds_dir.is_dir():
        for child in sorted(feeds_dir.iterdir()):
            if not child.is_dir():
                continue
            meta_dir = child / filesystem.METADATA_SUBDIR
            n_meta = 0
            if meta_dir.is_dir():
                n_meta = len(
                    list(meta_dir.glob("*.metadata.json"))
                    + list(meta_dir.glob("*.metadata.yaml"))
                    + list(meta_dir.glob("*.metadata.yml"))
                )
            idx_path = child / "index.json"
            idx_err: Optional[str] = None
            n_idx_failed = 0
            if idx_path.is_file():
                try:
                    idx = json.loads(idx_path.read_text(encoding="utf-8"))
                    if isinstance(idx, dict):
                        for ep in idx.get("episodes") or []:
                            if isinstance(ep, dict) and ep.get("status") == "failed":
                                n_idx_failed += 1
                        if n_idx_failed and isinstance(idx.get("episodes"), list):
                            for ep in idx["episodes"]:
                                if (
                                    isinstance(ep, dict)
                                    and ep.get("status") == "failed"
                                    and ep.get("error_message")
                                ):
                                    idx_err = str(ep.get("error_message"))[:200]
                                    break
                except json.JSONDecodeError:
                    idx_err = "index.json: invalid JSON"

            per_feed.append(
                {
                    "dir": child.name,
                    "metadata_files": n_meta,
                    "index_json_present": idx_path.is_file(),
                    "index_failed_episodes": n_idx_failed,
                    "sample_index_error": idx_err,
                }
            )

    search_dir = parent / "search"
    vectors = search_dir / "vectors.faiss"
    index_meta: Optional[Dict[str, Any]] = None
    meta_file = search_dir / "index_meta.json"
    if meta_file.is_file():
        try:
            blob = json.loads(meta_file.read_text(encoding="utf-8"))
            index_meta = blob if isinstance(blob, dict) else None
        except json.JSONDecodeError:
            index_meta = None

    return {
        "corpus_parent": str(parent),
        "manifest_present": manifest is not None,
        "manifest_schema": (manifest or {}).get("schema_version"),
        "feeds_subdirs": per_feed,
        "search_index_present": vectors.is_file(),
        "search_embedding_model": (index_meta or {}).get("embedding_model"),
        "search_index_kind": (index_meta or {}).get("index_kind"),
    }


def format_corpus_status_text(status: Dict[str, Any]) -> str:
    """Human-readable corpus status for CLI."""
    lines = [
        f"Corpus parent: {status['corpus_parent']}",
        f"Manifest: {'yes' if status['manifest_present'] else 'no'}",
    ]
    if status.get("manifest_schema"):
        lines.append(f"Manifest schema: {status['manifest_schema']}")
    lines.append(f"Unified search index: {'yes' if status['search_index_present'] else 'no'}")
    if status.get("search_embedding_model"):
        lines.append(f"Search embedding model: {status['search_embedding_model']}")
    if status.get("search_index_kind"):
        lines.append(f"Search index kind: {status['search_index_kind']}")
    lines.append("Per-feed directories:")
    for row in status.get("feeds_subdirs") or []:
        lines.append(
            f"  - {row['dir']}: metadata_files={row['metadata_files']} "
            f"index_json={row['index_json_present']} "
            f"failed_episodes={row['index_failed_episodes']}"
        )
        if row.get("sample_index_error"):
            lines.append(f"      sample_error: {row['sample_index_error']}")
    return "\n".join(lines) + "\n"


def finalize_multi_feed_batch(
    corpus_parent: str,
    template_cfg: "config.Config",
    feed_results: List[MultiFeedFeedResult],
) -> Dict[str, Any]:
    """Write manifest/summary logs and optionally build parent vector index (#505/#506).

    **Partial batches:** Manifest and summary are written even when some feeds failed.
    When ``vector_search`` and FAISS are enabled, ``index_corpus`` still runs on the
    corpus parent so completed feeds contribute to the unified index; failed feeds
    simply omit metadata until a later successful run.

    Returns:
        The ``corpus_run_summary.json`` document (``feeds`` may be empty when no rows were
        recorded).
    """
    overall_ok = all(fr.ok for fr in feed_results) if feed_results else True
    write_corpus_manifest(corpus_parent, feed_results)
    summary_doc = write_corpus_run_summary(corpus_parent, feed_results, overall_ok=overall_ok)
    log_multi_feed_summary_structured(feed_results)

    if not feed_results:
        return summary_doc
    if template_cfg.vector_search is not True:
        return summary_doc
    if getattr(template_cfg, "vector_backend", "faiss") != "faiss":
        logger.warning("Skipping parent corpus index: vector_backend is not faiss")
        return summary_doc

    from podcast_scraper.search.indexer import index_corpus

    first_url = feed_results[0].feed_url or template_cfg.rss_url
    idx_cfg = template_cfg.model_copy(
        update={
            "output_dir": corpus_parent,
            "rss_url": first_url,
            "rss_urls": None,
            "skip_auto_vector_index": False,
        }
    )
    try:
        index_corpus(corpus_parent, idx_cfg)
    except Exception as exc:
        logger.warning("Parent corpus vector index failed (non-fatal): %s", exc)
    return summary_doc
