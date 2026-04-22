"""Multi-feed corpus artifacts and status (GitHub #506)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from podcast_scraper import __version__
from podcast_scraper.utils import filesystem
from podcast_scraper.utils.audio_payload_limits import is_provider_audio_payload_limit_error

if TYPE_CHECKING:
    from podcast_scraper import config

logger = logging.getLogger(__name__)

CORPUS_MANIFEST_FILE = "corpus_manifest.json"
CORPUS_MANIFEST_SCHEMA_VERSION = "1.1.0"  # bumped from 1.0.0 in #650 — add cost_rollup
CORPUS_RUN_SUMMARY_FILE = "corpus_run_summary.json"
CORPUS_RUN_SUMMARY_SCHEMA_VERSION = "1.2.0"  # bumped from 1.1.0 in #650 — add cost_rollup
DEFAULT_CORPUS_INCIDENTS_BASENAME = "corpus_incidents.jsonl"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_iso_now() -> str:
    """Public UTC timestamp for per-feed batch rows (ISO 8601 with Z)."""
    return _utc_iso()


MultiFeedFailureKind = Literal["soft", "hard"]


def _default_corpus_incident_log_path(corpus_parent: str) -> str:
    return str(
        Path(filesystem.validate_and_normalize_output_dir(corpus_parent))
        / DEFAULT_CORPUS_INCIDENTS_BASENAME
    )


def _incident_episode_unique_key(obj: Dict[str, Any]) -> tuple[str, str, int]:
    """Stable key for episode-scoped incident rows (dedupe duplicate lines)."""
    fu = str(obj.get("feed_url") or "")
    eid = obj.get("episode_id")
    if eid is not None and str(eid).strip():
        return (fu, str(eid), -1)
    idx = obj.get("episode_idx")
    try:
        idx_i = int(idx) if idx is not None else -1
    except (TypeError, ValueError):
        idx_i = -1
    return (fu, "", idx_i)


def parse_corpus_incident_jsonl_window(
    log_path: str, start_offset_bytes: int
) -> tuple[list[Dict[str, Any]], int]:
    """Load incident records appended after ``start_offset_bytes`` (multi-feed batch window).

    When ``start_offset_bytes`` is not zero and does not immediately follow a line break, bytes
    through the next newline in the tail are skipped so we never decode a partial UTF-8 line from
    a mid-file offset. Offsets aligned on a line boundary parse from that byte onward.

    Returns:
        ``(records, file_size_bytes)`` — ``file_size_bytes`` is ``os.path.getsize`` when the file
        exists, else ``start_offset_bytes``.
    """
    p = Path(log_path)
    if not p.is_file():
        return [], int(start_offset_bytes)
    end_size = int(p.stat().st_size)
    raw = p.read_bytes()
    if start_offset_bytes <= 0:
        chunk = raw
    else:
        if start_offset_bytes >= len(raw):
            return [], end_size
        chunk = raw[start_offset_bytes:]
        # Only skip through the next newline when the offset is mid-line (not right after \n).
        # Otherwise an offset saved at a line boundary would drop the following complete line.
        if not (
            start_offset_bytes > 0 and raw[start_offset_bytes - 1 : start_offset_bytes] == b"\n"
        ):
            pos = chunk.find(b"\n")
            if pos == -1:
                return [], end_size
            chunk = chunk[pos + 1 :]
    out: list[Dict[str, Any]] = []
    for line in chunk.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out, end_size


def rollup_corpus_incidents_for_multi_feed_summary(
    records: list[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    """Aggregate incident rows into batch totals and per-feed episode counts.

    Returns:
        ``(batch_incidents_payload, per_feed_episode_incidents_unique)`` where the second map is
        ``feed_url -> {policy, soft, hard}`` counts of **distinct** episode keys.
    """
    ep_sets: Dict[str, set[tuple[str, str, int]]] = {
        "policy": set(),
        "soft": set(),
        "hard": set(),
    }
    feed_sets: Dict[str, set[str]] = {
        "policy": set(),
        "soft": set(),
        "hard": set(),
    }
    per_feed_ep: Dict[str, Dict[str, set[tuple[str, str, int]]]] = {}

    for r in records:
        scope = r.get("scope")
        cat = r.get("category")
        if cat not in ("policy", "soft", "hard"):
            continue
        cstr = str(cat)
        fu = str(r.get("feed_url") or "")
        if scope == "episode":
            key = _incident_episode_unique_key(r)
            ep_sets[cstr].add(key)
            if fu not in per_feed_ep:
                per_feed_ep[fu] = {"policy": set(), "soft": set(), "hard": set()}
            per_feed_ep[fu][cstr].add(key)
        elif scope == "feed":
            feed_sets[cstr].add(fu or "__unknown_feed__")

    def _count_map(d: Dict[str, set[Any]]) -> Dict[str, int]:
        return {k: len(v) for k, v in d.items()}

    batch = {
        "episode_incidents_unique": _count_map(ep_sets),
        "feed_incidents_unique": _count_map(feed_sets),
    }
    per_feed_counts: Dict[str, Dict[str, int]] = {}
    for url, buckets in per_feed_ep.items():
        per_feed_counts[url] = _count_map(buckets)
    return batch, per_feed_counts


def classify_multi_feed_feed_exception(exc: BaseException) -> MultiFeedFailureKind:
    """Classify a per-feed pipeline exception for multi-feed exit policy (GitHub #559).

    **Soft** failures are expected operational noise (bad RSS URL, oversize Whisper
    payload, subprocess decode quirks) where other feeds may still succeed.

    **Hard** failures are everything else (bugs, unexpected errors, invalid state).

    Args:
        exc: Exception raised from ``workflow.run_pipeline`` for one feed.

    Returns:
        ``\"soft\"`` or ``\"hard\"``.
    """
    if isinstance(exc, UnicodeDecodeError):
        return "soft"
    if isinstance(exc, ValueError):
        msg = str(exc)
        if "Failed to fetch RSS feed" in msg or "Failed to parse RSS XML" in msg:
            return "soft"
        return "hard"
    type_name = type(exc).__name__
    if type_name == "ProviderRuntimeError" and is_provider_audio_payload_limit_error(exc):
        return "soft"
    return "hard"


@dataclass
class MultiFeedFeedResult:
    """One feed's outcome from a multi-feed batch."""

    feed_url: str
    ok: bool
    error: Optional[str]
    episodes_processed: int
    finished_at: Optional[str] = None
    failure_kind: Optional[MultiFeedFailureKind] = None


def write_corpus_manifest(
    corpus_parent: str,
    feed_results: List[MultiFeedFeedResult],
) -> None:
    """Write ``corpus_manifest.json`` at the corpus parent (GitHub #506)."""
    parent = Path(filesystem.validate_and_normalize_output_dir(corpus_parent))
    feeds_out: List[Dict[str, Any]] = []
    for fr in feed_results:
        sub = filesystem.feed_workspace_dirname(fr.feed_url)
        row: Dict[str, Any] = {
            "feed_url": fr.feed_url,
            "stable_feed_dir": sub,
            "last_run_finished_at": fr.finished_at or _utc_iso(),
            "ok": fr.ok,
            "error": fr.error,
            "episodes_processed": fr.episodes_processed,
        }
        if fr.failure_kind is not None:
            row["failure_kind"] = fr.failure_kind
        feeds_out.append(row)
    from podcast_scraper.workflow.corpus_cost_aggregation import aggregate_corpus_costs

    cost_rollup = aggregate_corpus_costs(parent)
    doc = {
        "schema_version": CORPUS_MANIFEST_SCHEMA_VERSION,
        "tool_version": __version__,
        "corpus_parent": str(parent),
        "updated_at": _utc_iso(),
        "feeds": feeds_out,
        "cost_rollup": cost_rollup,
    }
    path = parent / CORPUS_MANIFEST_FILE
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Wrote corpus manifest: %s (cost_rollup.total_cost_usd=%s over %d runs)",
        path,
        cost_rollup["total_cost_usd"],
        cost_rollup["run_count"],
    )


def build_corpus_run_summary_document(
    corpus_parent_resolved: str,
    feed_results: List[MultiFeedFeedResult],
    *,
    overall_ok: bool,
    batch_finished_at: Optional[str] = None,
    batch_incidents: Optional[Dict[str, Any]] = None,
    per_feed_incidents: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Build ``corpus_run_summary.json`` (normative shape; GitHub #506 / #557 rollup)."""
    fin = batch_finished_at or _utc_iso()
    feeds_json: List[Dict[str, Any]] = []
    pfi = per_feed_incidents or {}
    for fr in feed_results:
        row: Dict[str, Any] = {
            "feed_url": fr.feed_url,
            "ok": fr.ok,
            "error": fr.error,
            "episodes_processed": fr.episodes_processed,
        }
        if fr.finished_at:
            row["finished_at"] = fr.finished_at
        if fr.failure_kind is not None:
            row["failure_kind"] = fr.failure_kind
        ep_row = pfi.get(fr.feed_url) or {}
        row["episode_incidents_unique"] = {
            "policy": int(ep_row.get("policy", 0)),
            "soft": int(ep_row.get("soft", 0)),
            "hard": int(ep_row.get("hard", 0)),
        }
        feeds_json.append(row)
    from podcast_scraper.workflow.corpus_cost_aggregation import aggregate_corpus_costs

    cost_rollup = aggregate_corpus_costs(corpus_parent_resolved)
    doc: Dict[str, Any] = {
        "schema_version": CORPUS_RUN_SUMMARY_SCHEMA_VERSION,
        "corpus_parent": corpus_parent_resolved,
        "finished_at": fin,
        "overall_ok": overall_ok,
        "feeds": feeds_json,
        "cost_rollup": cost_rollup,
    }
    if batch_incidents is not None:
        doc["batch_incidents"] = batch_incidents
    return doc


def write_corpus_run_summary(
    corpus_parent: str,
    feed_results: List[MultiFeedFeedResult],
    *,
    overall_ok: bool,
    batch_incidents: Optional[Dict[str, Any]] = None,
    per_feed_incidents: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Write machine-readable batch summary JSON (GitHub #506). Returns the written document."""
    parent = Path(filesystem.validate_and_normalize_output_dir(corpus_parent))
    doc = build_corpus_run_summary_document(
        str(parent),
        feed_results,
        overall_ok=overall_ok,
        batch_incidents=batch_incidents,
        per_feed_incidents=per_feed_incidents,
    )
    path = parent / CORPUS_RUN_SUMMARY_FILE
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote corpus run summary: %s", path)
    logger.info(
        "corpus_multi_feed_summary: %s", json.dumps(doc, ensure_ascii=False, sort_keys=True)
    )
    return doc


def log_multi_feed_summary_structured(
    feed_results: List[MultiFeedFeedResult],
    *,
    batch_incidents: Optional[Dict[str, Any]] = None,
) -> None:
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
        if fr.failure_kind is not None:
            r["failure_kind"] = fr.failure_kind
        rows.append(r)
    payload: Dict[str, Any] = {"feeds": rows}
    if batch_incidents is not None:
        payload["batch_incidents"] = batch_incidents
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
    *,
    incident_log_path: Optional[str] = None,
    incident_log_start_offset: int = 0,
) -> Dict[str, Any]:
    """Write manifest/summary logs and optionally build parent vector index (#505/#506).

    **Partial batches:** Manifest and summary are written even when some feeds failed.
    When ``vector_search`` and FAISS are enabled, ``index_corpus`` still runs on the
    corpus parent so completed feeds contribute to the unified index; failed feeds
    simply omit metadata until a later successful run.

    **Incident rollup (GitHub #557 / opportunity #6):** When ``incident_log_path`` is set (default:
    ``<corpus_parent>/corpus_incidents.jsonl``), rows appended after ``incident_log_start_offset``
    are summarized into ``batch_incidents`` on ``corpus_run_summary.json`` so ``ok: true`` with
    ``episodes_processed: 0`` is not mistaken for “no issues.”

    Returns:
        The ``corpus_run_summary.json`` document (``feeds`` may be empty when no rows were
        recorded).
    """
    overall_ok = all(fr.ok for fr in feed_results) if feed_results else True
    write_corpus_manifest(corpus_parent, feed_results)
    inc_path = (incident_log_path or "").strip() or _default_corpus_incident_log_path(corpus_parent)
    records, end_off = parse_corpus_incident_jsonl_window(inc_path, int(incident_log_start_offset))
    batch_counts, per_feed_inc = rollup_corpus_incidents_for_multi_feed_summary(records)
    ep_u = batch_counts["episode_incidents_unique"]
    pol = int(ep_u.get("policy", 0))
    sft = int(ep_u.get("soft", 0))
    hrd = int(ep_u.get("hard", 0))
    batch_incidents: Dict[str, Any] = {
        "log_path": inc_path,
        "window_start_offset_bytes": int(incident_log_start_offset),
        "window_end_offset_bytes": int(end_off),
        "lines_in_window": len(records),
        "episode_incidents_unique": {
            "policy": pol,
            "soft": sft,
            "hard": hrd,
        },
        "feed_incidents_unique": {
            "policy": int(batch_counts["feed_incidents_unique"].get("policy", 0)),
            "soft": int(batch_counts["feed_incidents_unique"].get("soft", 0)),
            "hard": int(batch_counts["feed_incidents_unique"].get("hard", 0)),
        },
        "episodes_documented_skips_unique": pol,
        "episodes_other_incidents_unique": sft + hrd,
        "semantics_note": (
            "episodes_processed is per-feed success count from the pipeline (no feed-level "
            "exception). It can be 0 while ok is true when work finished without throw but no "
            "episode reached a full success path. episode_incidents_unique uses distinct "
            "(feed_url, episode_id or episode_idx) from episode-scoped corpus_incidents.jsonl "
            "rows in this byte window: policy=documented skips (e.g. API audio limits); soft/hard "
            "follow incident category. feed_incidents_unique counts distinct feed_url rows for "
            "feed-scoped incidents."
        ),
    }
    summary_doc = write_corpus_run_summary(
        corpus_parent,
        feed_results,
        overall_ok=overall_ok,
        batch_incidents=batch_incidents,
        per_feed_incidents=per_feed_inc,
    )
    log_multi_feed_summary_structured(feed_results, batch_incidents=batch_incidents)

    if not feed_results:
        return summary_doc
    if template_cfg.vector_search is not True:
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
