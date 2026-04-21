"""GIL cross-episode explore: scan artifacts, filter by topic/speaker, return insights.

Used by gi explore to build an in-memory view from per-episode gi.json files
(no DB). Supports topic filter via Topic label (ABOUT edge) or substring in insight
text; optional speaker filter on quote speaker_id or graph speaker_name.

When ``<output_dir>/search/vectors.faiss`` exists and ``--topic`` is set, topic matching
uses the semantic corpus index first to propose insight candidates (metadata map loads
``gi.json`` paths). Each candidate must still satisfy the same **topic contains** rule as the
non-semantic path (substring on linked Topic labels or insight text); on failure or no hits,
behavior falls back to substring/topic-label matching over all artifacts.

Programmatic helpers implement UC1–UC5 patterns as thin wrappers over collect/sort.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast, Dict, List, Literal, Optional, Set, Tuple

from podcast_scraper.search.corpus_scope import (
    discover_metadata_files,
    episode_root_from_metadata_path,
    gi_map_lookup_key_from_vector_meta,
    index_fingerprint_scope_key,
    normalize_feed_id,
)
from podcast_scraper.utils.log_redaction import format_exception_for_log
from podcast_scraper.workflow.metadata_generation import _determine_gi_path

from .contracts import (
    ExploreOutput,
    InsightSummary,
    SupportingQuote,
    TopicEntry,
    TopSpeakerEntry,
)
from .io import read_artifact
from .load import build_inspect_output

logger = logging.getLogger(__name__)

# Exit codes: 0 success, 2 invalid args, 3 missing/no artifacts, 4 no matching results,
# 5 strict schema validation failed (explore lists 4 for strict; we keep 4=no match for
# backward compatibility with existing CLI tests — see GROUNDED_INSIGHTS_GUIDE).
EXIT_SUCCESS = 0
EXIT_INVALID_ARGS = 2
EXIT_NO_ARTIFACTS = 3
EXIT_NO_RESULTS = 4
EXIT_STRICT_VALIDATION_FAILED = 5


class ExploreValidationError(Exception):
    """Raised when --strict explore mode hits an invalid artifact."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        super().__init__(message)


def scan_artifact_paths(output_dir: Path) -> List[Path]:
    """List all .gi.json paths under output_dir (metadata/*.gi.json and rglob)."""
    out = Path(output_dir)
    if not out.is_dir():
        return []
    paths: List[Path] = []
    metadata_dir = out / "metadata"
    if metadata_dir.is_dir():
        paths.extend(metadata_dir.glob("*.gi.json"))
    for p in out.rglob("*.gi.json"):
        if p not in paths:
            paths.append(p)
    return sorted(set(paths))


def _topic_labels_for_insight(artifact: Dict[str, Any], insight_id: str) -> List[str]:
    """Return topic labels linked to this insight via ABOUT edges (Insight -> Topic)."""
    labels: List[str] = []
    for edge in artifact.get("edges", []):
        if edge.get("from") != insight_id or edge.get("type") != "ABOUT":
            continue
        to_id = edge.get("to")
        if not to_id:
            continue
        for node in artifact.get("nodes", []):
            if node.get("id") == to_id and node.get("type") == "Topic":
                props = node.get("properties") or {}
                label = props.get("label") or props.get("name") or ""
                if label:
                    labels.append(str(label))
                break
    return labels


def _insight_matches_topic(
    artifact: Dict[str, Any],
    insight_id: str,
    insight_text: str,
    topic: Optional[str],
) -> bool:
    """True if topic is None or matches insight (Topic label or substring in text)."""
    if not topic or not topic.strip():
        return True
    key = topic.strip().lower()
    labels = _topic_labels_for_insight(artifact, insight_id)
    for label in labels:
        if key in label.lower():
            return True
    if key in insight_text.lower():
        return True
    return False


def _supporting_quote_speaker_key(q: SupportingQuote) -> Optional[str]:
    """Stable aggregation key: prefer diarization id, else graph speaker name."""
    if q.speaker_id and str(q.speaker_id).strip():
        return str(q.speaker_id).strip()
    if q.speaker_name and str(q.speaker_name).strip():
        return str(q.speaker_name).strip()
    return None


def default_vector_index_dir(output_dir: Path) -> Optional[Path]:
    """Return ``output_dir / search`` if a FAISS index is present."""
    p = Path(output_dir) / "search"
    if (p / "vectors.faiss").is_file():
        return p
    return None


def _load_metadata_doc_quick(path: Path) -> Optional[Dict[str, Any]]:
    """Load episode metadata JSON/YAML for ``episode_id`` / ``grounded_insights`` mapping."""
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            doc = json.loads(raw)
        else:
            import yaml

            doc = yaml.safe_load(raw)
        return doc if isinstance(doc, dict) else None
    except Exception:
        return None


def _scoped_gi_paths_from_metadata(output_dir: Path) -> Dict[str, Path]:
    """Map scoped episode key → ``gi.json`` path (recursive metadata; GitHub #505)."""
    out: Dict[str, Path] = {}
    root = Path(output_dir)
    for meta_path in discover_metadata_files(root):
        doc = _load_metadata_doc_quick(meta_path)
        if not doc:
            continue
        ep = (doc.get("episode") or {}).get("episode_id")
        if not isinstance(ep, str) or not ep:
            continue
        feed = doc.get("feed") or {}
        scope_key = index_fingerprint_scope_key(normalize_feed_id(feed.get("feed_id")), ep)
        if scope_key in out:
            continue
        eroot = episode_root_from_metadata_path(meta_path)
        gi = doc.get("grounded_insights")
        gp: Optional[Path] = None
        if isinstance(gi, dict):
            rel = gi.get("artifact_path")
            if isinstance(rel, str) and rel.strip():
                cand = (eroot / rel.strip()).resolve()
                if cand.is_file():
                    gp = cand
        if gp is None:
            gp = Path(_determine_gi_path(str(meta_path))).resolve()
        if gp.is_file():
            out[scope_key] = gp
    return out


def _merge_episode_gi_paths(
    output_dir: Optional[Path],
    artifacts_with_paths: List[Tuple[Path, Dict[str, Any]]],
) -> Dict[str, Path]:
    """Prefer on-disk metadata index; fill gaps from pre-loaded ``(path, artifact)`` pairs."""
    merged: Dict[str, Path] = {}
    if output_dir is not None:
        merged.update(_scoped_gi_paths_from_metadata(output_dir))
    for path, art in artifacts_with_paths:
        eid = art.get("episode_id")
        if isinstance(eid, str) and eid not in merged and path.is_file():
            merged[eid] = path.resolve()
    return merged


def _load_gi_artifacts_from_paths(
    paths_in_order: List[Path],
    *,
    strict: bool,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Load ``gi.json`` artifacts for an explicit path list (semantic fast path)."""
    seen: Set[Path] = set()
    loaded: List[Tuple[Path, Dict[str, Any]]] = []
    for raw in paths_in_order:
        p = raw.resolve()
        if p in seen:
            continue
        seen.add(p)
        try:
            art = read_artifact(p, validate=strict, strict=strict)
            loaded.append((p, art))
        except Exception as e:
            if strict:
                raise ExploreValidationError(
                    p,
                    f"artifact validation failed: {e}",
                ) from e
            logger.warning(
                "Skip invalid artifact %s: %s",
                p,
                format_exception_for_log(e),
            )
    return loaded


def explore_resolve_insights_and_loaded(
    output_dir: Path,
    paths: List[Path],
    *,
    topic: Optional[str],
    speaker: Optional[str],
    grounded_only: bool,
    min_confidence: Optional[float],
    sort_by: Literal["confidence", "time"],
    strict: bool,
) -> Tuple[List[InsightSummary], bool, List[Tuple[Path, Dict[str, Any]]], int]:
    """Semantic-first when topic+index exist (metadata map, partial gi load); else full scan.

    Returns:
        (insights, semantic_ranked, loaded_for_aggregation, episodes_searched_count)
    """
    index_dir = default_vector_index_dir(output_dir)
    t = topic.strip() if isinstance(topic, str) else ""
    if t and index_dir is not None:
        try:
            sem = _collect_insights_semantic(
                output_dir,
                [],
                topic=t,
                speaker=speaker,
                grounded_only=grounded_only,
                min_confidence=min_confidence,
                limit=None,
                index_dir=index_dir,
                strict=strict,
            )
        except ExploreValidationError:
            raise
        except Exception as exc:
            logger.warning(
                "Semantic topic match unavailable (%s); using substring/topic-label match",
                format_exception_for_log(exc),
            )
            sem = ([], [])
        if sem[0]:
            insights_sem, gi_paths = sem
            loaded = _load_gi_artifacts_from_paths(gi_paths, strict=strict)
            n_ep = len({p.resolve() for p in gi_paths})
            return insights_sem, True, loaded, n_ep

    loaded = load_artifacts(paths, validate=strict, strict=strict)
    insights, ranked = collect_insights(
        loaded,
        topic=topic,
        speaker=speaker,
        grounded_only=grounded_only,
        min_confidence=min_confidence,
        limit=None,
        semantic_index_dir=None,
        output_dir=None,
        strict=strict,
    )
    if not ranked:
        insights = sort_insights(insights, sort_by=sort_by)
    return insights, ranked, loaded, len(paths)


def _find_insight_summary(artifact: Dict[str, Any], insight_id: str) -> Optional[InsightSummary]:
    """Return ``InsightSummary`` for ``insight_id`` in ``artifact``, if any."""
    inspect_out = build_inspect_output(artifact, transcript_text=None)
    for ins in inspect_out.insights:
        if ins.insight_id == insight_id:
            return ins
    return None


def _collect_insights_semantic(
    output_dir: Optional[Path],
    artifacts_with_paths: List[Tuple[Path, Dict[str, Any]]],
    *,
    topic: str,
    speaker: Optional[str],
    grounded_only: bool,
    min_confidence: Optional[float],
    limit: Optional[int],
    index_dir: Path,
    strict: bool = False,
) -> Tuple[List[InsightSummary], List[Path]]:
    """Rank insights by vector similarity; resolve ``gi.json`` via metadata map or pre-loaded.

    Returns:
        (insights, gi_paths) — ``gi_paths`` lists each ``gi.json`` used, in hit order
        (deduped later by the caller when counting episodes).
    """
    from podcast_scraper.providers.ml import embedding_loader
    from podcast_scraper.search.faiss_store import FaissVectorStore, VECTORS_FILE

    if not (index_dir / VECTORS_FILE).is_file():
        return [], []

    store = FaissVectorStore.load(index_dir)
    if store.ntotal == 0:
        return [], []

    model_id = store.stats().embedding_model
    qvec = embedding_loader.encode(
        topic,
        model_id,
        return_numpy=False,
        allow_download=False,
    )
    if not (isinstance(qvec, list) and qvec and isinstance(qvec[0], float)):
        raise ValueError("semantic explore: expected a single query embedding")
    qemb = cast(List[float], qvec)

    top_k = max(1, limit or 200)
    fetch_k = min(max(top_k * 25, top_k), max(store.ntotal, 1))
    hits = store.search(
        qemb,
        top_k=fetch_k,
        filters={"doc_type": "insight"},
        overfetch_factor=1,
    )

    by_ep_path = _merge_episode_gi_paths(output_dir, artifacts_with_paths)

    collected: List[InsightSummary] = []
    gi_paths: List[Path] = []
    seen: Set[Tuple[str, str]] = set()
    for h in hits:
        meta = h.metadata
        ep = meta.get("episode_id")
        iid = meta.get("source_id")
        if not isinstance(ep, str) or not isinstance(iid, str):
            continue
        gkey = gi_map_lookup_key_from_vector_meta(cast(Dict[str, Any], meta))
        gpath = by_ep_path.get(gkey) if gkey else None
        if gpath is None:
            gpath = by_ep_path.get(ep)
        if not gpath:
            continue
        key = (str(gpath.resolve()), iid)
        if key in seen:
            continue
        try:
            artifact = read_artifact(gpath, validate=strict, strict=strict)
        except Exception as e:
            if strict:
                raise ExploreValidationError(
                    gpath,
                    f"artifact validation failed: {e}",
                ) from e
            logger.warning(
                "Skip invalid artifact %s: %s",
                gpath,
                format_exception_for_log(e),
            )
            continue
        ins = _find_insight_summary(artifact, iid)
        if ins is None:
            continue
        # Same contract as non-semantic explore: "topic contains" is substring on linked
        # Topic labels or insight body (vector hits alone can be only loosely related).
        if not _insight_matches_topic(artifact, iid, ins.text, topic):
            continue
        if not _insight_matches_speaker(ins, speaker):
            continue
        if grounded_only and not ins.grounded:
            continue
        if min_confidence is not None and (
            ins.confidence is None or ins.confidence < min_confidence
        ):
            continue
        seen.add(key)
        collected.append(ins)
        gi_paths.append(gpath)
        if limit is not None and len(collected) >= limit:
            break
    return collected, gi_paths


def _insight_matches_speaker(ins: InsightSummary, speaker_substring: Optional[str]) -> bool:
    """True if filter is None or any quote matches speaker_id or speaker_name."""
    if not speaker_substring or not speaker_substring.strip():
        return True
    key = speaker_substring.strip().lower()
    for q in ins.supporting_quotes:
        sid = q.speaker_id
        if sid and key in str(sid).lower():
            return True
        sname = q.speaker_name
        if sname and key in str(sname).lower():
            return True
    return False


def load_artifacts(
    paths: List[Path],
    validate: bool = False,
    strict: bool = False,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Load artifacts from paths.

    When strict is True, validation runs on each file; the first failure raises
    ExploreValidationError. When strict is False, invalid files are skipped with a warning.
    """
    result: List[Tuple[Path, Dict[str, Any]]] = []
    for path in paths:
        try:
            artifact = read_artifact(path, validate=validate or strict, strict=strict)
            result.append((path, artifact))
        except Exception as e:
            if strict:
                raise ExploreValidationError(
                    path,
                    f"artifact validation failed: {e}",
                ) from e
            logger.warning(
                "Skip invalid artifact %s: %s",
                path,
                format_exception_for_log(e),
            )
    return result


def collect_insights(
    artifacts_with_paths: List[Tuple[Path, Dict[str, Any]]],
    topic: Optional[str] = None,
    speaker: Optional[str] = None,
    grounded_only: bool = False,
    min_confidence: Optional[float] = None,
    limit: Optional[int] = None,
    *,
    semantic_index_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    strict: bool = False,
) -> Tuple[List[InsightSummary], bool]:
    """Collect InsightSummary from artifacts with optional topic/speaker/grounded filters.

    When ``topic`` is set and ``semantic_index_dir`` contains ``vectors.faiss``, tries
    vector-ranked insights first. If that yields at least one result, returns
    ``(insights, True)`` (caller should not re-sort by confidence/time). Otherwise falls
    back to substring/topic-label matching and returns ``(insights, False)``.

    Pass ``output_dir`` to resolve ``gi.json`` paths from ``metadata/*`` without requiring
    pre-loaded artifacts (CLI semantic fast path). Pre-loaded pairs still merge into the map.

    Does not cap by relevance until after collection when using sort; pass
    ``limit=None`` to collect all matches before sort+slice in the caller.
    """
    t = topic.strip() if isinstance(topic, str) else ""
    if t and semantic_index_dir is not None:
        try:
            sem, _paths = _collect_insights_semantic(
                output_dir,
                artifacts_with_paths,
                topic=t,
                speaker=speaker,
                grounded_only=grounded_only,
                min_confidence=min_confidence,
                limit=limit,
                index_dir=semantic_index_dir,
                strict=strict,
            )
        except ExploreValidationError:
            raise
        except Exception as exc:
            logger.warning(
                "Semantic topic match unavailable (%s); using substring/topic-label match",
                format_exception_for_log(exc),
            )
            sem = []
        if sem:
            return sem, True

    collected: List[InsightSummary] = []
    for _path, artifact in artifacts_with_paths:
        inspect_out = build_inspect_output(artifact, transcript_text=None)
        for ins in inspect_out.insights:
            if not _insight_matches_topic(artifact, ins.insight_id, ins.text, topic):
                continue
            if not _insight_matches_speaker(ins, speaker):
                continue
            if grounded_only and not ins.grounded:
                continue
            if min_confidence is not None and (
                ins.confidence is None or ins.confidence < min_confidence
            ):
                continue
            collected.append(ins)
            if limit is not None and len(collected) >= limit:
                return collected, False
    return collected, False


def _parse_publish_sort_key(iso: Optional[str]) -> float:
    """Return sortable timestamp; missing or bad dates sort last."""
    if not iso or not str(iso).strip():
        return float("-inf")
    s = str(iso).strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return float("-inf")


def sort_insights(
    insights: List[InsightSummary],
    sort_by: Literal["confidence", "time"] = "confidence",
) -> List[InsightSummary]:
    """Sort insights: confidence descending (None last), or publish_date descending."""
    if sort_by == "time":
        return sorted(
            insights,
            key=lambda i: _parse_publish_sort_key(i.publish_date),
            reverse=True,
        )
    return sorted(
        insights,
        key=lambda i: (i.confidence is not None, i.confidence or 0.0),
        reverse=True,
    )


def _compute_top_speakers(
    insights: List[InsightSummary],
    *,
    max_entries: int = 20,
) -> List[TopSpeakerEntry]:
    """Aggregate quote and insight counts per speaker (id or graph name)."""
    quote_counts: Dict[str, int] = defaultdict(int)
    insight_ids_by_speaker: Dict[str, set] = defaultdict(set)
    for ins in insights:
        speakers_here: Set[str] = set()
        for q in ins.supporting_quotes:
            k = _supporting_quote_speaker_key(q)
            if k:
                speakers_here.add(k)
        for q in ins.supporting_quotes:
            key = _supporting_quote_speaker_key(q)
            if not key:
                continue
            quote_counts[key] += 1
        for key in speakers_here:
            insight_ids_by_speaker[key].add(ins.insight_id)
    ranked = sorted(quote_counts.keys(), key=lambda k: quote_counts[k], reverse=True)
    out: List[TopSpeakerEntry] = []
    for sid in ranked[:max_entries]:
        display_name: Optional[str] = None
        for ins in insights:
            for q in ins.supporting_quotes:
                if _supporting_quote_speaker_key(q) != sid:
                    continue
                if q.speaker_name and str(q.speaker_name).strip():
                    display_name = str(q.speaker_name).strip()
                    break
            if display_name:
                break
        out.append(
            TopSpeakerEntry(
                speaker_id=sid,
                name=display_name,
                quote_count=quote_counts[sid],
                insight_count=len(insight_ids_by_speaker.get(sid, set())),
            )
        )
    return out


def aggregate_topic_entries_for_insights(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    insights: List[InsightSummary],
) -> List[TopicEntry]:
    """Count Topic nodes (ABOUT edges) for insights present in the explore result (#487)."""
    wanted = {(i.episode_id, i.insight_id) for i in insights}
    counts: Dict[str, Dict[str, Any]] = {}

    for _path, art in loaded:
        ep = str(art.get("episode_id") or "")
        nodes_by_id = {
            str(n["id"]): n for n in (art.get("nodes") or []) if isinstance(n, dict) and n.get("id")
        }
        for edge in art.get("edges") or []:
            if not isinstance(edge, dict) or edge.get("type") != "ABOUT":
                continue
            ins_id = edge.get("from")
            to_id = edge.get("to")
            if not ins_id or not to_id:
                continue
            if (ep, ins_id) not in wanted:
                continue
            tn = nodes_by_id.get(str(to_id))
            if not tn or tn.get("type") != "Topic":
                continue
            _tp = tn.get("properties")
            props: Dict[str, Any] = _tp if isinstance(_tp, dict) else {}
            label = str(props.get("label") or "").strip()
            tid = str(tn.get("id") or to_id)
            if tid not in counts:
                counts[tid] = {"label": label or tid, "insight_count": 0}
            counts[tid]["insight_count"] += 1

    ranked = sorted(
        counts.items(),
        key=lambda x: (-x[1]["insight_count"], str(x[1]["label"]).lower()),
    )
    return [
        TopicEntry(
            topic_id=tid,
            label=str(meta["label"]),
            insight_count=int(meta["insight_count"]),
        )
        for tid, meta in ranked
    ]


def build_explore_output(
    insights: List[InsightSummary],
    episodes_searched: int,
    topic: Optional[str] = None,
    speaker_filter: Optional[str] = None,
    topics: Optional[List[TopicEntry]] = None,
) -> ExploreOutput:
    """Build ExploreOutput from collected insights and episode count."""
    grounded_count = sum(1 for i in insights if i.grounded)
    quote_count = sum(len(i.supporting_quotes) for i in insights)
    episode_ids = {i.episode_id for i in insights}
    distinct_speakers = set()
    for i in insights:
        for q in i.supporting_quotes:
            k = _supporting_quote_speaker_key(q)
            if k:
                distinct_speakers.add(k)
    top = _compute_top_speakers(insights)
    topics_out = topics if topics is not None else []
    summary: Dict[str, Any] = {
        "insight_count": len(insights),
        "grounded_insight_count": grounded_count,
        "quote_count": quote_count,
        "episode_count": len(episode_ids),
        "speaker_count": len(distinct_speakers),
        "topic_count": len(topics_out),
    }
    return ExploreOutput(
        topic=topic,
        speaker_filter=speaker_filter,
        insights=insights,
        summary=summary,
        top_speakers=top,
        topics=topics_out,
        episodes_searched=episodes_searched,
    )


def topic_slug_for_rfc(label: str) -> str:
    """Stable slug for synthetic topic_id."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower().strip()).strip("-")
    return slug or "topic"


def explore_output_to_rfc_dict(out: ExploreOutput) -> Dict[str, Any]:
    """JSON shape for gi explore (topic object, nested episode/speaker)."""
    topic_obj: Optional[Dict[str, str]] = None
    if out.topic and out.topic.strip():
        lab = out.topic.strip()
        topic_obj = {
            "topic_id": f"topic:{topic_slug_for_rfc(lab)}",
            "label": lab,
        }
    insights_payload: List[Dict[str, Any]] = []
    for ins in out.insights:
        quotes_payload: List[Dict[str, Any]] = []
        for q in ins.supporting_quotes:
            spk: Optional[Dict[str, Any]] = None
            sk = _supporting_quote_speaker_key(q)
            if sk:
                spk = {
                    "speaker_id": (
                        str(q.speaker_id).strip()
                        if q.speaker_id and str(q.speaker_id).strip()
                        else sk
                    ),
                    "name": (
                        q.speaker_name.strip()
                        if q.speaker_name and str(q.speaker_name).strip()
                        else None
                    ),
                }
            qd: Dict[str, Any] = {
                "quote_id": q.quote_id,
                "text": q.text,
                "timestamp_start_ms": q.timestamp_start_ms,
                "timestamp_end_ms": q.timestamp_end_ms,
                "evidence": {
                    "transcript_ref": q.evidence.transcript_ref,
                    "char_start": q.evidence.char_start,
                    "char_end": q.evidence.char_end,
                },
            }
            if q.evidence.excerpt is not None:
                qd["evidence"]["excerpt"] = q.evidence.excerpt
            if spk is not None:
                qd["speaker"] = spk
            quotes_payload.append(qd)
        insights_payload.append(
            {
                "insight_id": ins.insight_id,
                "text": ins.text,
                "grounded": ins.grounded,
                "confidence": ins.confidence,
                "episode": {
                    "episode_id": ins.episode_id,
                    "title": ins.episode_title or "",
                    "publish_date": ins.publish_date or "",
                },
                "supporting_quotes": quotes_payload,
            }
        )
    return {
        "topic": topic_obj,
        "speaker_filter": out.speaker_filter,
        "summary": out.summary,
        "topics": [t.model_dump() for t in out.topics],
        "insights": insights_payload,
        "top_speakers": [ts.model_dump() for ts in out.top_speakers],
        "episodes_searched": out.episodes_searched,
    }


# --- Explore use-case helpers (programmatic; CLI uses collect + sort + build) ---


def run_uc5_insight_explorer(
    output_dir: Path,
    *,
    topic: Optional[str] = None,
    speaker: Optional[str] = None,
    grounded_only: bool = False,
    min_confidence: Optional[float] = None,
    sort_by: Literal["confidence", "time"] = "confidence",
    limit: int = 50,
    strict: bool = False,
) -> ExploreOutput:
    """UC5: cross-episode insights with quotes (canonical gi explore behavior)."""
    paths = scan_artifact_paths(output_dir)
    if not paths:
        return build_explore_output([], 0, topic=topic, speaker_filter=speaker, topics=[])
    insights, _semantic_ranked, loaded, ep_count = explore_resolve_insights_and_loaded(
        output_dir,
        paths,
        topic=topic,
        speaker=speaker,
        grounded_only=grounded_only,
        min_confidence=min_confidence,
        sort_by=sort_by,
        strict=strict,
    )
    if limit > 0:
        insights = insights[:limit]
    topic_rows = aggregate_topic_entries_for_insights(loaded, insights)
    return build_explore_output(
        insights,
        episodes_searched=ep_count,
        topic=topic,
        speaker_filter=speaker,
        topics=topic_rows,
    )


def run_uc1_topic_research(
    output_dir: Path,
    *,
    topic: str,
    limit: int = 50,
    sort_by: Literal["confidence", "time"] = "confidence",
    strict: bool = False,
) -> ExploreOutput:
    """UC1: topic → insights → quotes → episodes (topic filter required)."""
    return run_uc5_insight_explorer(
        output_dir,
        topic=topic,
        limit=limit,
        sort_by=sort_by,
        strict=strict,
    )


def run_uc2_speaker_mapping(
    output_dir: Path,
    *,
    speaker: str,
    topic: Optional[str] = None,
    limit: int = 50,
    sort_by: Literal["confidence", "time"] = "confidence",
    strict: bool = False,
) -> ExploreOutput:
    """UC2: speaker-centric slice (substring match on quote speaker_id or name)."""
    return run_uc5_insight_explorer(
        output_dir,
        topic=topic,
        speaker=speaker,
        limit=limit,
        sort_by=sort_by,
        strict=strict,
    )


def find_insight_by_id(
    artifacts_with_paths: List[Tuple[Path, Dict[str, Any]]],
    insight_id: str,
) -> Optional[InsightSummary]:
    """UC3 helper: return InsightSummary for a given insight_id if present."""
    for _path, artifact in artifacts_with_paths:
        inspect_out = build_inspect_output(artifact, transcript_text=None)
        for ins in inspect_out.insights:
            if ins.insight_id == insight_id:
                return ins
    return None


def map_uc4_question_to_params(question: str) -> Optional[Dict[str, Any]]:
    """UC4: deterministic pattern map for ``gi query``.

    Unmatched questions stay ``None``; broadening this list is low priority (GitHub #487
    EV-2) because semantic corpus search (``podcast search``, #484) is expected to
    replace prefix-style matching for ad hoc questions.

    Returns:
        - ``{"topic": str}`` and/or ``{"speaker": str}`` for explore filters
        - ``{"topic_leaderboard": True}`` for UC4 topic-ranking question
        - ``None`` if no supported pattern matches.
    """
    q = question.strip()
    low = q.lower()
    lb = low.rstrip("?").strip()

    leaderboard_exact = {
        "which topics have the most insights",
        "what topics have the most insights",
        "top topics",
        "rank topics by insights",
        "what are the top topics",
        "show topic rankings",
    }
    leaderboard_prefixes = (
        "which topics have the most insights ",
        "what topics have the most insights ",
    )
    if lb in leaderboard_exact or any(lb.startswith(p) for p in leaderboard_prefixes):
        return {"topic_leaderboard": True}

    def topic_from_suffix(rest: str) -> Optional[str]:
        t = rest.strip()
        if t.endswith("?"):
            t = t[:-1].strip()
        return t if t else None

    topic_prefixes = (
        "what insights are there about ",
        "what insights about ",
        "insights about ",
        "show me insights about ",
        "tell me about insights on ",
        "what are insights about ",
    )
    for pref in topic_prefixes:
        if low.startswith(pref):
            t = topic_from_suffix(q[len(pref) :])
            if t:
                return {"topic": t}
            break

    # Compound: "What did Sam Altman say about innovation?"
    if low.startswith("what did ") and " say about " in low:
        rest = q[len("what did ") :]
        idx = rest.lower().index(" say about ")
        speaker_part = rest[:idx].strip()
        topic_part = topic_from_suffix(rest[idx + len(" say about ") :])
        if speaker_part and topic_part:
            return {"speaker": speaker_part, "topic": topic_part}

    if low.startswith("what did ") and " say" in low:
        end = low.index(" say")
        name = q[len("what did ") : end].strip()
        if name:
            return {"speaker": name}
    return None


def run_uc4_topic_leaderboard(
    output_dir: Path,
    question: str,
    *,
    limit: int = 20,
    strict: bool = False,
) -> Dict[str, Any]:
    """UC4: rank topic labels by linked insight count."""
    paths = scan_artifact_paths(output_dir)
    if not paths:
        return {
            "question": question,
            "answer": {
                "topics": [],
                "summary": {
                    "topic_count": 0,
                    "episodes_searched": 0,
                    "insight_count": 0,
                },
            },
            "explanation": "No .gi.json artifacts found.",
        }

    loaded = load_artifacts(paths, validate=strict, strict=strict)
    insight_count_by_topic: Dict[str, int] = defaultdict(int)
    episodes_by_topic: Dict[str, Set[str]] = defaultdict(set)
    total_insights = 0

    for _path, artifact in loaded:
        inspect_out = build_inspect_output(artifact, transcript_text=None)
        for ins in inspect_out.insights:
            total_insights += 1
            labels = _topic_labels_for_insight(artifact, ins.insight_id)
            if not labels:
                labels = ["(untagged)"]
            for lab in labels:
                insight_count_by_topic[lab] += 1
                episodes_by_topic[lab].add(ins.episode_id)

    ranked = sorted(insight_count_by_topic.items(), key=lambda x: (-x[1], x[0].lower()))
    if limit > 0:
        ranked = ranked[:limit]

    topics_payload = [
        {
            "topic_label": label,
            "insight_count": cnt,
            "episode_count": len(episodes_by_topic[label]),
        }
        for label, cnt in ranked
    ]

    return {
        "question": question,
        "answer": {
            "topics": topics_payload,
            "summary": {
                "topic_count": len(insight_count_by_topic),
                "episodes_searched": len(paths),
                "insight_count": total_insights,
            },
        },
        "explanation": (
            f"Ranked {len(insight_count_by_topic)} topic label(s) across "
            f"{len(paths)} episode artifact(s)."
        ),
    }


def run_uc4_semantic_qa(
    output_dir: Path,
    question: str,
    *,
    limit: int = 20,
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """UC4: map question to UC1/UC2-style explore; return answer envelope or None."""
    params = map_uc4_question_to_params(question)
    if not params:
        return None
    if params.get("topic_leaderboard"):
        return run_uc4_topic_leaderboard(output_dir, question, limit=limit, strict=strict)

    topic = params.get("topic")
    speaker = params.get("speaker")
    if isinstance(topic, str):
        topic = topic.strip() or None
    if isinstance(speaker, str):
        speaker = speaker.strip() or None

    out = run_uc5_insight_explorer(
        output_dir,
        topic=topic,
        speaker=speaker,
        limit=limit,
        strict=strict,
    )
    if topic and speaker:
        explain = (
            f"Matched {len(out.insights)} insight(s) via speaker and topic filters "
            f"({speaker!r} / {topic!r})."
        )
    elif topic:
        explain = f"Matched {len(out.insights)} insight(s) via topic filter."
    else:
        explain = f"Matched {len(out.insights)} insight(s) via speaker filter."
    return {
        "question": question,
        "answer": explore_output_to_rfc_dict(out),
        "explanation": explain,
    }
