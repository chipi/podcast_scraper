"""GIL query/load layer: load artifact from path, resolve transcript, evidence spans.

Used by gi inspect and gi show-insight to load artifact file, validate, and
optionally load transcript for evidence (char_start/char_end → text).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Tuple

from .contracts import EvidenceSpan, InsightSummary, InspectOutput, SupportingQuote
from .io import read_artifact

logger = logging.getLogger(__name__)


def _declared_transcript_ref(artifact: Dict[str, Any]) -> Optional[str]:
    """The ``transcript_ref`` the artifact's own offset-bearing nodes declare, if they agree.

    GI records, on every node that carries ``char_start``, the run-relative path of the body those
    offsets were measured against. That is the answer to "which file do I slice", stated by the
    producer, so nothing downstream has to infer it.

    ``None`` when no node declares one (an artifact written before the field existed) or when the
    declarations DISAGREE. Disagreement should be impossible — one artifact is produced from one
    body — and if it ever happens, guessing which of two bodies to slice would silently show the
    wrong evidence for half the insights. Returning ``None`` sends the caller to the derivation
    below instead, which is wrong in a single predictable way rather than wrong per-node.
    """
    refs = {
        str(ref).strip()
        for node in artifact.get("nodes", [])
        if isinstance(node, dict)
        for props in [node.get("properties") or {}]
        if isinstance(props, dict) and "char_start" in props
        for ref in [props.get("transcript_ref")]
        if isinstance(ref, str) and ref.strip()
    }
    if len(refs) != 1:
        if len(refs) > 1:
            logger.warning(
                "gi artifact declares %d different transcript_ref values %s; falling back to the "
                "derived path rather than slicing one body with another's offsets",
                len(refs),
                sorted(refs),
            )
        return None
    return refs.pop()


def _transcript_path_from_artifact_path(
    artifact_path: Path, artifact: Optional[Dict[str, Any]] = None
) -> Path:
    """The transcript whose character offsets this artifact's spans index.

    Artifact: ``output_dir/metadata/<base>.gi.json``

    BUG FIXED HERE (#2253). This derived ``output_dir/transcripts/<base>.txt`` and opened it
    unconditionally, while :func:`get_evidence_span` slices that text with ``char_start`` /
    ``char_end`` that GI computed against a DIFFERENT body — the ad-free one, which is minutes
    shorter. On any ad-excised episode every evidence span came back displaced by the total length
    of the ads before it: the right NUMBER of characters from the wrong place, so it printed as
    plausible text rather than raising. Measured on a two-segment fixture, an excerpt that should
    read "Welcome back to the show." printed "r: buy things at example " instead, mid-word inside
    the sponsor line.

    IT READS WHAT THE ARTIFACT DECLARES — it does not guess. Every offset-bearing node carries
    ``transcript_ref``, the path of the body the offsets were measured against (verified across
    both fixture corpora: 520 of 520 nodes). So the producer's own record answers the question,
    and this function obeys it.

    Two earlier attempts at this fix guessed, and both were wrong in the same way. The first
    routed through ``workflow.transcript_resolution`` — a module written for translation — which
    made an English-path bug fix depend on a language feature. The second checked for
    ``.adfree.txt`` inline, which fixed English and left a TRANSLATED episode reading its source
    body at English offsets: the identical bug, one path over. Reading the declared ref is correct
    for both, correct for any variant added later, and teaches ``gi`` nothing about language.

    The derivation survives only as the fallback for an artifact that declares nothing, which is
    one written before the field existed. Callers without the artifact in hand get that same
    fallback.

    Bounded to ``gi inspect`` and ``gi show-insight``; nothing is written from this path.
    """
    output_dir = artifact_path.parent.parent  # metadata -> output_dir
    if artifact is not None:
        ref = _declared_transcript_ref(artifact)
        if ref:
            # Confined under the run dir: a `transcript_ref` is produced by this pipeline, but it
            # is still a path read out of a file, so it must not escape the directory it describes.
            candidate = (output_dir / ref).resolve()
            root = output_dir.resolve()
            if candidate == root or root in candidate.parents:
                return candidate
            logger.warning(
                "gi artifact declares a transcript_ref outside its run directory (%r); "
                "falling back to the derived path",
                ref,
            )
    stem = artifact_path.stem  # e.g. "1 - episode_title.gi"
    base = stem[:-3] if stem.endswith(".gi") else stem
    return output_dir / "transcripts" / f"{base}.txt"


def load_transcript_for_evidence(transcript_path: Path) -> Optional[str]:
    """Load transcript text from path; return None if file missing or unreadable."""
    path = Path(transcript_path)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def get_evidence_span(
    transcript_text: str,
    char_start: int,
    char_end: int,
    transcript_ref: str = "transcript.txt",
) -> EvidenceSpan:
    """Build EvidenceSpan with excerpt from transcript slice."""
    excerpt = None
    if 0 <= char_start < char_end <= len(transcript_text):
        excerpt = transcript_text[char_start:char_end]
    return EvidenceSpan(
        transcript_ref=transcript_ref,
        char_start=char_start,
        char_end=char_end,
        excerpt=excerpt,
    )


def _node_by_id(artifact: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    """Return node dict with given id or None."""
    for n in artifact.get("nodes", []):
        if n.get("id") == node_id:
            return cast(Dict[str, Any], n)
    return None


def _speaker_graph_meta(
    artifact: Dict[str, Any],
    quote_node_id: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (person_node_id, display name) from SPOKEN_BY Quote -> Person or Speaker."""
    for edge in artifact.get("edges", []):
        if edge.get("type") != "SPOKEN_BY" or edge.get("from") != quote_node_id:
            continue
        to_id = edge.get("to")
        if not to_id:
            continue
        node = _node_by_id(artifact, to_id)
        if not node:
            continue
        nt = node.get("type")
        if nt not in ("Person", "Speaker"):
            continue
        name = (node.get("properties") or {}).get("name")
        nm = str(name).strip() if name is not None else ""
        return to_id, nm or None
    return None, None


def _edges_from(artifact: Dict[str, Any], from_id: str, edge_type: str) -> List[Dict[str, Any]]:
    """Return edges with given source and type."""
    return [
        e
        for e in artifact.get("edges", [])
        if e.get("from") == from_id and e.get("type") == edge_type
    ]


def load_artifact_and_transcript(
    artifact_path: Path,
    *,
    validate: bool = True,
    strict: bool = False,
    load_transcript: bool = True,
) -> Tuple[Dict[str, Any], Optional[str], Optional[Path]]:
    """Load artifact from path and optionally transcript.

    Returns:
        (artifact_dict, transcript_text or None, transcript_path or None)
    """
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact file not found: {path}")

    artifact = read_artifact(path, validate=validate, strict=strict)

    transcript_text: Optional[str] = None
    transcript_path: Optional[Path] = None
    if load_transcript:
        transcript_path = _transcript_path_from_artifact_path(path, artifact)
        transcript_text = load_transcript_for_evidence(transcript_path)

    return artifact, transcript_text, transcript_path


def _episode_display_from_artifact(artifact: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (title, publish_date) from first Episode node if present."""
    for node in artifact.get("nodes", []):
        if node.get("type") != "Episode":
            continue
        props = node.get("properties") or {}
        title = props.get("title")
        pub = props.get("publish_date")
        t = str(title).strip() if title is not None else None
        p = str(pub).strip() if pub is not None else None
        return (t or None, p or None)
    return None, None


def build_inspect_output(
    artifact: Dict[str, Any],
    transcript_text: Optional[str] = None,
) -> InspectOutput:
    """Build InspectOutput from artifact dict; optionally fill quote excerpts from transcript."""
    episode_id = artifact.get("episode_id", "")
    ep_title, ep_pub = _episode_display_from_artifact(artifact)
    insights_out: List[InsightSummary] = []
    grounded_count = 0
    ungrounded_count = 0
    quote_count = 0

    for node in artifact.get("nodes", []):
        if node.get("type") != "Insight":
            continue
        insight_id = node.get("id", "")
        props = node.get("properties", {})
        text = props.get("text", "")
        grounded = props.get("grounded", False)
        if grounded:
            grounded_count += 1
        else:
            ungrounded_count += 1
        confidence = node.get("confidence")

        supporting: List[SupportingQuote] = []
        for edge in _edges_from(artifact, insight_id, "SUPPORTED_BY"):
            to_id = edge.get("to")
            if not to_id:
                continue
            quote_node = _node_by_id(artifact, to_id)
            if not quote_node or quote_node.get("type") != "Quote":
                continue
            quote_count += 1
            qprops = quote_node.get("properties", {})
            qnid = quote_node.get("id", "")
            char_start = qprops.get("char_start", 0)
            char_end = qprops.get("char_end", 0)
            transcript_ref = qprops.get("transcript_ref", "transcript.txt")
            evidence = get_evidence_span(
                transcript_text or "",
                char_start,
                char_end,
                transcript_ref=transcript_ref,
            )
            _, sp_graph_name = _speaker_graph_meta(artifact, qnid)
            supporting.append(
                SupportingQuote(
                    quote_id=qnid,
                    text=qprops.get("text", ""),
                    speaker_id=qprops.get("speaker_id"),
                    speaker_name=sp_graph_name,
                    timestamp_start_ms=qprops.get("timestamp_start_ms"),
                    timestamp_end_ms=qprops.get("timestamp_end_ms"),
                    evidence=evidence,
                )
            )

        insights_out.append(
            InsightSummary(
                insight_id=insight_id,
                text=text,
                grounded=grounded,
                confidence=confidence,
                episode_id=episode_id,
                episode_title=ep_title,
                publish_date=ep_pub,
                supporting_quotes=supporting,
            )
        )

    stats: Dict[str, Any] = {
        "grounded_count": grounded_count,
        "ungrounded_count": ungrounded_count,
        "quote_count": quote_count,
        "insight_count": len(insights_out),
    }

    return InspectOutput(
        episode_id=episode_id,
        schema_version=artifact.get("schema_version", "1.0"),
        model_version=artifact.get("model_version", ""),
        insights=insights_out,
        stats=stats,
    )


def find_artifact_by_episode_id(
    output_dir: Path,
    episode_id: str,
    *,
    feed_id: Optional[str] = None,
) -> Optional[Path]:
    """Resolve ``.gi.json`` for ``episode_id`` (flat or multi-feed corpus root).

    When several feeds share the same ``episode_id``, pass ``feed_id`` (metadata
    ``feed.feed_id``) to disambiguate.
    """
    from podcast_scraper.utils.corpus_episode_paths import (
        list_artifact_paths_for_episode,
        pick_single_artifact_path,
    )

    paths = list_artifact_paths_for_episode(
        output_dir,
        episode_id,
        feed_id=feed_id,
        kind="gi",
    )
    return pick_single_artifact_path(paths)


def find_artifact_by_insight_id(output_dir: Path, insight_id: str) -> Optional[Path]:
    """Scan output_dir for *.gi.json containing a node with the given insight id."""
    output_path = Path(output_dir)
    for path in output_path.rglob("*.gi.json"):
        try:
            artifact = read_artifact(path, validate=False)
            for node in artifact.get("nodes", []):
                if node.get("id") == insight_id:
                    return path
        except Exception:
            continue
    return None
