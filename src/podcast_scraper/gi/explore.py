"""GIL cross-episode explore: scan artifacts, filter by topic, return insights with quotes.

Used by gi explore to build an in-memory view from per-episode gi.json files
(no DB). Supports topic filter via Topic label (ABOUT edge) or substring in insight text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .contracts import ExploreOutput, InsightSummary
from .io import read_artifact
from .load import build_inspect_output

logger = logging.getLogger(__name__)

# Exit code convention: 0 success, 2 invalid args, 3 missing files/no artifacts, 4 no results
EXIT_SUCCESS = 0
EXIT_INVALID_ARGS = 2
EXIT_NO_ARTIFACTS = 3
EXIT_NO_RESULTS = 4


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


def load_artifacts(
    paths: List[Path],
    validate: bool = False,
    strict: bool = False,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Load artifacts from paths; skip invalid with warning. Returns (path, artifact)."""
    from .schema import validate_artifact

    result: List[Tuple[Path, Dict[str, Any]]] = []
    for path in paths:
        try:
            artifact = read_artifact(path)
            if validate:
                validate_artifact(artifact, strict=strict)
            result.append((path, artifact))
        except Exception as e:
            logger.warning("Skip invalid artifact %s: %s", path, e)
    return result


def collect_insights(
    artifacts_with_paths: List[Tuple[Path, Dict[str, Any]]],
    topic: Optional[str] = None,
    grounded_only: bool = False,
    min_confidence: Optional[float] = None,
    limit: Optional[int] = None,
) -> List[InsightSummary]:
    """Collect InsightSummary from artifacts with optional topic/grounded/confidence filter."""
    collected: List[InsightSummary] = []
    for _path, artifact in artifacts_with_paths:
        inspect_out = build_inspect_output(artifact, transcript_text=None)
        for ins in inspect_out.insights:
            if not _insight_matches_topic(artifact, ins.insight_id, ins.text, topic):
                continue
            if grounded_only and not ins.grounded:
                continue
            if min_confidence is not None and (
                ins.confidence is None or ins.confidence < min_confidence
            ):
                continue
            collected.append(ins)
            if limit is not None and len(collected) >= limit:
                return collected
    return collected


def build_explore_output(
    insights: List[InsightSummary],
    episodes_searched: int,
    topic: Optional[str] = None,
) -> ExploreOutput:
    """Build ExploreOutput from collected insights and episode count."""
    grounded_count = sum(1 for i in insights if i.grounded)
    quote_count = sum(len(i.supporting_quotes) for i in insights)
    episode_ids = {i.episode_id for i in insights}
    summary: Dict[str, Any] = {
        "insight_count": len(insights),
        "grounded_insight_count": grounded_count,
        "quote_count": quote_count,
        "episode_count": len(episode_ids),
    }
    return ExploreOutput(
        topic=topic,
        insights=insights,
        summary=summary,
        episodes_searched=episodes_searched,
    )
