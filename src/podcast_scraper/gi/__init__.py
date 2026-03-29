"""Grounded Insight Layer (GIL): per-episode extraction and gi.json artifact.

Public API: build_artifact, validate_artifact, write_artifact, read_artifact;
load layer and contracts for gi inspect / show-insight.
"""

from .contracts import (
    EvidenceSpan,
    ExploreOutput,
    InsightSummary,
    InspectOutput,
    SupportingQuote,
    TopSpeakerEntry,
)
from .explore import (
    build_explore_output,
    collect_insights,
    explore_output_to_rfc_dict,
    load_artifacts,
    run_uc5_insight_explorer,
    scan_artifact_paths,
)
from .grounding import QuoteCandidate
from .io import read_artifact, write_artifact
from .load import (
    build_inspect_output,
    find_artifact_by_episode_id,
    find_artifact_by_insight_id,
    load_artifact_and_transcript,
)
from .pipeline import build_artifact
from .schema import validate_artifact

__all__ = [
    "build_artifact",
    "build_explore_output",
    "build_inspect_output",
    "collect_insights",
    "EvidenceSpan",
    "explore_output_to_rfc_dict",
    "ExploreOutput",
    "find_artifact_by_episode_id",
    "find_artifact_by_insight_id",
    "InspectOutput",
    "InsightSummary",
    "load_artifact_and_transcript",
    "load_artifacts",
    "QuoteCandidate",
    "read_artifact",
    "run_uc5_insight_explorer",
    "scan_artifact_paths",
    "SupportingQuote",
    "TopSpeakerEntry",
    "validate_artifact",
    "write_artifact",
]
