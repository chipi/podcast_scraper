"""Knowledge Graph Layer (KG): per-episode kg.json (separate from GIL / gi.json)."""

from .contracts import build_kg_inspect_output, KgEntityRow, KgInspectOutput, KgTopicRow
from .corpus import (
    build_embedding_document_for_kg_node,
    entity_rollup,
    load_kg_artifacts,
    scan_kg_artifact_paths,
    topic_cooccurrence,
)
from .io import read_artifact, write_artifact
from .load import find_kg_artifact_by_episode_id
from .pipeline import build_artifact
from .schema import validate_artifact

__all__ = [
    "build_embedding_document_for_kg_node",
    "KgEntityRow",
    "KgInspectOutput",
    "KgTopicRow",
    "build_artifact",
    "build_kg_inspect_output",
    "entity_rollup",
    "find_kg_artifact_by_episode_id",
    "load_kg_artifacts",
    "read_artifact",
    "scan_kg_artifact_paths",
    "topic_cooccurrence",
    "validate_artifact",
    "write_artifact",
]
