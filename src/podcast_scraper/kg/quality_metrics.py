"""PRD-019-oriented quality aggregates over per-episode ``kg.json`` (file-based, no DB).

Use for operator reports and optional CI gates via ``scripts/tools/kg_quality_metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .corpus import collect_kg_paths_from_inputs
from .io import read_artifact
from .schema import validate_artifact


@dataclass
class KgQualityMetrics:
    """Aggregated metrics over successfully parsed KG artifacts."""

    artifact_paths: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    nodes_per_artifact: List[int] = field(default_factory=list)
    edges_per_artifact: List[int] = field(default_factory=list)
    artifacts_with_extraction: int = 0
    errors: List[str] = field(default_factory=list)

    def avg_nodes_per_artifact(self) -> float:
        if not self.nodes_per_artifact:
            return 0.0
        return sum(self.nodes_per_artifact) / len(self.nodes_per_artifact)

    def avg_edges_per_artifact(self) -> float:
        if not self.edges_per_artifact:
            return 0.0
        return sum(self.edges_per_artifact) / len(self.edges_per_artifact)

    def extraction_coverage(self) -> float:
        """Share of loaded artifacts that include a non-empty ``extraction`` object."""
        if self.artifact_paths <= 0:
            return 0.0
        return self.artifacts_with_extraction / self.artifact_paths

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_paths": self.artifact_paths,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "avg_nodes_per_artifact": round(self.avg_nodes_per_artifact(), 4),
            "avg_edges_per_artifact": round(self.avg_edges_per_artifact(), 4),
            "artifacts_with_extraction": self.artifacts_with_extraction,
            "extraction_coverage": round(self.extraction_coverage(), 4),
            "errors": list(self.errors),
        }


def compute_kg_quality_metrics(
    paths: List[Any],
    *,
    strict_schema: bool = False,
) -> KgQualityMetrics:
    """Load artifacts from paths (files or dirs), validate, and compute PRD-019-oriented metrics."""
    raw_paths = [Path(p) for p in paths]
    try:
        kg_paths = collect_kg_paths_from_inputs(raw_paths)
    except (FileNotFoundError, ValueError) as e:
        m = KgQualityMetrics()
        m.errors.append(str(e))
        return m

    out = KgQualityMetrics()
    if not kg_paths:
        out.errors.append("No .kg.json files found under given paths.")
        return out

    for gpath in kg_paths:
        try:
            data = read_artifact(gpath)
            validate_artifact(data, strict=strict_schema)
        except Exception as e:
            out.errors.append(f"{gpath}: {e}")
            continue

        ep = data.get("episode_id")
        if not (isinstance(ep, str) and ep.strip()):
            out.errors.append(f"{gpath}: missing or empty episode_id")
            continue

        nodes = data.get("nodes") or []
        edges = data.get("edges") or []
        nn, ne = len(nodes), len(edges)
        out.artifact_paths += 1
        out.total_nodes += nn
        out.total_edges += ne
        out.nodes_per_artifact.append(nn)
        out.edges_per_artifact.append(ne)

        ext = data.get("extraction")
        if isinstance(ext, dict) and ext:
            out.artifacts_with_extraction += 1

    return out


def enforce_prd019_thresholds(
    m: KgQualityMetrics,
    *,
    min_artifacts: int = 1,
    min_avg_nodes: float = 1.0,
    min_avg_edges: float = 0.0,
    min_extraction_coverage: float = 1.0,
) -> Tuple[bool, List[str]]:
    """Return (all_passed, human-readable failures)."""
    failures: List[str] = []
    if m.artifact_paths <= 0:
        failures.append("No KG artifacts successfully scored (check paths and errors).")
        if m.errors:
            failures.append(f"Load errors: {len(m.errors)}")
        return False, failures

    if m.artifact_paths < min_artifacts:
        failures.append(f"artifact_paths {m.artifact_paths} < {min_artifacts}")

    if m.avg_nodes_per_artifact() + 1e-9 < min_avg_nodes:
        failures.append(
            f"avg_nodes_per_artifact {m.avg_nodes_per_artifact():.3f} < {min_avg_nodes}"
        )
    if m.avg_edges_per_artifact() + 1e-9 < min_avg_edges:
        failures.append(
            f"avg_edges_per_artifact {m.avg_edges_per_artifact():.3f} < {min_avg_edges}"
        )
    if m.extraction_coverage() + 1e-9 < min_extraction_coverage:
        failures.append(
            f"extraction_coverage {m.extraction_coverage():.3f} < {min_extraction_coverage}"
        )

    return len(failures) == 0, failures
