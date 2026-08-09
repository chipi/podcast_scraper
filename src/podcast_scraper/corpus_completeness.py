"""Corpus-completeness + index-staleness gates (#1494 / #1497).

Two read-only guards that catch a corpus which would silently under-serve the MCP / player /
search surfaces if deployed — the preventative half of the two incidents that surfaced this arc:

- **Index staleness (#1494).** The served LanceDB index schema is older than the code's
  ``LANCE_SCHEMA_VERSION`` → every semantic search returns ``no_index`` (a FAISS-v1 index sat on
  v3 code for months, silently dead). ``assess_index_staleness`` is positive-evidence based, like
  the read path's own ``lance_index_is_stale``.
- **Edge / stage completeness (#1497).** The corpus is missing the typed edges / enrichments /
  diarization the current relational-query layer traverses → ~16 MCP tools return empty. The old
  prod corpus had only generic ``MENTIONS`` (no ``MENTIONS_PERSON/ORG``), no ``HAS_EPISODE``, no
  ``SPOKEN_BY``, no ``enrichments/`` — all silently.

HARD checks fail the gate (they come from deterministic stages that a full pipeline run always
produces); SOFT checks warn only (diarization is legitimately optional — bridge-only audio may be
unavailable). The CLI (``corpus-completeness-check``) / ``make corpus-completeness-check`` turn a
HARD failure into a non-zero exit so a deploy can gate on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .search.backends.lancedb_backend import LANCE_SCHEMA_VERSION, stored_schema_version
from .search.hybrid_search import lance_index_dir
from .search.topic_clusters import TOPIC_CLUSTERS_FILENAME
from .utils.path_validation import safe_resolve_directory

# Stage → (edge types that satisfy it, MCP tools it kills). From #1497's evidence table.
# ``any_of`` = the stage is satisfied when ANY of these edge types is present in the corpus.
_HARD_STAGES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "relational edges — HAS_EPISODE (make enrich-relational-edges)": {
        "any_of": ("HAS_EPISODE",),
        "kills": ("show_episodes",),
    },
    "relational edges — typed MENTIONS_PERSON/ORG (make enrich-relational-edges)": {
        "any_of": ("MENTIONS_PERSON", "MENTIONS_ORG"),
        "kills": (
            "entities_in_topic",
            "insights_about_entity",
            "co_occurring_entities",
            "cross_show_synthesis",
            "related_topics",
        ),
    },
}
_SOFT_STAGES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "diarization — SPOKEN_BY (optional: needs audio)": {
        "any_of": ("SPOKEN_BY",),
        "kills": (
            "person_positions",
            "who_said",
            "position_arc",
            "person_topics",
            "top_people",
            "topic_perspective_leaders",
        ),
    },
}


@dataclass
class IndexStaleness:
    """Served LanceDB index schema vs the code's ``LANCE_SCHEMA_VERSION`` (#1494)."""

    present: bool
    served_version: Optional[int]
    code_version: int
    stale: bool

    @property
    def ok(self) -> bool:
        # A servable index must exist AND match (or exceed) the code schema.
        return self.present and not self.stale

    def reason(self) -> str:
        if not self.present:
            return "no LanceDB index (search/lance_index) — every semantic search returns no_index"
        if self.stale:
            return (
                f"index schema v{self.served_version} < code v{self.code_version} "
                "— stale; read path reports no_index (reindex: make index-two-tier)"
            )
        return f"index schema v{self.served_version} matches code v{self.code_version}"


@dataclass
class MissingStage:
    stage: str
    kills: Tuple[str, ...]


@dataclass
class CompletenessReport:
    index: IndexStaleness
    missing_hard: List[MissingStage] = field(default_factory=list)
    missing_soft: List[MissingStage] = field(default_factory=list)
    edge_types_present: Set[str] = field(default_factory=set)
    has_enrichments: bool = False
    has_topic_clusters: bool = False
    episodes_scanned: int = 0

    @property
    def ok(self) -> bool:
        """Gate verdict: index servable + no HARD stage missing + enrichments + topic clusters.

        ``has_topic_clusters`` only bites when the index is servable (``index.ok``) — i.e. a
        populated corpus. An absent index already fails via ``index.ok``, so a bare/empty
        corpus isn't spuriously flagged for missing clusters.
        """
        return (
            self.index.ok
            and not self.missing_hard
            and self.has_enrichments
            and self.has_topic_clusters
        )


def _collect_edge_types(corpus_root: Path) -> Tuple[Set[str], int]:
    """Union of edge ``type`` values across every ``*.gi.json`` under *corpus_root*.

    Returns (edge_types, gi_files_seen). Unreadable / malformed files are skipped (a partial
    corpus should read as *incomplete*, not crash the gate).
    """
    edge_types: Set[str] = set()
    seen = 0
    for gi_path in sorted(corpus_root.rglob("*.gi.json")):
        seen += 1
        try:
            with open(gi_path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        for edge in doc.get("edges", []) or []:
            if isinstance(edge, dict) and isinstance(edge.get("type"), str):
                edge_types.add(edge["type"])
    return edge_types, seen


def _has_enrichments(corpus_root: Path) -> bool:
    """True when at least one non-empty ``enrichments/`` dir exists under a run."""
    for enr in corpus_root.rglob("enrichments"):
        if enr.is_dir() and any(enr.iterdir()):
            return True
    return False


def _has_topic_clusters(corpus_root: Path) -> bool:
    """True when ``search/topic_clusters.json`` exists AND carries a non-empty ``clusters`` array.

    Mirrors the post-deploy smoke, which fails on BOTH the 404 (missing) and the empty-clusters
    case on a populated corpus — a present-but-empty file would still break the surface.
    """
    p = corpus_root / "search" / TOPIC_CLUSTERS_FILENAME
    if not p.is_file():
        return False
    try:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    clusters = doc.get("clusters") if isinstance(doc, dict) else None
    return isinstance(clusters, list) and len(clusters) > 0


def assess_index_staleness(corpus_root: Path) -> IndexStaleness:
    index_dir = lance_index_dir(corpus_root)
    served = stored_schema_version(index_dir)
    stale = served is not None and served < LANCE_SCHEMA_VERSION
    return IndexStaleness(
        present=served is not None,
        served_version=served,
        code_version=LANCE_SCHEMA_VERSION,
        stale=stale,
    )


def _missing_stages(
    edge_types: Set[str], stages: Dict[str, Dict[str, Tuple[str, ...]]]
) -> List[MissingStage]:
    missing: List[MissingStage] = []
    for stage, spec in stages.items():
        if not any(t in edge_types for t in spec["any_of"]):
            missing.append(MissingStage(stage=stage, kills=spec["kills"]))
    return missing


def assess_completeness(corpus_root: Path) -> CompletenessReport:
    """Read-only completeness assessment of an on-disk corpus root (#1494 + #1497)."""
    resolved = safe_resolve_directory(corpus_root)
    root = Path(resolved) if resolved is not None else Path(corpus_root)

    index = assess_index_staleness(root)
    edge_types, seen = _collect_edge_types(root)
    topic_clusters = _has_topic_clusters(root)
    return CompletenessReport(
        index=index,
        missing_hard=_missing_stages(edge_types, _HARD_STAGES),
        missing_soft=_missing_stages(edge_types, _SOFT_STAGES),
        edge_types_present=edge_types,
        has_enrichments=_has_enrichments(root),
        has_topic_clusters=topic_clusters,
        episodes_scanned=seen,
    )


def format_report(report: CompletenessReport) -> str:
    """Human-readable gate report (used by the CLI / make target)."""
    lines: List[str] = []
    lines.append(f"episodes scanned (*.gi.json): {report.episodes_scanned}")
    lines.append(f"index: {report.index.reason()}")
    enr = "present" if report.has_enrichments else "MISSING (kills corpus_enrichment_signals)"
    lines.append(f"enrichments/: {enr}")
    # search/topic_clusters.json is a query-time-read file the pipeline/prep never generated —
    # its absence 404s /api/corpus/topic-clusters on a populated corpus (the post-deploy smoke
    # rule). Only a fault when the index is servable (populated).
    if report.index.present:
        tc = "present" if report.has_topic_clusters else "MISSING (404s /api/corpus/topic-clusters)"
        lines.append(f"search/topic_clusters.json: {tc}")
    lines.append(f"edge types present: {', '.join(sorted(report.edge_types_present)) or '(none)'}")
    if report.missing_hard:
        lines.append("HARD gaps (fail):")
        for m in report.missing_hard:
            lines.append(f"  ✗ {m.stage} — kills: {', '.join(m.kills)}")
    if not report.has_enrichments:
        lines.append("  ✗ enrichments/ missing — kills: corpus_enrichment_signals")
    if report.index.present and not report.has_topic_clusters:
        lines.append("  ✗ search/topic_clusters.json missing — 404s /api/corpus/topic-clusters")
    if report.missing_soft:
        lines.append("SOFT gaps (warn — optional):")
        for m in report.missing_soft:
            lines.append(f"  ⚠ {m.stage} — degrades: {', '.join(m.kills)}")
    lines.append(f"VERDICT: {'PASS' if report.ok else 'FAIL'}")
    return "\n".join(lines)


def check_corpus(corpus_root: Path) -> Tuple[bool, str]:
    """Convenience: (ok, formatted_report). ``ok`` is the HARD-gate verdict."""
    report = assess_completeness(corpus_root)
    return report.ok, format_report(report)
