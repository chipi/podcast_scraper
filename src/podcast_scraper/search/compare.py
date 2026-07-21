"""Search v3 §S8 — Compare (2 subjects) operator.

Orchestrates ``build_briefing_pack`` twice: one call per picker slot on
the client, one subject-scoped search per side. Response shape is
``{pack_a, pack_b, judge_summary}`` — the judge summary is a
deterministic comparison string over the two packs, muted when either
side reports ``grounded=false`` (RFC-107 §S8 acceptance).

Never an LLM call — CI stays airgapped
(``feedback_no_llm_in_ci.md``). Never re-enables ``_combine_hybrid_results``
/ ``_normalize_scores`` — the whole compare pipeline runs in Python
after ``rrf_fuse`` returns (SIGSEGV guard #1205).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

SubjectKind = Literal["person", "topic", "episode", "feed", "show"]


@dataclass
class SubjectRef:
    """One picker slot — plain-Python mirror of ``CompareSubjectRefModel``."""

    kind: SubjectKind
    id: str
    label: Optional[str] = None


@dataclass
class BriefingPack:
    """One side of a compare — plain-Python mirror of ``CompareBriefingPackModel``."""

    subject: SubjectRef
    query: str
    query_type: str = "semantic"
    rendered: str = ""
    token_count: int = 0
    max_tokens: int = 2000
    top_insight_id: Optional[str] = None
    top_insight_text: str = ""
    supporting_segment_ids: List[str] = field(default_factory=list)
    supporting_segment_texts: List[str] = field(default_factory=list)
    coverage_summary: Dict[str, Any] = field(default_factory=dict)
    confidence_p50: float = 0.0
    result_count: int = 0
    grounded: bool = False


@dataclass
class CompareOutcome:
    """Compare orchestrator result — mirrors ``SearchCompareResponse``."""

    pack_a: BriefingPack
    pack_b: BriefingPack
    judge_summary: Optional[str] = None
    error: Optional[str] = None
    detail: Optional[str] = None


def _scope_for_subject(subject: SubjectRef) -> Dict[str, Any]:
    """Map subject kind → ``structured_corpus_search`` scope kwargs."""
    kind = subject.kind
    subject_value = subject.id.strip()
    if kind == "person":
        # ``speaker=`` is substring over the resolved speaker field on
        # insight / quote / transcript hits. The client typically passes
        # the display name here.
        return {"speaker": subject_value}
    if kind == "topic":
        # ``topic=`` is substring over kg_topic id/text + insight
        # ABOUT-edge topic labels — accepts either ``topic:foo`` or
        # ``Foo`` (label).
        return {"topic": subject_value}
    if kind == "episode":
        return {"episode_id": subject_value}
    # feed / show — same underlying scope, one is an alias.
    return {"feed": subject_value}


def _query_for_side(subject: SubjectRef, shared_q: str) -> str:
    """Choose the query used for this side.

    When the caller supplied a shared query it wins. Otherwise fall back
    to the subject label (or bare id) so the retrieval has something to
    embed. Empty label + no shared q returns an empty string; the search
    layer surfaces ``empty_query`` and the pack renders as ungrounded.
    """
    shared = shared_q.strip()
    if shared:
        return shared
    return (subject.label or subject.id).strip()


def _pack_from_search_dicts(
    subject: SubjectRef,
    query: str,
    query_type: str,
    rows: Sequence[Dict[str, Any]],
    max_tokens: int,
) -> BriefingPack:
    """Adapt search dicts → typed ``ScoredResult`` → ``build_briefing_pack`` output.

    Mirrors the MCP ``corpus_briefing_pack`` adapter to keep the two
    entry points shape-compatible. Lazy-import so unit tests can stub
    the retrieval layer without pulling in the backend on import.
    """
    from .backend import ScoredResult
    from .context_pack import build_briefing_pack

    typed: List[ScoredResult] = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        payload: Dict[str, Any] = {"text": row.get("text") or "", **metadata}
        if row.get("supporting_quotes"):
            payload["supporting_quotes"] = row["supporting_quotes"]
        if row.get("lifted"):
            payload["lifted"] = row["lifted"]
        typed.append(
            ScoredResult(
                doc_id=str(row.get("doc_id", "")),
                score=float(row.get("score") or 0.0),
                rank=int(row.get("rank") or 0),
                payload=payload,
                signal=str(row.get("signal", "rrf")),
                source_tier=str(row.get("source_tier", "segment")),
            )
        )
    pack = build_briefing_pack(query, query_type, typed, max_tokens=max_tokens)
    return BriefingPack(
        subject=subject,
        query=query,
        query_type=pack.query_type,
        rendered=pack.render(),
        token_count=pack.token_count,
        max_tokens=max_tokens,
        top_insight_id=pack.top_insight.doc_id if pack.top_insight else None,
        top_insight_text=(
            str(pack.top_insight.payload.get("text", "")) if pack.top_insight else ""
        ),
        supporting_segment_ids=[s.doc_id for s in pack.supporting_segments],
        supporting_segment_texts=[str(s.payload.get("text", "")) for s in pack.supporting_segments],
        coverage_summary=pack.coverage_summary,
        confidence_p50=pack.confidence_p50,
        result_count=len(typed),
        grounded=pack.top_insight is not None,
    )


def _ungrounded_pack(subject: SubjectRef, query: str, max_tokens: int) -> BriefingPack:
    """Return a placeholder pack when retrieval failed / returned nothing."""
    return BriefingPack(
        subject=subject,
        query=query,
        max_tokens=max_tokens,
    )


def _judge_summary(pack_a: BriefingPack, pack_b: BriefingPack) -> Optional[str]:
    """Deterministic comparison sentence — NO LLM call.

    Muted when either side is ungrounded (RFC-107 §S8 acceptance).
    Compares confidence p50, episode coverage, and top-insight score
    across the two sides. Keeps CI airgapped.
    """
    if not (pack_a.grounded and pack_b.grounded):
        return None
    a_label = pack_a.subject.label or pack_a.subject.id
    b_label = pack_b.subject.label or pack_b.subject.id
    a_eps = int(pack_a.coverage_summary.get("episode_count") or 0)
    b_eps = int(pack_b.coverage_summary.get("episode_count") or 0)
    a_conf = float(pack_a.confidence_p50)
    b_conf = float(pack_b.confidence_p50)
    if a_conf > b_conf:
        conf_bit = f"{a_label} shows higher confidence ({a_conf:.2f} vs {b_conf:.2f} for {b_label})"
    elif b_conf > a_conf:
        conf_bit = f"{b_label} shows higher confidence ({b_conf:.2f} vs {a_conf:.2f} for {a_label})"
    else:
        conf_bit = f"{a_label} and {b_label} at equal confidence ({a_conf:.2f})"
    cov_bit = f"episode coverage {a_eps} ({a_label}) vs {b_eps} ({b_label})"
    return f"{conf_bit}; {cov_bit}."


def compare_subjects(
    root: Path,
    subject_a: SubjectRef,
    subject_b: SubjectRef,
    *,
    q: str = "",
    top_k: int = 10,
    max_tokens: int = 2000,
) -> CompareOutcome:
    """Build briefing packs for *subject_a* and *subject_b* over the corpus.

    Runs ``structured_corpus_search`` twice (one subject scope per call),
    feeds each result set through ``build_briefing_pack``, and returns a
    deterministic judge summary when both sides are grounded.
    """
    from .capability import structured_corpus_search

    def _one_side(subject: SubjectRef) -> BriefingPack:
        query = _query_for_side(subject, q)
        if not query:
            return _ungrounded_pack(subject, query, max_tokens)
        scope = _scope_for_subject(subject)
        outcome = structured_corpus_search(
            root,
            query,
            top_k=top_k,
            **scope,
        )
        if outcome.get("error"):
            return _ungrounded_pack(subject, query, max_tokens)
        rows: List[Dict[str, Any]] = list(outcome.get("results") or [])
        if not rows:
            return _ungrounded_pack(subject, query, max_tokens)
        query_type = str(outcome.get("query_type") or "semantic")
        return _pack_from_search_dicts(subject, query, query_type, rows, max_tokens)

    pack_a = _one_side(subject_a)
    pack_b = _one_side(subject_b)
    return CompareOutcome(
        pack_a=pack_a,
        pack_b=pack_b,
        judge_summary=_judge_summary(pack_a, pack_b),
    )
