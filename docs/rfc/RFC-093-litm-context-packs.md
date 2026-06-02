# RFC-093: LITM-Aware MCP Context Packs

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Core team
- **Related PRDs**:
  - `docs/prd/PRD-032-hybrid-corpus-search.md` — hybrid corpus search
  - `docs/prd/PRD-031-search.md` — Search product (agent endpoint, V3)
- **Related RFCs**:
  - `docs/rfc/RFC-090-hybrid-retrieval.md` — produces the results this packages
  - `docs/rfc/RFC-091-kg-proximity-signal.md` — contradiction/relational inputs
  - `docs/rfc/RFC-088-enrichment-layer-architecture.md` — contradiction signals
- **Related Documents**:
  - _(MCP tool layer — does not exist yet; see Constraints)_

> **Stabilization note (2026-05-30):** Split out of an earlier combined draft (RFC-079). This is the
> LITM-context-pack / MCP-tool concern only. **Blocking prerequisite:** there is no MCP tool layer
> in this repo (`src/podcast_scraper/mcp/` does not exist, no `@mcp_tool` decorator). The original
> draft referenced an "RFC-075 MCP integration layer" that does not exist here — that layer is
> **[TBD — not yet specified]** and must be defined before this RFC can land its tool. The pack
> *builder* (below) is implementable independently of MCP and is useful to the autoresearch loop
> directly.

---

## Abstract

Reshape RFC-090 retrieval output into a structured, token-budgeted, LITM-positioned context
document for agent consumers. LITM (Lost-In-The-Middle) is well-validated: models attend poorly to
the middle of long contexts. A `corpus_briefing_pack` builder positions critical grounding at the
start and caveats at the end, extracting materially more value from the same retrieval results. The
builder is plain Python; exposing it as an MCP tool requires an MCP layer that does not exist yet.

**Architecture Alignment:** `build_briefing_pack()` consumes `RetrievalLayer` output
(`ScoredResult` / `CompoundResult`) and produces a `CorpusBriefingPack` dataclass — no retrieval
change. The MCP wrapper is additive and gated on the MCP layer.

## Problem Statement

RFC-075-style MCP tools (in the original draft's world) returned raw `ScoredResult` lists. Agents
consuming raw lists face: no signal about what to attend to first, LITM degradation on long lists,
no token-budget management, and no distinction between grounded insight and raw evidence. A
briefing pack solves all four by selecting and positioning results into a fixed attention-aware
structure within a token budget.

In this repo there is an additional gap: **no MCP layer exists** to host the tool. The pack builder
is still valuable standalone — the autoresearch loop can call it directly.

**Use Cases:**

1. **Research synthesis**: an agent preparing a briefing gets grounded intelligence, not a list to
   re-rank itself.
2. **Token-budgeted context**: the pack fits a `max_tokens` budget, positioning the strongest
   grounding where attention is highest.
3. **Direct autoresearch use**: the loop calls the builder without an MCP round-trip.

## Goals

1. **LITM-structured pack**: critical grounding first, supporting evidence middle, caveats last.
2. **Token-budgeted**: select within `max_tokens`; report actual usage.
3. **Tier-aware**: distinguish grounded insight from raw segment; unpack compound results.
4. **MCP-ready (gated)**: a thin `corpus_briefing_pack` tool once an MCP layer exists.

## Constraints & Assumptions

**Constraints — Prerequisites:**

1. **MCP tool layer does not exist** (`src/podcast_scraper/mcp/` absent; no `@mcp_tool`). The tool
   wrapper is blocked on defining that layer **[TBD — not yet specified]**. The builder is not
   blocked.
2. **Contradiction content** (`top_contradiction`) requires typed contradiction KG edges, which do
   not exist yet (RFC-088 / RFC-091 OQ). The field is `None` until then.
3. **Coverage gaps** require a corpus-impact / coverage surface that does not exist yet
   **[TBD]**; the field is empty until then.

**Assumptions:**

- `RetrievalLayer` output carries `source_tier` and per-result `payload` (`confidence`, `show_id`,
  `episode_id`, dates) sufficient to compute coverage and confidence summaries.

## Design & Implementation

### 1. LITM Structure

```text
[WINDOW START — high attention]
  Critical grounding:
  - Canonical entity definition (who/what is being asked about)
  - Strongest grounded insight (highest confidence, most relevant)
  - Most significant contradiction (if any)   <- None until typed edges exist

[MIDDLE — lower attention, supporting detail]
  Supporting evidence:
  - Top-N segment results (raw evidence with timestamps)
  - Cross-show coverage summary (which shows, how many episodes)

[WINDOW END — high attention]
  Caveats and metadata:
  - Coverage gaps (topics/shows not represented)   <- empty until coverage surface exists
  - Confidence distribution
  - Temporal range of evidence
```

### 2. Pack Builder

```python
# src/podcast_scraper/search/context_pack.py

@dataclass
class CorpusBriefingPack:
    query: str
    query_type: str
    canonical_entity: dict | None          # RFC-072 canonical entity if resolved
    top_insight: ScoredResult | None
    top_contradiction: dict | None         # from typed KG edges (not yet built)
    supporting_segments: list[ScoredResult]
    coverage_summary: dict                  # show_ids, episode_count, date_range
    coverage_gaps: list[str]                # from corpus-impact surface (not yet built)
    confidence_p50: float
    token_count: int

def build_briefing_pack(query, query_type, results, canonical_entity, max_tokens=2000):
    """Select and position results into a LITM-aware structure within token budget."""
    insights  = [r for r in results if getattr(r, "source_tier", "") == "insight"]
    segments  = [r for r in results if getattr(r, "source_tier", "") == "segment"]
    compounds = [r for r in results if getattr(r, "source_tier", "") == "compound"]

    all_insights = insights + [r.insight for r in compounds]
    all_segments = segments + [r.segment for r in compounds]
    top_insight = all_insights[0] if all_insights else None

    coverage = {
        "show_ids": list({r.payload.get("show_id") for r in results}),
        "episode_count": len({r.payload.get("episode_id") for r in results}),
        "date_range": _compute_date_range(results),
    }
    confs = [r.payload.get("confidence", 0) for r in all_insights if r.payload.get("confidence")]
    p50 = sorted(confs)[len(confs) // 2] if confs else 0.0

    return CorpusBriefingPack(
        query=query, query_type=query_type, canonical_entity=canonical_entity,
        top_insight=top_insight,
        top_contradiction=None,              # until typed contradiction edges exist
        supporting_segments=all_segments[:5],
        coverage_summary=coverage,
        coverage_gaps=[],                    # until corpus-impact surface exists
        confidence_p50=p50, token_count=0,   # set after serialisation
    )
```

### 3. MCP Tool (gated on MCP layer)

```python
# src/podcast_scraper/mcp/tools/corpus_briefing_pack.py   <- requires MCP layer (does not exist)

@mcp_tool(
    name="corpus_briefing_pack",
    description="""
    Returns a LITM-aware, token-budgeted context pack for a corpus query.
    Use instead of raw search results for research synthesis or briefing prep.
    Returns: canonical entity, strongest grounded insight, detected contradictions,
    supporting raw evidence, coverage summary, confidence distribution.
    When NOT to use: exhaustive raw results, or debugging the retrieval pipeline —
    use corpus_search instead.
    """,
)
def corpus_briefing_pack(query: str, max_tokens: int = 2000) -> CorpusBriefingPack:
    query_type = router.classify(query)
    # NOTE: the shipped RetrievalLayer.retrieve is keyword-only and intent-driven —
    #   retrieve(query, embed(query), intent=query_type)
    # It derives weights from `intent` internally; query_type=/signal_weights= below
    # are the original sketch and do not match the real signature.
    results = retrieval_layer.retrieve(
        text=query, embedding=embed(query), query_type=query_type,
        signal_weights=SIGNAL_WEIGHTS[query_type],
        tier_weights=TIER_WEIGHTS_BY_QUERY[query_type],
    )
    canonical_entity = entity_resolver.resolve(query)
    return build_briefing_pack(query, query_type, results, canonical_entity, max_tokens)
```

## Key Decisions

1. **Builder decoupled from MCP.**
   - **Decision**: `build_briefing_pack()` is plain Python over `RetrievalLayer` output.
   - **Rationale**: Usable by autoresearch directly; not blocked on the MCP layer.
2. **Packs in RFC-093, not RFC-090.**
   - **Decision**: Defer packs to this RFC.
   - **Rationale**: Full value depends on KG proximity (contradictions) and the router
     (query_type accuracy); shipping in RFC-090 would produce worse packs.
3. **Contradiction/coverage fields present but null until inputs exist.**
   - **Decision**: Keep `top_contradiction` / `coverage_gaps` in the schema, populated later.
   - **Rationale**: Stable consumer schema; no breaking change when typed edges / coverage surface
     land.

## Alternatives Considered

1. **Return raw `ScoredResult` lists to agents.**
   - **Pros**: Simplest.
   - **Cons**: LITM degradation; no budget; agents re-rank themselves.
   - **Why rejected**: The whole point is shaped, attention-aware context.
2. **Generate the pack with an LLM summarisation call.**
   - **Pros**: Fluent prose.
   - **Cons**: Cost/latency; hallucination risk; loses provenance.
   - **Why rejected**: Packs must be extractive and grounded, not generated.

## Testing Strategy

**Test Coverage:**

- **Unit**: pack structure (sections populated in order), compound unpacking into insight/segment,
  coverage/confidence computation, null-safety for contradiction/coverage fields.
- **Integration**: token-budget enforcement; pack built end-to-end from `RetrievalLayer` output on
  a fixture corpus.

**Test Organization:** `tests/unit/podcast_scraper/search/test_context_pack.py`.

**Test Execution:** Unit in `ci-fast`; no MCP integration test until the MCP layer exists.

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1 — Builder**: `CorpusBriefingPack` + `build_briefing_pack()` + serialisation/token
  counting; wire into autoresearch as a direct consumer.
- **Phase 2 — MCP tool (gated)**: once an MCP layer is defined, add the `corpus_briefing_pack`
  wrapper.
- **Phase 3 — Enrichment**: populate `top_contradiction` (typed edges) and `coverage_gaps`
  (corpus-impact surface) when those exist.

**Monitoring:** Track actual token usage vs budget; track agent task quality with packs vs raw
lists in the eval loop.

**Success Criteria:** Packs stay within budget; agents using packs outperform agents using raw
lists on the eval task set.

## Relationship to Other RFCs

**Key Distinction:**

- **RFC-090/091/092**: produce and rank results.
- **RFC-093 (this)**: packages results for agent attention; does not retrieve.

| RFC | Relationship |
| --- | --- |
| RFC-090 | Provides `RetrievalLayer` output consumed by the builder |
| RFC-091 | Contradiction/relational inputs (when typed edges exist) |
| RFC-088 | Enrichment produces contradiction signals to model as edges |

## Benefits

1. **More value per token** for agents via LITM positioning.
2. **Provenance-preserving** — extractive, not generated.
3. **Stable schema** — contradiction/coverage fields slot in later without breaking consumers.

## Migration Path

1. **Phase 1**: Ship the builder; autoresearch consumes it directly.
2. **Phase 2**: Define the MCP layer (separate RFC), then add the tool wrapper.
3. **Phase 3**: Populate contradiction/coverage fields as their inputs land.

## Open Questions

1. **OQ-1 MCP layer definition.** This repo has no MCP layer. Define it (separate RFC/issue) before
   the tool wrapper; until then the builder is autoresearch-only.
2. **OQ-2 Contradiction-edge dependency.** `top_contradiction` is `None` until typed contradiction
   KG edges exist. Define the contradiction-as-edge RFC (extension to RFC-088) or this field is
   permanently null.
3. **OQ-3 Token counting accuracy.** Which tokenizer defines the budget (consumer-model-specific)?
   Pick one canonical tokenizer for `token_count`.

## References

- **Related PRD**: `docs/prd/PRD-032-hybrid-corpus-search.md`
- **Related RFC**: `docs/rfc/RFC-090-hybrid-retrieval.md`, `docs/rfc/RFC-091-kg-proximity-signal.md`,
  `docs/rfc/RFC-088-enrichment-layer-architecture.md`
- **Source Code**: `src/podcast_scraper/search/` (RetrievalLayer output), `autoresearch/` (direct
  consumer); MCP layer — does not exist yet
