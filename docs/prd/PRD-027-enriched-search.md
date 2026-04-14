# PRD-027: Enriched Search

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: TBD
- **Related RFCs**:
  - `docs/rfc/RFC-073-enrichment-layer-architecture.md` -- enrichment layer trust
    contract (`derived: true`, tier classification, opt-in); query-time enricher
    protocol to be defined as Phase 4 extension
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` -- bridge artifact
    that enables chunk-to-Insight lift (the resolution path from FAISS chunks to
    structured GIL Insights)
  - `docs/rfc/RFC-061-semantic-corpus-search.md` -- FAISS index and search API that
    this feature extends
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` -- GIL grounding contract
  - `docs/rfc/RFC-062-gi-kg-viewer-v2.md` -- viewer SPA shell and search panel
- **Related PRDs**:
  - `docs/prd/PRD-021-semantic-corpus-search.md` -- baseline search this feature
    extends; PRD-027 adds enrichment on top, not a replacement
  - `docs/prd/PRD-017-grounded-insight-layer.md` -- GIL artifact foundation
  - `docs/prd/PRD-019-knowledge-graph-layer.md` -- KG artifact foundation
  - `docs/prd/PRD-026-topic-entity-view.md` -- topic pills in enriched sources
    open Topic Entity View
  - `docs/prd/PRD-028-position-tracker.md` -- speaker names in enriched sources
    can open Position Tracker via Person Landing
  - `docs/prd/PRD-029-person-profile.md` -- speaker names in enriched sources can
    open Person Profile via Person Landing
- **Related UX specs**:
  - `docs/uxs/UXS-008-enriched-search.md` -- visual contract for Enriched Answer
    panel, provider attribution, degradation states
  - `docs/uxs/UXS-005-semantic-search.md` -- baseline search panel this extends
  - `docs/uxs/UXS-001-gi-kg-viewer.md` -- shared design system (tokens, typography,
    states)
  - `docs/uxs/UXS-007-topic-entity-view.md` -- topic pill handoff from enriched
    sources to Topic Entity View
  - `docs/uxs/UXS-009-position-tracker.md` -- speaker name handoff from enriched
    sources to Position Tracker
  - `docs/uxs/UXS-010-person-profile.md` -- speaker name handoff from
    enriched sources to Person Profile / Person Landing

---

## Summary

Enriched Search extends the existing semantic search panel (PRD-021) with a
**query-time LLM enrichment layer** that lifts raw FAISS transcript chunks into
structured, grounded, attributed answers.

Without enrichment, a search query returns a list of transcript excerpts -- text
fragments ranked by semantic similarity. The user has to read each fragment and
mentally assemble the answer. With enrichment, the user gets a synthesised answer
with cited sources, speaker attribution, and timestamps -- while the underlying search
results remain available for verification.

This is the **capability showcase** feature: it demonstrates GIL + KG + semantic
search + the enrichment layer working as a coherent system. Every claim in the enriched
answer is traceable to a verbatim quote with a timestamp. The LLM adds readability;
the grounding contract ensures verifiability.

Enriched Search uses the existing LLM provider system (ADR-026) -- the same providers
already configured for summarization and other pipeline features. No separate API key
or provider setup is needed. When no LLM provider is configured, the baseline search
(PRD-021) is unchanged -- no degradation, no error.

---

## Background

PRD-021 (Semantic Corpus Search) shipped a FAISS-based search panel that returns
transcript chunks ranked by embedding similarity. It is useful but raw -- results are
unlabelled text fragments with episode metadata. There is no speaker attribution, no
Insight structure, no synthesis.

RFC-072 (CIL + Bridge) establishes the resolution path: a FAISS chunk with
`(episode_id, char_start, char_end)` can be matched to a GIL Quote node via character
offset overlap, then lifted to an Insight via SUPPORTED_BY, then attributed to a person
via SPOKEN_BY. This path was anticipated in RFC-072 Known Limitations §1 (char offset
alignment must be verified before enabling).

RFC-073 (Enrichment Layer) defines the batch enricher protocol for episode-scope and
corpus-scope enrichers that run after the core pipeline. Query-time enrichment --
running an LLM enricher per-search-request -- is architecturally different: it takes
a query + FAISS results as input and returns a response, rather than writing a file.
This PRD requires a **QueryEnricher** protocol that shares RFC-073's trust contract
(`derived: true`, tier classification, provider opt-in) but operates at request time.
The QueryEnricher protocol will be defined as an RFC-073 Phase 4 extension or a
companion RFC.

**Phase numbering disambiguation:** "RFC-073 Phase 4" refers to the QueryEnricher
protocol extension (request-time enrichment). "RFC-072 Phase 4" refers to CIL query
patterns (cross-layer joins, topic/person queries). These are distinct capabilities
in different RFCs that happen to share the same phase number.

The enricher extends the existing provider system (ADR-026) -- users who already have
an LLM provider configured for summarization or other pipeline features get Enriched
Search with no additional setup. Users without a provider configured get the existing
raw search results unchanged -- no degradation of the baseline.

---

## Goals

1. When an LLM provider is configured and the enricher is enabled, display a
   synthesised answer above raw search results -- grounded, attributed, with
   source citations.
2. Every claim in the answer must be traceable to a specific GIL Insight, verbatim
   Quote, and timestamp.
3. Extend the existing provider system -- any LLM provider already configured for the
   pipeline (OpenAI, Anthropic, Ollama, Gemini) works if it supports completion. No
   separate provider setup; the enricher declares provider compatibility.
4. The baseline search experience (PRD-021) is unchanged when enrichment is disabled
   or unavailable.
5. The user always knows when they are looking at enriched (LLM-derived) output vs
   raw search results -- `derived: true` is surfaced visually.

---

## Non-Goals

- **Not** a replacement for the raw search results -- the FAISS ranked list remains
  visible alongside the enriched answer.
- **Not** a conversational interface or multi-turn Q&A -- single query, single enriched
  response.
- **Not** guaranteed for all queries -- queries that return no grounded Insights produce
  no enriched answer (honest degradation, see FR4).
- **Not** a new search backend -- FAISS remains the retrieval layer; enrichment is
  strictly a response transformation layer on top.
- **Not** cross-show synthesis -- single corpus only.

---

## User Stories

**As a podcast listener**, I want to ask "what does this podcast think about AI
regulation?" and get a synthesised answer with the key positions and who said them,
not a list of raw text fragments I have to read and assemble myself.

**As a researcher**, I want every claim in the answer to link to the exact moment in
the episode where it was said, so I can verify the attribution and listen to the
context.

**As a developer**, I want search enrichment to use the LLM provider I already have
configured for the pipeline -- no separate provider setup for search.

---

## User-Facing Requirements

### Enriched Answer Panel

**FR1.1** -- When enrichment is enabled and the query returns at least one grounded
Insight, display an **Enriched Answer** panel above the raw search results list.

**FR1.2** -- The panel is visually distinct from raw results using a `gi` domain token
border or background tint (consistent with GIL identity in UXS-001). It carries a
visible **"AI-generated · grounded"** badge to signal its derived nature.

**FR1.3** -- The answer text is a synthesised paragraph or short structured response
produced by the LLM enricher. It cites speakers by name and references specific
positions from the grounded Insights.

**FR1.4** -- Below the answer text, a **Sources** section lists all Insights used to
generate the answer. Each source shows: speaker name, Insight text, episode title,
publish date, and a timestamp deep-link (opens episode at that moment if audio is
available, otherwise highlights the transcript segment). Speaker names are clickable
and open the Person Landing (PRD-029) for that person, giving access to their Guest
Brief and Position Tracker. Topic tags on sources are clickable and open the Topic
Entity View (PRD-026).

**FR1.5** -- Sources are collapsible -- the panel defaults to showing 3 sources with
a "Show all N sources" control if more exist.

**FR1.6** -- A **Grounded source count** indicator shows "Based on N grounded
insights" so the user knows how much evidence the answer rests on.

### Provider Configuration and Availability

**FR2.1** -- Enriched Search is enabled only when:
  - An LLM provider is configured in the pipeline config
  - `llm_search_synthesis` enricher is enabled with `opt_in: true` in config
  - The char offset alignment between FAISS chunks and GIL Quotes has been verified
    (RFC-072 Known Limitations §1; the API health endpoint exposes a
    `enriched_search_available` flag)

**FR2.2** -- When enrichment is not available, the search panel is unchanged from
PRD-021. No degraded state, no error, no mention of missing configuration unless the
user explicitly inspects health.

**FR2.3** -- When enrichment is available, a subtle **"Enhanced"** chip appears in the
search panel header indicating enriched results are active.

**FR2.4** -- Provider compatibility: the `llm_search_synthesis` enricher declares
which providers it supports via the enricher manifest. Providers that do not support
enrichment (e.g. a provider that only supports transcription) are excluded silently.
The user configures enrichment via the existing `enrichment` config block (RFC-073):

```yaml
enrichment:
  enabled: true
  enrichers:
    - id: llm_search_synthesis
      enabled: true
      opt_in: true
      provider: openai   # or anthropic, ollama, gemini -- uses existing provider config
```

**RFC-073 extension note:** The `provider` field per enricher entry is new -- RFC-073's
current config schema does not include per-enricher provider selection. Batch enrichers
(deterministic tier) do not need it. LLM-tier and query-time enrichers do. This field
will be added as part of the QueryEnricher protocol extension (RFC-073 Phase 4).

### Degraded States

**FR3.1** -- When the query returns FAISS results but no grounded Insights can be
lifted (chunk-to-Insight lift returns empty), the Enriched Answer panel is hidden
entirely. Raw results are shown as normal.

**FR3.2** -- When the LLM call fails (timeout, provider error), the Enriched Answer
panel shows a minimal error state: "Enrichment unavailable for this query." Raw
results remain unaffected.

**FR3.3** -- When all lifted Insights are `grounded: false`, no enriched answer is
generated. The panel is hidden. The system does not synthesise from ungrounded
Insights -- this is a hard rule from the grounding contract.

**FR3.4** -- Latency: if the enricher has not responded within 5 seconds, display a
loading indicator in the Enriched Answer panel. Raw results are shown immediately
while the enricher works.

### Integration with Existing Search Panel

**FR4.1** -- The existing **Search result insights** modal (UXS-005) is unchanged.
Enriched Search adds a panel *above* the results list, not replacing or modifying
existing search UI elements.

**FR4.2** -- The **Advanced search** modal (UXS-005) gains a new toggle:
**"Enriched answers"** -- on/off, defaults to on when enrichment is configured.
This allows users to disable enrichment for a specific query without changing config.

**FR4.3** -- Grounded-only search hits: when enrichment is active, the existing
**"Grounded insights only"** filter in Advanced search gains additional meaning --
filtering for grounded-only results improves enrichment quality by ensuring the lift
only operates on high-quality chunks.

### Trust and Provenance

**FR5.1** -- The enriched answer panel always shows the LLM provider used (e.g.
"Synthesised by OpenAI gpt-4o-mini") in a muted caption below the answer.

**FR5.2** -- The panel carries a one-line explanation: "This answer was generated from
grounded insights in your corpus. All sources are verbatim quotes with timestamps."

**FR5.3** -- Sources that were used in the answer are visually linked to their
corresponding raw search result cards (highlight or badge) so users can see which
raw results contributed to the synthesis.

---

## Data Flow

```
User query
    ↓
FAISS retrieval (existing -- RFC-061)
    ↓
Raw results returned immediately -> shown in results list
    ↓
Bridge resolution (RFC-072) -- match chunks to GIL Quotes via char offsets
    ↓
GIL lift -- Quotes -> Insights (SUPPORTED_BY) -> Persons (SPOKEN_BY) -> Topics (ABOUT)
    ↓
Filter: grounded: true only
    ↓
QueryEnricher: llm_search_synthesis (RFC-073 Phase 4 extension)
    ↓
Enriched answer with sources
    ↓
Shown in Enriched Answer panel above raw results
```

---

## API Requirements

**Extended endpoint:**

`POST /api/search` -- existing endpoint gains an optional `enrich: bool` parameter
(default: follows config). When `enrich: true` and enrichment is available, the
response includes an `enriched` block alongside the existing `results` array:

```json
{
  "query": "what does this podcast think about AI regulation",
  "results": [ ],
  "enriched": {
    "available": true,
    "answer_text": "The podcast has discussed AI regulation across 14 episodes...",
    "provider": "openai/gpt-4o-mini",
    "grounded_source_count": 6,
    "derived": true,
    "sources": [
      {
        "person_id": "person:lex-fridman",
        "person_name": "Lex Fridman",
        "insight_text": "AI regulation will lag innovation by 3-5 years",
        "quote_text": "Regulation will lag innovation. That's my prediction.",
        "timestamp_start_ms": 120000,
        "episode_id": "episode:abc123",
        "episode_title": "AI Regulation",
        "publish_date": "2024-03-10"
      }
    ]
  }
}
```

When enrichment is unavailable or the query produces no grounded Insights:
```json
{
  "enriched": {
    "available": false,
    "reason": "no_grounded_insights"
  }
}
```

**New health flag:**

`GET /api/health` gains `enriched_search_available: bool` -- true when the enricher
is configured, opted in, and char offset alignment has been verified. The viewer uses
this to show/hide the "Enhanced" chip (FR2.3) and disable the enriched toggle in
Advanced search when unavailable.

---

## Success Criteria

1. A user with an LLM provider configured and enrichment opted in receives a
   synthesised answer above search results for queries that have grounded Insights
   in the corpus.
2. Every source citation in the enriched answer is traceable to a specific GIL
   Insight and verbatim Quote in the corpus.
3. The enricher never synthesises from `grounded: false` Insights -- verified by test.
4. The baseline search experience (PRD-021) is pixel-identical when enrichment is
   disabled or unavailable.
5. The enriched answer panel degrades gracefully for all failure modes (no grounded
   Insights, LLM timeout, provider error) -- raw results unaffected.
6. The existing `Advanced search` modal works correctly with the new
   "Enriched answers" toggle.
7. At least 3 LLM providers work as enrichment backends: OpenAI, Anthropic, and
   Ollama (local).
8. Response latency: raw results appear in < 500ms; enriched answer appears within
   5 seconds (or shows loading indicator).
9. Playwright E2E coverage: enriched state (mocked enricher response), disabled state,
   degraded state (no grounded Insights).

---

## Dependencies and Prerequisites

1. **Char offset alignment (hard prerequisite)** -- RFC-072 Known Limitations §1
   must be verified before enriched search can ship. FAISS chunk `(char_start,
   char_end)` must overlap with GIL Quote `(char_start, char_end)` for the lift to
   work. If offsets diverge, the lift returns empty and enrichment never fires. This
   is a verification task against the eval corpus, not a code change. Enriched Search
   cannot ship until this is confirmed green.

2. **RFC-073 Phase 4: QueryEnricher protocol** -- The batch enricher protocol
   (episode/corpus scope) does not cover request-time enrichment. A QueryEnricher
   protocol extension must be defined before `llm_search_synthesis` can be
   implemented. This PRD defines the product requirements; the protocol RFC is a
   separate deliverable.

3. **LLM provider configured** -- The user's pipeline config must include at least
   one LLM provider (the same one used for summarization or other features).
   Enriched Search is inert without it -- no separate setup required.

---

## Open Questions

1. **Answer length** -- Should the enriched answer be a paragraph (conversational) or
   a structured list of positions? The LLM prompt in `llm_search_synthesis` controls
   this. A structured list may be more trustworthy (each bullet cites a source) but
   less readable. Recommend: structured list with sources initially, paragraph mode
   as a config option later.

2. **Provider declaration** -- Not all providers support the completion call shape
   needed by `llm_search_synthesis`. The enricher manifest declares
   `supported_providers`. Whether to validate this at config load time or silently
   skip unsupported providers at runtime is an implementation decision for the
   QueryEnricher protocol RFC.

---

## References

- `docs/rfc/RFC-073-enrichment-layer-architecture.md`
- `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
- `docs/rfc/RFC-061-semantic-corpus-search.md`
- `docs/prd/PRD-021-semantic-corpus-search.md`
- `docs/uxs/UXS-001-gi-kg-viewer.md`
- `docs/architecture/gi/ontology.md`
