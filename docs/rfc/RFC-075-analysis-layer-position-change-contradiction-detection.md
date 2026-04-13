# RFC-075: Analysis Layer -- Position Change & Contradiction Detection

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Core team
- **Related PRDs**:
  - `docs/prd/PRD-028-position-tracker.md` -- defines the `position_change_detected`
    and `stance_summary` field contracts that this RFC must implement
  - `docs/prd/PRD-029-guest-intelligence-brief.md` -- defines the
    `potential_challenges` field contract that this RFC must implement
- **Related RFCs**:
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` -- provides the
    data foundation (bridge, CIL queries, Phases 1--5) that this RFC operates on
  - `docs/rfc/RFC-073-enrichment-layer-architecture.md` -- `nli_contradiction`
    enricher pre-computes contradiction candidate pairs (base layer); this RFC
    refines them at query time
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` -- GIL Insight and Quote nodes
- **Related UX specs**:
  - `docs/uxs/UXS-009-position-tracker.md` -- Phase 6 UI: stance summary banner,
    position-change markers
  - `docs/uxs/UXS-010-guest-intelligence-brief.md` -- Phase 6 UI: Potential
    Challenges section

---

## Abstract

This RFC defines the **analysis layer** that operates on data collected by RFC-072
(Phases 1--5) and pre-computed by RFC-073's batch enrichers. It covers two
capabilities:

1. **Position change detection** -- flagging episodes where a person's stated
   position on a topic contradicts or evolves from a prior episode.
2. **Contradiction detection** -- identifying topics where different persons hold
   conflicting positions, surfaced as "potential challenges" in the Guest Brief.

Both capabilities use a dual-layer strategy: RFC-073's `nli_contradiction` enricher
pre-computes candidate pairs to disk (batch, no LLM); this RFC adds query-time
refinement with human-readable summaries and ranking (LLM or NLI).

---

## Problem Statement

RFC-072 Phases 1--5 collect and join data across episodes -- position arcs (person +
topic over time) and guest briefs (person dossier). But the flagship use cases lose
their "wow" moment without automated analysis:

- The Position Tracker can show that a person said X in 2023 and Y in 2025, but
  cannot flag that Y contradicts X. The user must read every Insight and mentally
  detect the shift.
- The Guest Brief can list all positions and quotes, but cannot identify which
  positions conflict with other guests. Interview prep requires manual cross-referencing.

The analysis layer bridges the gap between *collection* (what RFC-072 provides) and
*intelligence* (what the user actually wants).

**Use cases:**

1. **Position shift flagging**: Automatically mark episodes where a person's stance
   changed, so the user sees the pivotal moment without reading every entry.
2. **Stance summarisation**: Generate a 1--3 sentence summary of how a person's
   position evolved over time.
3. **Cross-person contradiction surfacing**: Identify topic + person pairs where
   stated positions conflict, enabling interview challenge preparation.

---

## Goals

1. Implement position change detection on assembled position arcs (RFC-072
   Pattern A output).
2. Implement cross-person contradiction detection on guest briefs (RFC-072
   Pattern B output).
3. Use the dual-layer strategy: consume RFC-073 `nli_contradiction` enricher output
   as the base signal; add query-time LLM refinement for summaries and ranking.
4. Populate the field contracts defined in PRD-028 (`position_change_detected`,
   `stance_summary`) and PRD-029 (`potential_challenges`).
5. Define eval requirements and golden test sets for both capabilities.
6. Carry `derived: true` on all analysis output with LLM provider attribution.

---

## Constraints and Assumptions

**Constraints:**

- Must run on Apple M4 Pro with 48 GB RAM (local-first tool).
- Query-time analysis must complete in < 5 seconds for a single position arc or
  guest brief request.
- No database -- reads from filesystem artifacts (bridge, GIL, enrichments).
- LLM provider must already be configured in the pipeline (same providers as
  summarisation; no separate setup).

**Assumptions:**

- RFC-072 Phases 1--4 have been run on a real corpus so extraction quality is
  visible.
- RFC-073 `nli_contradiction` enricher has run (produces the base candidate pairs).
  If not, the analysis layer produces empty results (graceful degradation).
- A golden eval set exists in `tests/fixtures/cil_phase6_golden/` with
  human-labelled verdicts.

---

## Prerequisites

This RFC should **not be implemented** until:

1. RFC-072 Phases 1--4 have been run on a real corpus and extraction quality is
   assessed (Known Limitations 2--3 in RFC-072).
2. RFC-073 Phase 3 (`nli_contradiction` enricher) has shipped and produced candidate
   pairs on the eval corpus.
3. The eval corpus contains enough multi-episode persons with evolving positions to
   validate change detection.

---

## Design

### Dual-Layer Architecture

```text
Batch layer (RFC-073, runs during enrichment pipeline):
    nli_contradiction enricher
        -> candidate pairs to disk
        -> {topic_id, person_a_id, person_b_id,
            insight_a_id, insight_b_id, contradiction_score}

Query-time layer (this RFC, runs per API request):
    Position arc request or Guest Brief request
        -> read enricher candidate pairs from disk
        -> filter to relevant person/topic
        -> LLM refinement pass (optional):
            -> position summaries
            -> ranking
            -> human-readable change descriptions
        -> populate response fields
```

The batch layer provides the **signal** (which pairs are contradictory). The
query-time layer provides the **intelligence** (what changed, who disagrees, and
why it matters).

### Position Change Detection (PRD-028 contract)

**Input:** The assembled position arc (list of episodes with Insights for
person + topic).

**Output per episode** (after the first):

- `position_change_detected: bool` -- true when the person's stance in this
  episode contradicts or meaningfully evolves from their stance in a prior episode.
- `change_description: string | null` -- optional explanation of what changed.

**Output per arc:**

- `stance_summary: string | null` -- a 1--3 sentence summary of the position
  evolution across all episodes.

**Mechanism options** (to be decided during implementation):

- **LLM-as-judge**: Prompt an LLM with the previous and current episode's Insights,
  ask whether the position changed. Pros: flexible, handles nuance. Cons: latency,
  cost.
- **NLI-based**: Entailment/contradiction scoring between Insight pairs from
  consecutive episodes. Pros: fast, deterministic. Cons: requires good NLI model,
  struggles with nuance.
- **Embedding + stance**: Compute stance embeddings and flag when cosine distance
  exceeds a threshold. Pros: fast. Cons: less interpretable.
- **Hybrid**: NLI for detection, LLM for description/summary. Best of both.

`insight_type` scoping: Position change detection is most meaningful when scoped
to `claim` vs `claim` pairs (RFC-072 Section 2a). Comparing a `claim` to an
`observation` rarely constitutes a position change.

### Contradiction Detection (PRD-029 contract)

**Input:** For each topic a person discusses, the set of Insights from other persons
on the same topic.

**Output:** A list of `ChallengeEntry` objects (PRD-029 FR1.7):

```json
{
  "topic": {"id": "topic:ai-regulation", "display_name": "AI Regulation"},
  "this_guest_position": "AI regulation will lag innovation; self-regulation preferred",
  "conflicting_guest": {"id": "person:timnit-gebru", "display_name": "Timnit Gebru"},
  "conflicting_position": "Voluntary safety commitments are insufficient; binding regulation is essential",
  "episodes": ["episode:ghi789"],
  "derived": true
}
```

**Three states for `potential_challenges`:**

1. Empty array -- no enricher has run.
2. Partially populated -- RFC-073 enricher ran but no query-time LLM refinement.
   Raw candidate pairs with Insight text, no `this_guest_position` /
   `conflicting_position` summaries.
3. Fully populated -- enricher + this RFC's analysis. Candidate pairs with
   human-readable summaries and ranking.

**Pruning strategy** (corpus scale):

At corpus scale, pairwise comparison is expensive. This RFC must define a pruning
strategy:

- Only compare persons who share >= 2 topics.
- Only compare `claim`-type Insights (via `insight_type`).
- Use `contradiction_score` from the enricher to filter weak candidates
  (threshold TBD via eval).

### Stance Summarisation

Both the position arc `stance_summary` and the contradiction `this_guest_position` /
`conflicting_position` fields are short summaries generated by an LLM.

- Constrained to cite specific episodes and dates.
- Carries `derived: true` and LLM provider attribution.
- The summarisation mechanism and prompt design are implementation details of this
  RFC.

---

## Eval Requirements

### Position Change Detection

A golden eval set of Insight pairs with human-labelled verdicts:

- **Same-person shift**: Insight A (2023) vs Insight B (2025) on the same topic.
  Labels: `changed` / `unchanged` / `nuanced`.
- Minimum 30 labelled pairs for initial validation.
- Placeholder directory: `tests/fixtures/cil_phase6_golden/position_change/`.

### Contradiction Detection

A golden eval set of cross-person Insight pairs:

- **Cross-person disagreement**: Person A's Insight vs Person B's Insight on the
  same topic. Labels: `agree` / `disagree` / `unrelated`.
- Minimum 30 labelled pairs for initial validation.
- Placeholder directory: `tests/fixtures/cil_phase6_golden/contradiction/`.

### Eval Framework

Both eval sets feed into the existing eval framework. Metrics:

- Precision and recall for detection (binary classification).
- Human preference rating for generated summaries (1--5 scale).
- Latency: query-time analysis must complete in < 5 seconds.

---

## Key Decisions

1. **Dual-layer over pure query-time**
   - **Decision**: Use RFC-073 batch enricher for candidate pre-computation; add
     query-time LLM pass for refinement.
   - **Rationale**: Pure query-time pairwise comparison at corpus scale is too slow.
     Pre-computing candidates to disk makes the query-time pass a filter + refine
     operation on a small set, not a full corpus scan.

2. **`insight_type` scoping for contradiction detection**
   - **Decision**: Scope contradiction detection to `claim` vs `claim` pairs.
   - **Rationale**: Comparing an observation to a claim rarely constitutes a
     meaningful contradiction. `insight_type` (RFC-072 Section 2a) enables this
     filtering.

3. **Graceful degradation without enricher**
   - **Decision**: When `nli_contradiction` enricher has not run, the analysis layer
     returns empty results (no errors, no degraded UI).
   - **Rationale**: The enricher is opt-in (ML tier). Users who have not enabled it
     should not see broken UI or error states.

---

## Testing Strategy

**Unit tests:**

- Position change detection on synthetic Insight pairs (golden set).
- Contradiction detection on synthetic cross-person pairs (golden set).
- Summarisation output format validation (schema, length, citation requirements).
- Graceful degradation when enricher output is missing.

**Integration tests:**

- End-to-end: enricher output on disk -> API request -> populated response fields.
- Latency benchmarks on eval corpus.

**E2E tests (Playwright):**

- Position Tracker: stance summary banner appears when populated.
- Position Tracker: position-change markers appear on correct episode cards.
- Guest Brief: Potential Challenges section appears with challenge cards.
- Both: Phase 6 UI hidden when fields are empty.

---

## Rollout

**Phase 1 -- Eval infrastructure:**

Create golden eval sets in `tests/fixtures/cil_phase6_golden/`. Define metrics.
Validate that RFC-073 `nli_contradiction` enricher produces reasonable candidates on
the eval corpus.

**Phase 2 -- Position change detection:**

Implement detection logic (mechanism TBD from eval results). Populate
`position_change_detected` and `change_description` in the Position Tracker API
response. Ship viewer UI (UXS-009 position-change markers).

**Phase 3 -- Contradiction detection refinement:**

Implement query-time LLM refinement of enricher candidates. Populate
`potential_challenges` with full summaries and ranking. Ship viewer UI (UXS-010
Potential Challenges section).

**Phase 4 -- Stance summarisation:**

Implement `stance_summary` generation for position arcs. Ship viewer UI (UXS-009
stance summary banner).

---

## Relationship to Other RFCs

This RFC (RFC-075) is the analysis layer that sits on top of:

1. **RFC-072** (Phases 1--5): Provides the data foundation -- CIL identity, bridge
   artifact, query patterns, and the field contracts (`position_change_detected`,
   `stance_summary`, `potential_challenges`) that this RFC populates.
2. **RFC-073** (Phase 3 `nli_contradiction`): Provides pre-computed contradiction
   candidate pairs. This RFC consumes them at query time.

**Key distinction:**

- **RFC-072**: Collects and joins data across layers (the plumbing).
- **RFC-073**: Enriches data with derived computations (batch processing).
- **RFC-075**: Analyses collected data to produce intelligence (query-time reasoning).

Together, these RFCs provide the complete path from raw podcast transcripts to
automated position tracking and contradiction detection.

---

## Benefits

1. **Automated position shift detection**: The "wow" moment -- the system flags the
   pivotal episode where a person changed their mind, without manual reading.
2. **Interview prep intelligence**: Cross-person contradictions surfaced
   automatically, replacing manual cross-referencing of past episodes.
3. **Dual-layer efficiency**: Batch pre-computation keeps query-time latency low
   while delivering high-quality analysis.
4. **Graceful progression**: PRDs 028/029 deliver value without this RFC (manual
   arc reading); this RFC adds the intelligence layer.
5. **Eval-driven quality**: Golden eval sets ensure detection accuracy is measured
   and improvable.

---

## Open Questions

1. **Mechanism selection**: LLM-as-judge vs NLI vs hybrid for position change
   detection. Depends on eval results with real corpus data.
2. **Contradiction threshold**: What `contradiction_score` threshold from the
   enricher should gate inclusion in `potential_challenges`? Needs tuning via eval.
3. **Summary length**: Should `stance_summary` be 1 sentence or up to 3? May
   depend on arc complexity.
4. **Provider requirements**: Which LLM providers support the analysis pass?
   Likely the same set as summarisation (OpenAI, Anthropic, Ollama, Gemini) but
   prompt complexity may vary.

---

## References

- `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
- `docs/rfc/RFC-073-enrichment-layer-architecture.md`
- `docs/prd/PRD-028-position-tracker.md` -- Phase 6 Analysis Contract section
- `docs/prd/PRD-029-guest-intelligence-brief.md` -- Phase 6 Analysis Contract section
- `docs/uxs/UXS-009-position-tracker.md` -- Phase 6 UI (stance banner, change markers)
- `docs/uxs/UXS-010-guest-intelligence-brief.md` -- Phase 6 UI (Potential Challenges)
- `tests/fixtures/cil_phase6_golden/` -- golden eval set placeholder
