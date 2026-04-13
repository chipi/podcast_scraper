# PRD-029: Guest Intelligence Brief

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: TBD
- **Related RFCs**:
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` -- CIL identity
    layer, bridge artifact, Flagship 2 definition, and query Pattern B
    (`guest_brief`)
  - `docs/rfc/RFC-062-gi-kg-viewer-v2.md` -- viewer SPA shell and tab model
  - `docs/rfc/RFC-069-graph-exploration-toolkit.md` -- graph chrome (Cytoscape)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` -- GIL Insight and Quote nodes
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` -- KG Episode and Entity nodes
  - `docs/rfc/RFC-061-semantic-corpus-search.md` -- FAISS search and chunk-to-Insight
    lift (Phase 5)
  - `docs/rfc/RFC-073-enrichment-layer-architecture.md` -- `nli_contradiction`
    enricher pre-computes contradiction candidate pairs (base for
    `potential_challenges`); `grounding_rate` enricher provides corpus grounding
    statistics
  - `docs/rfc/RFC-075-analysis-layer-position-change-contradiction-detection.md` --
    analysis layer; implements query-time refinement of `potential_challenges`
    (human-readable summaries, ranking)
- **Related PRDs**:
  - `docs/prd/PRD-017-grounded-insight-layer.md` -- GIL artifact foundation
  - `docs/prd/PRD-019-knowledge-graph-layer.md` -- KG artifact foundation
  - `docs/prd/PRD-026-topic-entity-view.md` -- topic-first navigation; cross-linked
    from each topic group in the brief
  - `docs/prd/PRD-027-enriched-search.md` -- enriched search entry point; speaker
    names in enriched sources can open Guest Brief via Person Landing
  - `docs/prd/PRD-028-position-tracker.md` -- companion flagship; each topic group
    in the brief links to the Position Tracker for that person + topic
- **Related UX specs**:
  - `docs/uxs/UXS-010-guest-intelligence-brief.md` -- visual contract for Guest Brief
    panel layout, sections, degradation states
  - `docs/uxs/UXS-001-gi-kg-viewer.md` -- shared design system (tokens, typography,
    states)
- **Related Architecture**:
  - `docs/architecture/gi/ontology.md`
  - `docs/architecture/kg/ontology.md`

---

## Summary

The **Guest Intelligence Brief** answers: "Before I interview person X, what should I
know about their publicly stated positions, their best quotes, and where they might
get challenged?"

It is a dedicated viewer panel where `person:{slug}` is the subject. The user
navigates to a person and sees a structured dossier: known positions grouped by topic
(with the strongest Insight per topic highlighted), a ranked list of best quotes with
timestamps, and corpus-wide summary statistics. The brief replaces the manual process
of skimming past episodes or searching -- it surfaces cross-episode patterns and
positions that a manual scan would miss.

The full product vision includes automated **contradiction detection** (Phase 6): the
system identifies topics where this person's stated position conflicts with another
guest's position, and surfaces these as "potential challenges" for interview
preparation. Until Phase 6 ships, the challenges section is empty.

No LLM and no database are required for the base brief. All data comes from the CIL
bridge artifact, GIL Insights and Quotes, and KG episode metadata. Phase 6
contradiction detection requires an LLM or NLI model for cross-person stance
comparison.

---

## Background

RFC-072 defines the Guest Intelligence Brief as Flagship 2 of the Canonical Identity
Layer. The CIL bridge makes corpus-wide person queries possible by scanning small
`bridge.json` files. The query pattern (`guest_brief`) is implemented in
`cil_queries.py` and exposed via `GET /api/persons/{person_id}/brief` (GitHub #527).

Today the API returns Insights grouped by topic and a flat list of quotes, but there
is no viewer surface. The response also lacks several fields needed for a complete
brief: display names, strongest-Insight ranking, best-quote ranking, and the Phase 6
`potential_challenges` array. This PRD adds the viewer panel, tightens the API
response, defines ranking logic, and specifies acceptance criteria for the complete
product experience.

PRD-028 (Position Tracker) is the companion flagship. Clicking a topic group in the
Guest Brief navigates to the Position Tracker for that person + topic combination,
enabling a drill-down from "what topics does this person discuss?" to "how has their
position on this specific topic evolved?"

---

## Goals

1. Make `person:{slug}` a navigable concept in the viewer -- the user can see a
   comprehensive dossier of a person's corpus presence.
2. Present known positions grouped by topic, with the strongest Insight per topic
   highlighted.
3. Surface the best quotes ranked by grounding quality and specificity.
4. Provide corpus-wide summary statistics: total topics, Insights, quotes, and
   grounding rate.
5. Enable multiple entry points: graph Person node click, search result speaker click,
   and a dedicated person browse.
6. Cross-link to PRD-028 Position Tracker (per-topic drill-down) and PRD-026 Topic
   Entity View (topic-first navigation).
7. Define the analysis contract: `potential_challenges` array that
   `docs/rfc/RFC-075-analysis-layer-position-change-contradiction-detection.md`
   must implement.
8. Work entirely from core artifacts (bridge, GIL, KG) for the base brief -- no LLM,
   no database, no new extraction.

---

## Non-Goals

- **Not** a conversational interface -- the brief is a static, structured view, not
  a chatbot.
- **Not** audio playback -- timestamps are displayed as text (e.g. "3:45 -- 3:51"),
  not as playable clips.
- **Not** a cross-corpus view -- single corpus only.
- **Not** a replacement for the graph view -- the Cytoscape Person node and its edges
  remain. The Guest Brief is a complementary surface.
- **Not** a new top-level tab -- it opens in the right rail panel or a dedicated route,
  depending on entry point.
- **Not** position-change detection within a single person's arc -- that is PRD-028
  (Position Tracker) Phase 6 scope.

---

## Personas

- **User**: A single person operating the podcast scraper tool. They prepare for
  interviews, research a guest's public positions, or explore what a specific person
  has contributed to the corpus.

---

## User Stories

- _As a user, I can click a person in the graph and see a structured brief of their
  known positions, best quotes, and corpus presence so that I can prepare for an
  interview without manually searching episodes._
- _As a user, I can see which topics a person has discussed and what their strongest
  position on each topic is so that I can quickly identify their key stances._
- _As a user, I can see a ranked list of the person's best quotes with timestamps so
  that I can find the most impactful moments across all their appearances._
- _As a user, I can click a topic group to drill into the Position Tracker (PRD-028)
  so that I can see how the person's position on that topic evolved over time._
- _As a user, I can see potential challenges (Phase 6) where another guest contradicts
  this person so that I can prepare tough questions for the interview._

---

## Functional Requirements

### FR1: API Layer

The Guest Brief API builds on the existing CIL endpoint. Requirements marked
"existing" are already implemented; requirements marked "new" require implementation.

- **FR1.1** (existing): `GET /api/persons/{person_id}/brief` returns a
  `CilGuestBriefResponse` with Insights grouped by topic and a flat list of quotes.
- **FR1.2** (new): The response must include `person_display_name` resolved from the
  bridge `identities` array. Each topic key in the `topics` dict must be accompanied
  by a `topic_display_names: dict[str, str]` mapping topic IDs to display names.
- **FR1.3** (new): Each topic group must include a `strongest_insight` field: the
  single Insight within that topic group that best represents the person's position.
  Ranking criteria (applied in order):
  1. `grounded: true` preferred over `grounded: false`.
  2. `insight_type == "claim"` preferred, then `"recommendation"`, then others.
  3. Highest `confidence` score.
  4. Highest `position_hint` (later in episode = more likely to be a settled
     position).
  5. Most recent `publish_date` (latest episode wins ties).
- **FR1.4** (new): The response must include a `best_quotes` array: the top N quotes
  (default N=10) across all topics, ranked by:
  1. Quote is from a grounded Insight (`grounded: true`).
  2. Quote length (longer quotes tend to be more substantive).
  3. Most recent `publish_date`.
  Each quote entry includes: `text`, `topic_id`, `topic_display_name`, `episode_id`,
  `episode_title`, `publish_date`, `timestamp_start_ms`, `timestamp_end_ms`.
- **FR1.5** (new): The response must include a `topic_summary` object:
  - `total_topics: int` -- number of distinct topics this person discusses.
  - `total_insights: int` -- total Insight count across all topics.
  - `total_quotes: int` -- total quote count.
  - `grounding_rate: float` -- fraction of Insights that are `grounded: true`
    (0.0 -- 1.0).
- **FR1.6** (new): The response must include `appearances: int` (total episodes where
  this person appears in the corpus via bridge scan) and `date_range` (`earliest` and
  `latest` publish dates).
- **FR1.7** (new): The response must include a
  `potential_challenges: list[ChallengeEntry]` array, defaulting to an empty list.
  Phase 6 analysis will populate this with cross-person contradictions. Each
  `ChallengeEntry` contains:
  - `topic: {id, display_name}` -- the topic where positions conflict.
  - `this_guest_position: string` -- summary of this person's stance.
  - `conflicting_guest: {id, display_name}` -- the other person.
  - `conflicting_position: string` -- summary of the other person's stance.
  - `episodes: list[string]` -- episode IDs where the conflict is evidenced.
  - `derived: true` -- trust marker indicating this was generated by analysis.
- **FR1.8** (new): The response must include `person_id` (already present) and
  `path` (already present) for consistency with other CIL responses.

### FR2: Viewer Panel -- Guest Brief

- **FR2.1**: This PRD **owns the shared Person Landing component** -- the
  entry-point surface that hosts both Guest Brief and Position Tracker (PRD-028).
  The Person Landing opens from three entry points:
  - Clicking a Person node in the Cytoscape graph (right rail panel).
  - Clicking a speaker name in a search result card (including lifted results).
  - A dedicated person browse (accessible from the viewer navigation).

  The Person Landing defaults to the Guest Brief tab. A tab or toggle switches to
  the Position Tracker (PRD-028). PRD-028 references this component; it does not
  define its own landing.
- **FR2.2**: Person header section shows:
  - Person display name (`text-lg font-semibold`).
  - Canonical `person:{slug}` ID (`muted`, `text-xs`, monospace).
  - Appearance count: "N episodes" badge.
  - Date range of first and last appearance.
  - Grounding rate badge: percentage of grounded Insights. Uses intent tokens:
    `success` for >= 80%, `muted` for 50--79%, `warning` for < 50%.
- **FR2.3**: Topic summary stats bar:
  - A horizontal bar showing: total topics, total Insights, total quotes, grounding
    rate.
  - Compact layout: four metric chips in a row (`surface` background, `text-sm`).
- **FR2.4**: Known Positions section:
  - Topics are listed as collapsible groups, sorted by Insight count descending
    (most-discussed topic first).
  - Each topic group header shows: topic display name, Insight count badge, and the
    `strongest_insight` text preview (first ~100 characters).
  - Expanding a topic group reveals all Insights for that topic, each with:
    `insight_type` badge, `position_hint` indicator, confidence, grounding status,
    episode title, and publish date.
  - A "Track positions" link on each topic group navigates to the Position Tracker
    (PRD-028) for this person + topic.
  - A "View topic" link navigates to the Topic Entity View (PRD-026) for that topic.
- **FR2.5**: Best Quotes section:
  - A ranked list of the top 10 quotes (from FR1.4).
  - Each quote card shows:
    - Verbatim quote text in blockquote style.
    - Topic tag (display name, clickable to Topic Entity View).
    - Episode title and publish date.
    - Timestamp display as text (e.g. "3:45 -- 3:51").
  - A "Show more quotes" control loads additional quotes if available.
- **FR2.6**: Cross-navigation:
  - "View in graph" button navigates to the Graph tab and focuses on this person's
    node.
  - "Track positions" on each topic group navigates to Position Tracker (PRD-028).
  - "View topic" on each topic group navigates to Topic Entity View (PRD-026).
- **FR2.7**: Phase 6 UI -- Potential Challenges section:
  - Visible only when `potential_challenges` is non-empty.
  - Each challenge card shows:
    - Topic display name.
    - This person's position summary.
    - Conflicting guest: display name (clickable to their Guest Brief) and their
      position summary.
    - Episode references.
    - A `derived: true` badge and provider attribution (`muted`, `text-xs`).
  - A section header: "Potential Challenges" with a `warning` token icon.
  - Until Phase 6 ships, this section is hidden entirely (not shown as an empty
    placeholder).

### FR3: Empty and Degraded States

These are acceptance criteria -- each must have defined UI behavior and test coverage.

- **FR3.1**: Person has 0 Insights (appears in KG but no GIL attribution):
  - The person header (FR2.2) still shows with metadata from KG (name, episode count
    from MENTIONS edges).
  - The topic summary stats bar shows zeros.
  - The Known Positions section shows: "No grounded insights found for this person.
    Insights appear when the pipeline runs with GIL extraction enabled."
  - The Best Quotes section is hidden.
- **FR3.2**: Person has Insights but 0 grounded Quotes:
  - Known Positions section displays Insights with "ungrounded" badges.
  - Best Quotes section shows: "No grounded quotes available. Grounding improves
    with diarization and GIL extraction quality."
  - The grounding rate in the header shows 0% with `warning` token.
- **FR3.3**: A topic in the brief has 0 Insights (edge case -- topic appears in
  bridge but no ABOUT edges in GIL):
  - The topic group appears in Known Positions with "0 insights" badge.
  - Expanding it shows: "No insights on this topic. The topic was mentioned but not
    deeply analysed."
- **FR3.4**: Lift fails (char offset mismatch on search entry point):
  - The search result card shows the raw transcript chunk without a speaker link.
  - No Guest Brief navigation is offered from that card.
- **FR3.5**: No `bridge.json` for some or all episodes:
  - Episodes without a bridge file are silently excluded from the brief.
  - If all episodes lack bridge files, the brief area shows: "No cross-layer data
    available. Run the pipeline with bridge generation enabled."
  - The person header still shows KG-derived metadata if available.

---

## Data Flow

```text
Entry point (graph / search / browse)
    |
    v
GET /api/persons/{id}/brief
    |
    v
CIL query: scan bridge.json files across corpus
    |
    v
Filter: episodes where person is in bridge GI identities
    |
    v
Load gi.json per episode: ABOUT + SUPPORTED_BY + SPOKEN_BY edges
    |
    v
Group Insights by topic; rank strongest per topic
    |
    v
Collect and rank best quotes across all topics
    |
    v
Compute topic_summary stats
    |
    v
Attach potential_challenges (empty until Phase 6)
    |
    v
Viewer renders brief: header, stats, positions, quotes, challenges
```

---

## API Response Shape

The enriched `CilGuestBriefResponse` (FR1.2 -- FR1.8):

```json
{
  "path": "/path/to/corpus",
  "person_id": "person:sam-altman",
  "person_display_name": "Sam Altman",
  "appearances": 8,
  "date_range": {
    "earliest": "2023-01-10",
    "latest": "2026-03-15"
  },
  "topic_summary": {
    "total_topics": 12,
    "total_insights": 34,
    "total_quotes": 89,
    "grounding_rate": 0.85
  },
  "topic_display_names": {
    "topic:ai-regulation": "AI Regulation",
    "topic:open-source": "Open Source"
  },
  "topics": {
    "topic:ai-regulation": {
      "insight_count": 5,
      "strongest_insight": {
        "id": "insight:x1y2z3",
        "text": "AI regulation will significantly lag behind the pace of innovation",
        "insight_type": "claim",
        "position_hint": 0.78,
        "confidence": 0.92,
        "grounded": true,
        "episode_id": "episode:abc123",
        "publish_date": "2024-03-10T00:00:00Z"
      },
      "insights": [
        {
          "episode_id": "episode:abc123",
          "insight": {},
          "insight_type": "claim",
          "position_hint": 0.78
        }
      ]
    }
  },
  "best_quotes": [
    {
      "text": "We need guardrails, not bans. The difference matters.",
      "topic_id": "topic:ai-regulation",
      "topic_display_name": "AI Regulation",
      "episode_id": "episode:abc123",
      "episode_title": "AI Regulation Deep Dive",
      "publish_date": "2024-03-10T00:00:00Z",
      "timestamp_start_ms": 225000,
      "timestamp_end_ms": 231000
    }
  ],
  "potential_challenges": []
}
```

When Phase 6 analysis is available, `potential_challenges` is populated:

```json
{
  "potential_challenges": [
    {
      "topic": {
        "id": "topic:ai-regulation",
        "display_name": "AI Regulation"
      },
      "this_guest_position": "AI regulation will lag innovation; self-regulation preferred",
      "conflicting_guest": {
        "id": "person:timnit-gebru",
        "display_name": "Timnit Gebru"
      },
      "conflicting_position": "Voluntary safety commitments are insufficient; binding regulation is essential",
      "episodes": ["episode:ghi789"],
      "derived": true
    }
  ]
}
```

---

## Success Criteria

1. A user can click a Person node in the graph and see the full Guest Brief in the
   right rail without leaving the viewer session.
2. Known Positions are grouped by topic with the strongest Insight highlighted per
   topic.
3. Best Quotes are ranked and show timestamps in human-readable format.
4. The topic summary stats bar correctly reflects corpus-wide counts and grounding
   rate.
5. Cross-navigation to Position Tracker (PRD-028) and Topic Entity View (PRD-026)
   works from the Guest Brief panel.
6. All five empty/degraded states (FR3.1 -- FR3.5) render correctly with honest
   messaging and no broken UI.
7. The `potential_challenges` array is present in the API response (defaulting to
   empty) and the viewer hides the Potential Challenges section when it is empty.
8. When Phase 6 analysis populates `potential_challenges`, the viewer renders
   challenge cards with conflicting guest links and `derived: true` badges without
   code changes to the viewer.
9. Playwright E2E coverage for: populated brief, empty person, empty quotes, and
   Phase 6 placeholder behavior.

---

## Dependencies

- **RFC-072** (Phases 1--4): CIL identity, bridge artifact, `guest_brief` query
  pattern, and CIL API endpoints.
- **RFC-073** (`nli_contradiction` enricher): pre-computes contradiction candidate
  pairs (base layer for `potential_challenges`). Also provides `grounding_rate`
  corpus enricher used in topic summary stats.
- **PRD-026** (Topic Entity View): cross-linked from each topic group.
- **PRD-027** (Enriched Search): entry point via speaker names in enriched sources.
- **PRD-028** (Position Tracker): companion flagship; "Track positions" navigates
  there. References the Person Landing component owned by this PRD.
- **RFC-075** (Analysis Layer): implements query-time refinement of contradiction
  candidates from RFC-073 enricher. Adds human-readable position summaries and
  ranking. This PRD defines the contract; RFC-075 implements the enhanced layer.

---

## Constraints and Assumptions

**Constraints:**

- Must run on Apple M4 Pro with 48 GB RAM (local-first tool).
- Brief assembly must complete in < 5 seconds for a corpus of 200 episodes.
- No database -- all data read from filesystem artifacts.

**Assumptions:**

- The user has run the pipeline with GIL extraction and bridge generation enabled.
- `bridge.json` files exist for the episodes the user wants to explore.
- Display names are available in the bridge `identities` array.

---

## Design Considerations

### Strongest Insight Ranking

- **Option A**: Server-side ranking (compute `strongest_insight` in the API).
  - Pros: Consistent ranking across clients; ranking logic is testable in Python.
  - Cons: Ranking criteria are opaque to the viewer.
- **Option B**: Client-side ranking (viewer sorts Insights per topic).
  - Pros: User can change ranking criteria in the UI.
  - Cons: Duplicates logic; ranking may differ between API consumers.
- **Decision**: Server-side ranking (Option A). The API computes
  `strongest_insight` using the criteria in FR1.3. The viewer displays it as-is.
  A future iteration may add a client-side re-sort option.

### Best Quotes Ranking

- The ranking in FR1.4 prioritises grounding, then length, then recency. This is a
  heuristic -- there is no learned quality score. The ranking may be refined in a
  future iteration with a quote quality model or user feedback signal.

### Known Positions vs All Insights

- The Known Positions section shows topics with their `strongest_insight` as a
  summary. Expanding a topic group reveals all Insights. This two-level structure
  keeps the brief scannable while allowing deep exploration.

### Phase 6 Contradiction Detection -- Dual-Layer Strategy

Contradiction detection uses a **dual-layer strategy** that separates batch
pre-computation from query-time refinement:

**Base layer (RFC-073 `nli_contradiction` enricher):**

- A batch corpus enricher (RFC-073 Phase 3 ML tier) that pre-computes contradiction
  candidate pairs to disk during the enrichment pipeline run.
- Each candidate pair records: `topic_id`, `person_a_id`, `person_b_id`,
  `insight_a_id`, `insight_b_id`, and a `contradiction_score` (0.0 -- 1.0).
- This enricher runs without an LLM -- it uses NLI or embedding-based stance
  comparison.
- The `potential_challenges` array (FR1.7) can be **partially populated** from
  enricher output alone: the system reads pre-computed candidate pairs and surfaces
  them with raw Insight text (no `this_guest_position` / `conflicting_position`
  summaries).

**Enhanced layer (RFC-072 Phase 6 query-time analysis):**

- A request-time analysis pass that refines the enricher candidates.
- Adds `this_guest_position` and `conflicting_position` human-readable summaries
  (LLM-generated).
- Ranks challenges by relevance and filters out weak contradictions.
- Carries `derived: true` and LLM provider attribution.

This means `potential_challenges` has **three states**:
1. Empty array (no enricher has run).
2. Partially populated (enricher ran but no Phase 6 LLM refinement -- raw candidate
   pairs with Insight text, no summaries).
3. Fully populated (enricher + Phase 6 analysis -- candidate pairs with human-readable
   summaries and ranking).

---

## Phase 6 Analysis Contract

This section defines what
[RFC-075](../rfc/RFC-075-analysis-layer-position-change-contradiction-detection.md)
must implement to fulfil the Guest Brief's full vision.

### Contradiction Detection (Potential Challenges)

- **Input**: For each topic this person discusses, the set of Insights from other
  persons on the same topic (retrieved via `topic_person_ids` +
  per-person `guest_brief`).
- **Output**: A list of `ChallengeEntry` objects (FR1.7) where the other person's
  position meaningfully contradicts this person's position.
- **Mechanism options** (to be decided in RFC-075):
  - LLM-as-judge: prompt an LLM with both persons' Insights on the topic, ask
    whether they contradict.
  - NLI-based: entailment/contradiction scoring between Insight pairs from different
    persons.
  - Embedding + stance: compute stance embeddings and flag when positions are
    semantically opposed.
- **Pruning**: At corpus scale, pairwise comparison is expensive. RFC-075 must
  define a pruning strategy (e.g. only compare persons who share >= 2 topics, or
  only compare `claim`-type Insights).
- **Eval requirement**: A golden eval set of Insight pairs with human-labelled
  "agree" / "disagree" / "unrelated" verdicts. Placeholder directory:
  `tests/fixtures/cil_phase6_golden/`.

### Position Summaries for Challenges

- Each `ChallengeEntry` includes `this_guest_position` and `conflicting_position` --
  short summaries of each person's stance on the topic.
- These summaries are generated by the analysis layer (LLM or template-based) and
  carry `derived: true`.
- RFC-075 must define the summarisation mechanism and trust contract.

---

## Known Limitations

- **Single-episode corpus**: The brief shows one episode's worth of data. Topic
  grouping and cross-episode patterns are not possible. The viewer does not show a
  special state -- it displays whatever data exists.
- **Analysis layer not yet available**: `potential_challenges` is always an empty
  array until RFC-075 is implemented. The viewer hides the Potential Challenges
  section when the array is empty.

---

## Open Questions

1. **Best quotes count**: Should the default be 10 quotes or configurable via a query
   parameter? Recommend: default 10, with a `quote_limit` query parameter for API
   consumers.
2. **Topic group ordering**: Should topics be sorted by Insight count (most discussed
   first) or by recency (most recently discussed first)? Recommend: Insight count
   descending as default, with a client-side toggle for recency.
3. **Appearance count source**: Same question as PRD-028 -- bridge scan (GIL
   attribution) vs KG (mentions). Recommend: bridge scan for consistency with the
   brief's GIL-centric data.

---

## Related Work

- Issue #527: CIL query API implementation
- PRD-026: Topic Entity View
- PRD-028: Position Tracker
- RFC-072: Canonical Identity Layer
- UXS-010: `docs/uxs/UXS-010-guest-intelligence-brief.md` -- visual contract

---

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-075 technical design reviewed for contradiction detection
- [ ] API enrichments implemented (FR1.2 -- FR1.8)
- [ ] Strongest Insight ranking implemented and tested (FR1.3)
- [ ] Best Quotes ranking implemented and tested (FR1.4)
- [ ] Viewer panel implemented with all entry points (FR2.1 -- FR2.7)
- [ ] Empty/degraded states implemented and tested (FR3.1 -- FR3.5)
- [ ] UXS-010 tokens implemented in viewer styles
- [ ] Playwright E2E coverage for populated, empty, and Phase 6 states
- [ ] Documentation updated (SERVER_GUIDE, E2E_SURFACE_MAP)
- [ ] Integration with PRD-028 (Position Tracker) verified
- [ ] Integration with PRD-026 (Topic Entity View) verified
