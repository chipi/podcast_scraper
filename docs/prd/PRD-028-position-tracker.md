# PRD-028: Position Tracker

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: TBD
- **Related RFCs**:
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` -- CIL identity
    layer, bridge artifact, Flagship 1 definition, and query Pattern A
    (`position_arc`)
  - `docs/rfc/RFC-062-gi-kg-viewer-v2.md` -- viewer SPA shell and tab model
  - `docs/rfc/RFC-069-graph-exploration-toolkit.md` -- graph chrome (Cytoscape)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` -- GIL Insight and Quote nodes
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` -- KG Episode and Entity nodes
  - `docs/rfc/RFC-061-semantic-corpus-search.md` -- FAISS search and chunk-to-Insight
    lift (Phase 5)
  - `docs/rfc/RFC-075-analysis-layer-position-change-contradiction-detection.md` --
    analysis layer; implements `position_change_detected` and `stance_summary`
- **Related PRDs**:
  - `docs/prd/PRD-017-grounded-insight-layer.md` -- GIL artifact foundation
  - `docs/prd/PRD-019-knowledge-graph-layer.md` -- KG artifact foundation
  - `docs/prd/PRD-026-topic-entity-view.md` -- topic-first navigation; cross-linked
    from Position Tracker topic selector
  - `docs/prd/PRD-027-enriched-search.md` -- enriched search entry point; speaker
    names in enriched sources can open Position Tracker via Person Landing
  - `docs/prd/PRD-029-guest-intelligence-brief.md` -- companion flagship; Guest Brief
    links to Position Tracker for per-topic drill-down; **owns the shared Person
    Landing component** that hosts both Guest Brief and Position Tracker
- **Related UX specs**:
  - `docs/uxs/UXS-009-position-tracker.md` -- visual contract for Position Tracker
    panel layout, timeline, quote cards, degradation states
  - `docs/uxs/UXS-001-gi-kg-viewer.md` -- shared design system (tokens, typography,
    states)
- **Related Architecture**:
  - `docs/architecture/gi/ontology.md`
  - `docs/architecture/kg/ontology.md`

---

## Summary

The **Position Tracker** answers: "How has person X's thinking on topic Y evolved
across episodes -- and when did it change?"

It is a dedicated viewer panel where `person:{slug}` + `topic:{slug}` are the
subject. The user navigates to a person, selects a topic, and sees a chronological
arc of grounded Insights with verbatim quotes and timestamps -- one entry per
episode, ordered by publish date. The difference between a quote search and a
Position Tracker is *narrative structure over time*: not "here are 5 things Satya
Nadella said about AI" but "here is how his thinking shifted from 2023 to 2025, and
here is the moment it changed, grounded in his own words."

The full product vision includes automated **position-change detection** (Phase 6):
the system flags episodes where a person's stated position on a topic contradicts or
evolves from their earlier position, and produces a human-readable stance summary.
Until Phase 6 ships, the arc is assembled but change detection fields are empty.

No LLM and no database are required for the base arc. All data comes from the CIL
bridge artifact, GIL Insights and Quotes, and KG episode metadata. Phase 6 analysis
requires an LLM or NLI model for stance comparison.

---

## Background

RFC-072 defines the Position Tracker as Flagship 1 of the Canonical Identity Layer.
The CIL bridge makes cross-episode person + topic queries possible by scanning small
`bridge.json` files instead of loading every `gi.json` in the corpus. The query
pattern (`position_arc`) is implemented in `cil_queries.py` and exposed via
`GET /api/persons/{person_id}/positions?topic={topic_id}` (GitHub #527).

Today the API returns structured data but there is no viewer surface. The user must
call the API directly or use the CLI. This PRD adds the viewer panel, tightens the
API response to include display names and Phase 6 placeholder fields, and defines
acceptance criteria for the complete product experience.

PRD-029 (Guest Intelligence Brief) is the companion flagship. The Guest Brief shows
all topics a person discusses; clicking a topic group in the brief navigates to the
Position Tracker for that person + topic combination.

---

## Goals

1. Make `person:{slug}` + `topic:{slug}` a navigable concept in the viewer -- the
   user can see how a person's position on a topic has evolved across episodes.
2. Present a chronological arc of grounded Insights with verbatim quotes, timestamps,
   and episode attribution.
3. Enable filtering by `insight_type` so the user can focus on claims, recommendations,
   or all types.
4. Provide multiple entry points: graph Person node click, search result speaker click,
   and a dedicated person browse.
5. Cross-link to PRD-026 Topic Entity View from the topic selector.
6. Define the analysis contract: `position_change_detected` and `stance_summary`
   fields that `docs/rfc/RFC-075-analysis-layer-position-change-contradiction-detection.md`
   must implement.
7. Work entirely from core artifacts (bridge, GIL, KG) for the base arc -- no LLM, no
   database, no new extraction.

---

## Non-Goals

- **Not** automated contradiction detection or stance comparison across persons --
  that is the Guest Brief's `potential_challenges` (PRD-029 Phase 6), not Position
  Tracker.
- **Not** a cross-corpus view -- single corpus only.
- **Not** audio playback -- timestamps are displayed as text (e.g. "14:23 -- 14:35"),
  not as playable clips.
- **Not** a replacement for the graph view -- the Cytoscape Person node and its edges
  remain. Position Tracker is a complementary surface.
- **Not** a new top-level tab -- it opens in the right rail panel or a dedicated route,
  depending on entry point.

---

## Personas

- **User**: A single person operating the podcast scraper tool. They explore their
  corpus to understand how guests' positions evolve, prepare for interviews, or
  research public figures' stated views over time.

---

## User Stories

- _As a user, I can click a person in the graph and select a topic so that I see a
  chronological arc of their stated positions on that topic across all episodes._
- _As a user, I can see verbatim quotes with timestamps for each position so that I
  can verify the attribution and find the exact moment in the episode._
- _As a user, I can filter the arc to show only claims (or recommendations, or all
  types) so that I can focus on the signal that matters for my research._
- _As a user, I can click a topic in the selector to navigate to the Topic Entity View
  (PRD-026) so that I can see the full corpus picture for that concept._
- _As a user, I can see when a person's position changed (Phase 6) so that I can
  identify the pivotal episode without reading every Insight manually._

---

## Functional Requirements

### FR1: API Layer

The Position Tracker API builds on the existing CIL endpoints. Requirements marked
"existing" are already implemented; requirements marked "new" require implementation.

- **FR1.1** (existing): `GET /api/persons/{person_id}/positions?topic={topic_id}`
  returns a `CilPositionArcResponse` with one `CilArcEpisodeBlock` per episode where
  the person discusses the topic, ordered by `publish_date`.
- **FR1.2** (existing): `GET /api/persons/{person_id}/topics` returns a list of all
  topic IDs this person discusses (for the topic selector).
- **FR1.3** (new): The `CilPositionArcResponse` must include `person_display_name`
  and `topic_display_name` resolved from the bridge `identities` array. The viewer
  must not have to resolve IDs to display names client-side.
- **FR1.4** (new): Each `CilArcEpisodeBlock` must include `podcast_title` resolved
  from the KG Episode node, so the viewer can show which show the episode belongs to.
- **FR1.5** (new): The `CilPositionArcResponse` must include summary fields:
  `position_count` (total Insights across all episodes), `episode_count`, and
  `date_range` (`earliest` and `latest` publish dates).
- **FR1.6** (new): Each `CilArcEpisodeBlock` must include a
  `position_change_detected: bool` field, defaulting to `false`. Phase 6 analysis
  will populate this when it detects that the person's stance in this episode
  contradicts or evolves from their stance in a prior episode.
- **FR1.7** (new): The `CilPositionArcResponse` must include a
  `stance_summary: string | null` field, defaulting to `null`. Phase 6 analysis will
  populate this with a human-readable summary of how the person's position evolved
  (e.g. "Shifted from pro-innovation to pro-regulation between June 2023 and
  February 2025").
- **FR1.8** (new): The `CilIdListResponse` for `/persons/{id}/topics` must include
  `display_names: dict[str, str]` mapping each topic ID to its display name from the
  bridge, so the topic selector can show human-readable labels.
- **FR1.9** (existing): The `insight_types` query parameter on the positions endpoint
  filters Insights by type. Default is `claim` only; `all` or `*` disables the
  filter.

### FR2: Viewer Panel -- Position Tracker

- **FR2.1**: The Position Tracker is one view within the **shared Person Landing**
  component (owned by PRD-029). It opens from three entry points:
  - Clicking a Person node in the Cytoscape graph (right rail panel).
  - Clicking a speaker name in a search result card (including lifted results).
  - A dedicated person browse (accessible from the viewer navigation).

  All entry points navigate to the Person Landing, which defaults to the Guest Brief
  tab (PRD-029). The user toggles to Position Tracker from there. When the entry
  point is topic-specific (e.g. a topic link from PRD-026), the Position Tracker tab
  is preselected with that topic active.
- **FR2.2**: Person header section shows:
  - Person display name (`text-lg font-semibold`).
  - Canonical `person:{slug}` ID (`muted`, `text-xs`, monospace).
  - Appearance count (total episodes where this person appears in the corpus).
  - Date range of first and last appearance.
- **FR2.3**: Topic selector:
  - A searchable dropdown populated from `GET /api/persons/{id}/topics` with display
    names (FR1.8).
  - Selecting a topic loads the position arc from
    `GET /api/persons/{id}/positions?topic={id}`.
  - A "View topic" link next to the selected topic navigates to the PRD-026 Topic
    Entity View for that topic.
- **FR2.4**: Insight type filter:
  - A segmented control with options: "Claims" (default), "Recommendations", "All".
  - Changing the filter re-fetches the arc with the corresponding `insight_types`
    parameter.
- **FR2.5**: Timeline visualization:
  - Episodes are displayed as cards in a vertical timeline, ordered by `publish_date`
    (most recent first).
  - Each episode card shows: episode title, podcast title, publish date, and the
    number of matching Insights.
  - A connecting line between episode cards visualises the chronological progression.
- **FR2.6**: Insight cards within each episode:
  - Insight text.
  - `insight_type` badge (e.g. "claim", "recommendation", "observation") using intent
    tokens.
  - `position_hint` indicator: a small bar or number showing the Insight's relative
    position within the episode (0.0 = early, 1.0 = late).
  - Confidence score (if available).
  - Grounding status badge: "grounded" (green/`success`) or "ungrounded"
    (`warning`).
- **FR2.7**: Quote cards within each Insight:
  - Verbatim quote text in a blockquote style.
  - Timestamp display as text (e.g. "20:00 -- 20:15"), formatted from
    `timestamp_start_ms` and `timestamp_end_ms`.
  - Episode attribution (episode title, publish date).
- **FR2.8**: Cross-navigation:
  - "View in graph" button navigates to the Graph tab and focuses on this person's
    node.
  - "Open brief" button navigates to the Guest Brief (PRD-029) for this person.
- **FR2.9**: Phase 6 UI (placeholder until analysis layer ships):
  - Position-change markers: when `position_change_detected: true` on an episode
    block, a visual marker (e.g. a "Position shifted" badge with `warning` token)
    appears on that episode card in the timeline.
  - Stance summary banner: when `stance_summary` is non-null, a banner at the top of
    the arc displays the summary text. Uses `surface` background with `gi` domain
    token left border.
  - Until Phase 6 ships, these elements are hidden (not shown as empty placeholders).

### FR3: Empty and Degraded States

These are acceptance criteria -- each must have defined UI behavior and test coverage.

- **FR3.1**: Person has 0 Insights (appears in KG but no GIL attribution):
  - The person header (FR2.2) still shows with metadata from KG (name, episode count).
  - The topic selector is empty or shows "No topics found."
  - The arc area shows: "No grounded insights found for this person. Insights appear
    when the pipeline runs with GIL extraction enabled."
- **FR3.2**: Person has Insights but 0 grounded Quotes (all Insights are ungrounded):
  - Insight cards display with an "ungrounded" badge (`warning` token).
  - The quote section within each Insight is hidden (no empty quote placeholder).
  - A note below the Insights: "These insights are not grounded in verbatim quotes.
    Grounding improves with diarization and GIL extraction quality."
- **FR3.3**: Selected topic has 0 Insights:
  - The topic appears in the selector (it exists in the bridge).
  - Selecting it shows: "No insights on this topic for [person name]. Try selecting
    a different topic or viewing all types."
- **FR3.4**: Lift fails (char offset mismatch on search entry point):
  - The search result card shows the raw transcript chunk without a speaker link.
  - No Position Tracker navigation is offered from that card.
  - The `lift_stats` in the search response indicates the lift failure.
- **FR3.5**: No `bridge.json` for some episodes:
  - Episodes without a bridge file are silently excluded from the arc.
  - If all episodes lack bridge files, the arc area shows: "No cross-layer data
    available. Run the pipeline with bridge generation enabled."
  - The person header still shows KG-derived metadata if available.

---

## Data Flow

```text
Entry point (graph / search / browse)
    |
    v
GET /api/persons/{id}/topics  -->  topic selector populated
    |
    v  (user selects topic)
GET /api/persons/{id}/positions?topic={id}&insight_types=claim
    |
    v
CIL query: scan bridge.json files across corpus
    |
    v
Filter: episodes where person + topic both in bridge GI identities
    |
    v
Load gi.json per episode: ABOUT + SUPPORTED_BY + SPOKEN_BY edges
    |
    v
Assemble arc: Insights + Quotes, ordered by publish_date
    |
    v
Viewer renders timeline with Insight and Quote cards
```

---

## API Response Shape

The enriched `CilPositionArcResponse` (FR1.3 -- FR1.7):

```json
{
  "path": "/path/to/corpus",
  "person_id": "person:satya-nadella",
  "person_display_name": "Satya Nadella",
  "topic_id": "topic:ai-safety",
  "topic_display_name": "AI Safety",
  "position_count": 4,
  "episode_count": 2,
  "date_range": {
    "earliest": "2023-06-15",
    "latest": "2025-02-20"
  },
  "stance_summary": null,
  "episodes": [
    {
      "episode_id": "episode:abc123",
      "publish_date": "2023-06-15T00:00:00Z",
      "podcast_title": "Lex Fridman Podcast",
      "position_change_detected": false,
      "insights": [
        {
          "id": "insight:a1b2c3d4",
          "text": "AI safety is important but should not slow down innovation",
          "insight_type": "claim",
          "position_hint": 0.15,
          "grounded": true,
          "confidence": 0.88,
          "supporting_quotes": [
            {
              "text": "We need to move fast. Safety is a priority, but paralysis is not safety.",
              "timestamp_start_ms": 1200000,
              "timestamp_end_ms": 1215000
            }
          ]
        }
      ]
    },
    {
      "episode_id": "episode:def456",
      "publish_date": "2025-02-20T00:00:00Z",
      "podcast_title": "Hard Fork",
      "position_change_detected": false,
      "insights": [
        {
          "id": "insight:e5f6g7h8",
          "text": "AI safety requires mandatory external audits before deployment",
          "insight_type": "claim",
          "position_hint": 0.85,
          "grounded": true,
          "confidence": 0.91,
          "supporting_quotes": [
            {
              "text": "I've changed my mind. We need third-party audits. Full stop.",
              "timestamp_start_ms": 840000,
              "timestamp_end_ms": 852000
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Success Criteria

1. A user can click a Person node in the graph, select a topic, and see the full
   position arc in the right rail without leaving the viewer session.
2. Each Insight in the arc is accompanied by at least one verbatim quote with a
   human-readable timestamp.
3. The `insight_type` filter correctly narrows the arc to the selected types.
4. The topic selector shows human-readable display names, not raw IDs.
5. Cross-navigation to Topic Entity View (PRD-026) and Guest Brief (PRD-029) works
   from the Position Tracker panel.
6. All five empty/degraded states (FR3.1 -- FR3.5) render correctly with honest
   messaging and no broken UI.
7. The `position_change_detected` and `stance_summary` fields are present in the API
   response (defaulting to `false` / `null`) and the viewer hides Phase 6 UI when
   they are empty.
8. When Phase 6 analysis populates these fields, the viewer renders position-change
   markers and the stance summary banner without code changes to the viewer.
9. Playwright E2E coverage for: populated arc, empty person, empty topic, and Phase 6
   placeholder behavior.

---

## Dependencies

- **RFC-072** (Phases 1--4): CIL identity, bridge artifact, `position_arc` query
  pattern, and CIL API endpoints.
- **PRD-026** (Topic Entity View): cross-linked from the topic selector.
- **PRD-029** (Guest Brief): companion flagship; "Open brief" button navigates there.
- **RFC-075** (Analysis Layer): implements `position_change_detected` and
  `stance_summary` analysis. This PRD defines the contract; RFC-075 implements it.

---

## Constraints and Assumptions

**Constraints:**

- Must run on Apple M4 Pro with 48 GB RAM (local-first tool).
- Arc assembly must complete in < 3 seconds for a corpus of 200 episodes.
- No database -- all data read from filesystem artifacts.

**Assumptions:**

- The user has run the pipeline with GIL extraction and bridge generation enabled.
- `bridge.json` files exist for the episodes the user wants to explore.
- Display names are available in the bridge `identities` array.

---

## Design Considerations

### Timeline Layout

- **Option A**: Horizontal timeline with episode markers on a date axis.
  - Pros: Visually shows time gaps between episodes; familiar timeline metaphor.
  - Cons: Horizontal scrolling on narrow screens; sparse for few episodes.
- **Option B**: Vertical card list ordered by date.
  - Pros: Works well in a right rail panel; scales to many episodes; no horizontal
    scroll.
  - Cons: Less visually "timeline-like."
- **Decision**: Vertical card list (Option B) for the right rail entry point. A
  connecting line between cards provides the timeline metaphor. Full-width mode
  (from browse entry point) may use a horizontal timeline in a future iteration.

### Insight Ranking Within an Episode

- Insights within a single episode are ordered by `position_hint` (ascending), so
  the user reads them in the order they appeared in the conversation. This preserves
  the argumentative progression (setup before conclusion).

### Phase 6 Field Placement

- `position_change_detected` is per-episode (on `CilArcEpisodeBlock`) because change
  is relative to the *previous* episode in the arc.
- `stance_summary` is per-arc (on `CilPositionArcResponse`) because it summarises the
  entire evolution, not a single episode.

---

## Phase 6 Analysis Contract

This section defines what
[RFC-075](../rfc/RFC-075-analysis-layer-position-change-contradiction-detection.md)
must implement to fulfil the Position Tracker's full vision.

### Position Change Detection

- **Input**: The assembled position arc (list of episodes with Insights).
- **Output**: For each episode after the first, a boolean
  `position_change_detected` and an optional `change_description: string` explaining
  what changed.
- **Mechanism options** (to be decided in RFC-075):
  - LLM-as-judge: prompt an LLM with the previous and current episode's Insights,
    ask whether the position changed.
  - NLI-based: entailment/contradiction scoring between Insight pairs from
    consecutive episodes.
  - Embedding + stance: compute stance embeddings and flag when cosine distance
    exceeds a threshold.
- **Eval requirement**: A golden eval set of Insight pairs with human-labelled
  "changed" / "unchanged" / "nuanced" verdicts. Placeholder directory:
  `tests/fixtures/cil_phase6_golden/`.

### Stance Summary

- **Input**: The full arc with change detection results.
- **Output**: A 1--3 sentence human-readable summary of the position evolution.
- **Mechanism**: LLM pass over the arc's Insights, constrained to cite specific
  episodes and dates.
- **Trust**: The summary carries `derived: true` and the LLM provider is attributed
  in the response.

---

## Known Limitations

- **Single-episode corpus**: The arc shows one data point. No evolution is possible.
  The viewer does not show a special state for this -- it simply displays one episode
  card.
- **Analysis layer not yet available**: `position_change_detected` is always `false`
  and `stance_summary` is always `null` until RFC-075 is implemented. The viewer
  hides analysis UI elements when these fields are empty.

---

## Open Questions

1. **Appearance count source**: Should the person header's appearance count come from
   the bridge scan (episodes where the person has GIL attribution) or from KG
   (episodes where the person is mentioned)? The bridge count is more accurate for
   "episodes where this person spoke" but may undercount mentions.
2. **Cross-show arc**: When the corpus contains multiple podcasts, should the arc
   interleave episodes from different shows or group by show? Interleaving by date
   is the default; grouping by show is a future option.

---

## Related Work

- Issue #527: CIL query API implementation
- PRD-026: Topic Entity View
- PRD-029: Guest Intelligence Brief
- RFC-072: Canonical Identity Layer
- UXS-009: `docs/uxs/UXS-009-position-tracker.md` -- visual contract

---

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-075 technical design reviewed for change detection and stance summary
- [ ] API enrichments implemented (FR1.3 -- FR1.8)
- [ ] Viewer panel implemented with all entry points (FR2.1 -- FR2.9)
- [ ] Empty/degraded states implemented and tested (FR3.1 -- FR3.5)
- [ ] UXS-009 tokens implemented in viewer styles
- [ ] Playwright E2E coverage for populated, empty, and Phase 6 states
- [ ] Documentation updated (SERVER_GUIDE, E2E_SURFACE_MAP)
- [ ] Integration with PRD-026 (Topic Entity View) verified
- [ ] Integration with PRD-029 (Guest Brief) verified
