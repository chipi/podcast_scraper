# PRD-029: Person Profile

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: TBD
- **Related RFCs**:
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` -- CIL identity
    layer, bridge artifact, and query Pattern B (`person_profile`)
  - `docs/rfc/RFC-062-gi-kg-viewer-v2.md` -- viewer SPA shell and tab model
  - `docs/rfc/RFC-069-graph-exploration-toolkit.md` -- graph chrome (Cytoscape)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` -- GIL Insight and Quote nodes
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` -- KG Episode and Entity nodes
  - `docs/rfc/RFC-061-semantic-corpus-search.md` -- FAISS search and chunk-to-Insight
    lift (Phase 5)
- **Related PRDs**:
  - `docs/prd/PRD-017-grounded-insight-layer.md` -- GIL artifact foundation
  - `docs/prd/PRD-019-knowledge-graph-layer.md` -- KG artifact foundation
  - `docs/prd/PRD-026-topic-entity-view.md` -- topic-first navigation; cross-linked
    from each topic group in the profile
  - `docs/prd/PRD-027-enriched-search.md` -- enriched search entry point; speaker
    names in enriched sources can open Person Profile via Person Landing
  - `docs/prd/PRD-028-position-tracker.md` -- companion flagship; each topic group
    in the profile links to the Position Tracker for that person + topic
- **Related UX specs**:
  - `docs/uxs/UXS-010-person-profile.md` -- visual contract for Person
    Profile panel layout and degradation states
  - `docs/uxs/UXS-001-gi-kg-viewer.md` -- shared design system (tokens, typography,
    states)
- **Related Architecture**:
  - `docs/architecture/gi/ontology.md`
  - `docs/architecture/kg/ontology.md`

---

## Summary

The **Person Profile** answers: "What has person X talked about across this corpus?"

It is a dedicated viewer panel where `person:{slug}` is the subject. The user
navigates to a person and sees their corpus presence at a glance: identity metadata,
the list of topics they have discussed (with Insight counts), their episode
appearances, and navigation links to drill deeper via the Position Tracker (PRD-028)
or Topic Entity View (PRD-026).

The Person Profile provides the **shared Person Landing component** used by both this
PRD and PRD-028 (Position Tracker).

No LLM and no database are required. All data comes from the CIL bridge artifact,
GIL Insights, and KG episode metadata.

---

## Background

RFC-072 defines the Person Profile as Flagship 2 of the Canonical Identity Layer.
The CIL bridge makes corpus-wide person queries possible by scanning small
`bridge.json` files. The query pattern (`person_profile`) is defined in RFC-072
and exposed via `GET /api/persons/{person_id}/brief` (GitHub #527).

This PRD defines the person surface: the Person Landing component, identity header,
topic overview, cross-navigation, and degraded states.

PRD-028 (Position Tracker) is the companion flagship. Clicking a topic group in the
Person Profile navigates to the Position Tracker for that person + topic combination,
enabling a drill-down from "what topics does this person discuss?" to "how has their
position on this specific topic evolved?"

---

## Goals

1. Make `person:{slug}` a navigable concept in the viewer -- the user can see a
   person's corpus presence and topic footprint.
2. Own the shared Person Landing component that hosts both Person Profile and
   Position Tracker (PRD-028).
3. Present the list of topics a person has discussed, with Insight counts per topic.
4. Enable multiple entry points: graph Person node click, search result speaker click,
   and a dedicated person browse.
5. Cross-link to PRD-028 Position Tracker (per-topic drill-down) and PRD-026 Topic
   Entity View (topic-first navigation).
6. Work entirely from core artifacts (bridge, GIL, KG) -- no LLM, no database, no
   new extraction.

---

## Non-Goals

- **Not** analytical ranking or curation -- strongest Insight ranking, best-quote
  ranking, and summary statistics are out of scope.
- **Not** contradiction detection -- potential challenges and cross-person stance
  comparison are out of scope.
- **Not** a conversational interface -- the profile is a static, structured view.
- **Not** audio playback -- timestamps are displayed as text, not playable clips.
- **Not** a cross-corpus view -- single corpus only.
- **Not** a replacement for the graph view -- the Cytoscape Person node and its edges
  remain. The Person Profile is a complementary surface.
- **Not** a new top-level tab -- it opens in the right rail panel or a dedicated route,
  depending on entry point.
- **Not** position-change detection -- out of scope.

---

## Personas

- **User**: A single person operating the podcast scraper tool. They browse a
  person's corpus presence, see which topics they have discussed, and navigate to
  deeper views (Position Tracker, Topic Entity View, Graph).

---

## User Stories

- _As a user, I can click a person in the graph and see their identity, episode
  appearances, and topic footprint so that I understand their corpus presence at a
  glance._
- _As a user, I can see which topics a person has discussed and how many Insights
  exist per topic so that I know what to explore further._
- _As a user, I can click a topic to drill into the Position Tracker (PRD-028) so
  that I can see how the person's thinking on that topic evolved._
- _As a user, I can click "View topic" to navigate to the Topic Entity View (PRD-026)
  so that I see the topic from a corpus-wide perspective._
- _As a user, I can click "View in graph" to return to the Cytoscape graph focused on
  this person's node._

---

## Functional Requirements

### FR1: API Layer

The Person Profile API uses the existing CIL endpoint with a lightweight response.

- **FR1.1** (existing): `GET /api/persons/{person_id}/brief` returns a
  `CilPersonProfileResponse` with Insights grouped by topic and a flat list of quotes.
- **FR1.2** (new): The response must include `person_display_name` resolved from the
  bridge `identities` array. Each topic key in the `topics` dict must be accompanied
  by a `topic_display_names: dict[str, str]` mapping topic IDs to display names.
- **FR1.3** (new): The response must include `appearances: int` (total episodes where
  this person appears in the corpus via bridge scan) and `date_range` (`earliest` and
  `latest` publish dates).
- **FR1.4** (new): Each topic group must include `insight_count: int` -- the number of
  Insights for this person on this topic.
- **FR1.5** (new): The response must include `person_id` (already present) and
  `path` (already present) for consistency with other CIL responses.

### FR2: Viewer Panel -- Person Profile

- **FR2.1**: This PRD **owns the shared Person Landing component** -- the
  entry-point surface that hosts both Person Profile and Position Tracker (PRD-028).
  The Person Landing opens from three entry points:
  - Clicking a Person node in the Cytoscape graph (right rail panel).
  - Clicking a speaker name in a search result card (including lifted results).
  - A dedicated person browse (accessible from the viewer navigation).

  The Person Landing defaults to the Person Profile tab. A tab or toggle switches to
  the Position Tracker (PRD-028). PRD-028 references this component; it does not
  define its own landing.
- **FR2.2**: Person header section shows:
  - Person display name (`text-lg font-semibold`).
  - Canonical `person:{slug}` ID (`muted`, `text-xs`, monospace).
  - Appearance count: "N episodes" badge.
  - Date range of first and last appearance.
- **FR2.3**: Topic Overview section:
  - A list of topics this person has discussed, sorted by Insight count descending
    (most-discussed topic first).
  - Each topic row shows: topic display name, Insight count badge.
  - Each topic row has two navigation links:
    - "Track positions" navigates to the Position Tracker (PRD-028) for this
      person + topic.
    - "View topic" navigates to the Topic Entity View (PRD-026) for that topic.
- **FR2.4**: Cross-navigation:
  - "View in graph" button navigates to the Graph tab and focuses on this person's
    node.
  - "Track positions" on each topic row navigates to Position Tracker (PRD-028).
  - "View topic" on each topic row navigates to Topic Entity View (PRD-026).

### FR3: Empty and Degraded States

These are acceptance criteria -- each must have defined UI behavior and test coverage.

- **FR3.1**: Person has 0 Insights (appears in KG but no GIL attribution):
  - The person header (FR2.2) still shows with metadata from KG (name, episode count
    from MENTIONS edges).
  - The Topic Overview section shows: "No grounded insights found for this person.
    Insights appear when the pipeline runs with GIL extraction enabled."
- **FR3.2**: A topic has 0 Insights (edge case -- topic appears in bridge but no
  ABOUT edges in GIL):
  - The topic row appears in the Topic Overview with "0 insights" badge.
- **FR3.3**: Lift fails (char offset mismatch on search entry point):
  - The search result card shows the raw transcript chunk without a speaker link.
  - No Person Profile navigation is offered from that card.
- **FR3.4**: No `bridge.json` for some or all episodes:
  - Episodes without a bridge file are silently excluded from the profile.
  - If all episodes lack bridge files, the profile area shows: "No cross-layer data
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
Load gi.json per episode: ABOUT edges for topic list
    |
    v
Group topics; count Insights per topic
    |
    v
Viewer renders profile: header, topic overview, navigation
```

---

## API Response Shape

The Person Profile uses a subset of the `CilPersonProfileResponse`:

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
  "topic_display_names": {
    "topic:ai-regulation": "AI Regulation",
    "topic:open-source": "Open Source"
  },
  "topics": {
    "topic:ai-regulation": {
      "insight_count": 5,
      "insights": []
    },
    "topic:open-source": {
      "insight_count": 3,
      "insights": []
    }
  }
}
```

The `insights` array within each topic is available in the API response but not
rendered by the Person Profile viewer. Future extensions may use it.

---

## Success Criteria

1. A user can click a Person node in the graph and see the Person Profile in the
   right rail without leaving the viewer session.
2. The person header shows display name, slug, appearance count, and date range.
3. The Topic Overview lists all topics with correct Insight counts, sorted by count
   descending.
4. Cross-navigation to Position Tracker (PRD-028), Topic Entity View (PRD-026), and
   Graph all work from the Person Profile panel.
5. All four degraded states (FR3.1 -- FR3.4) render correctly with honest messaging
   and no broken UI.
6. Playwright E2E coverage for: populated profile, empty person, and missing bridge
   states.

---

## Dependencies

- **RFC-072** (Phases 1--4): CIL identity, bridge artifact, `person_profile` query
  pattern, and CIL API endpoints.
- **PRD-026** (Topic Entity View): cross-linked from each topic row.
- **PRD-027** (Enriched Search): entry point via speaker names in enriched sources.
- **PRD-028** (Position Tracker): companion flagship; "Track positions" navigates
  there. References the Person Landing component owned by this PRD.

---

## Constraints and Assumptions

**Constraints:**

- Must run on Apple M4 Pro with 48 GB RAM (local-first tool).
- Profile assembly must complete in < 2 seconds for a corpus of 200 episodes.
- No database -- all data read from filesystem artifacts.

**Assumptions:**

- The user has run the pipeline with GIL extraction and bridge generation enabled.
- `bridge.json` files exist for the episodes the user wants to explore.
- Display names are available in the bridge `identities` array.

---

## Design Considerations

### Extension Point Design

The API returns a `CilPersonProfileResponse` that includes the full `insights` array
per topic and other optional fields. The Person Profile viewer uses only the subset
defined in this PRD (display names, topic counts, appearances, date range). Additional
response fields are available for future extensions. The viewer conditionally renders
sections when data is present and omits them otherwise -- no feature flags are needed.

---

## Known Limitations

- **No analytical ranking**: The Person Profile shows raw topic lists and Insight
  counts. It does not highlight the strongest position or curate quotes.
- **Single-episode corpus**: The profile shows one episode's worth of data. Topic
  diversity and cross-episode patterns are minimal. The viewer does not show a special
  state -- it displays whatever data exists.

---

## Open Questions

1. **Appearance count source**: Bridge scan (GIL attribution) vs KG (mentions).
   Recommend: bridge scan for consistency with the profile's CIL-centric data.
   (Same question as PRD-028.)

---

## Related Work

- Issue #527: CIL query API implementation
- PRD-026: Topic Entity View
- PRD-028: Position Tracker
- RFC-072: Canonical Identity Layer

---

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] API fields implemented (FR1.2 -- FR1.5)
- [ ] Person Landing component implemented with all entry points (FR2.1)
- [ ] Person header implemented (FR2.2)
- [ ] Topic Overview implemented (FR2.3)
- [ ] Cross-navigation verified (FR2.4)
- [ ] Empty/degraded states implemented and tested (FR3.1 -- FR3.4)
- [ ] Playwright E2E coverage for populated and empty states
- [ ] Documentation updated (SERVER_GUIDE, E2E_SURFACE_MAP)
- [ ] Integration with PRD-028 (Position Tracker) verified
- [ ] Integration with PRD-026 (Topic Entity View) verified
