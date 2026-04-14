# PRD-026: Topic Entity View

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: TBD
- **Related RFCs**:
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` -- CIL topic
    identity (`topic:{slug}`) and bridge artifact that powers cross-episode queries
  - `docs/rfc/RFC-073-enrichment-layer-architecture.md` -- `topic_cooccurrence` and
    `temporal_velocity` enrichers that provide the data this view consumes
  - `docs/rfc/RFC-062-gi-kg-viewer-v2.md` -- viewer SPA shell and tab model
  - `docs/rfc/RFC-069-graph-exploration-toolkit.md` -- graph chrome (Cytoscape)
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` -- KG MENTIONS edges and Topic nodes
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` -- GIL Insight and Quote nodes
- **Related PRDs**:
  - `docs/prd/PRD-019-knowledge-graph-layer.md` -- KG artifact foundation
  - `docs/prd/PRD-017-grounded-insight-layer.md` -- GIL artifact foundation
  - `docs/prd/PRD-024-graph-exploration-toolkit.md` -- graph chrome this view extends
  - `docs/prd/PRD-025-corpus-intelligence-dashboard-viewer.md` -- operational dashboard
    (distinct: PRD-025 is operator/pipeline health; this PRD is end-user intelligence)
  - `docs/prd/PRD-027-enriched-search.md` -- enriched search; topic pills in search
    results open this view
  - `docs/prd/PRD-028-position-tracker.md` -- Position Tracker links back via
    "View topic"; person chips here can open Position Tracker
  - `docs/prd/PRD-029-guest-intelligence-brief.md` -- Guest Brief links back via
    "View topic"; person chips here can open Guest Brief
- **Related UX specs**:
  - `docs/uxs/UXS-007-topic-entity-view.md` -- visual contract for Topic Entity View
    panel layout, sections, degradation states
  - `docs/uxs/UXS-001-gi-kg-viewer.md` -- shared design system (tokens, typography,
    states)
- **Related Architecture**:
  - `docs/architecture/gi/ontology.md`
  - `docs/architecture/kg/ontology.md`

---

## Summary

A **Topic Entity View** is a first-class navigable page in the GI/KG viewer where
`topic:{slug}` is the subject -- not an episode, not a graph node detail panel. The user
navigates *to a topic* and sees the entire corpus through that lens: which episodes discuss
it, how frequently over time, what the most insightful things said about it are, and who
said them.

This is a qualitatively different navigation model than the existing viewer, which is
episode-first and graph-first. Topic Entity View is **concept-first** -- it answers
"what does this corpus know about X?" rather than "what happened in episode Y?".

It requires no LLM and no database. All data comes from deterministic enricher
outputs (`topic_cooccurrence`, `temporal_velocity`) and the core GIL/KG/bridge
artifacts produced by RFC-072.

---

## Background

The GI/KG viewer currently shows per-episode graph views, a corpus library, a digest,
and a search panel. Topics appear as nodes in the graph and as filter values in search.
But there is no way to navigate *to* a topic and see the full corpus picture for that
concept.

The `temporal_velocity` enricher (RFC-073) computes monthly topic mention counts and
trend signals across the corpus. The `topic_cooccurrence` enricher computes which topics
appear together. The bridge (RFC-072) links KG MENTIONS edges (which episodes) with GIL
ABOUT edges (what Insights were grounded). These three artifacts together contain
everything needed for a rich topic view -- they just have no UI surface.

PRD-025 (Dashboard) shows operational corpus health. This PRD shows content intelligence
about a specific topic. They are complementary, not competing.

---

## Goals

1. Make `topic:{slug}` a first-class navigable entity in the viewer -- not just a node
   in the graph.
2. Show the temporal footprint of a topic across the corpus -- when it appeared, how
   often, trend direction.
3. Surface the most insightful things said about the topic, grounded in verbatim quotes
   with timestamps.
4. Show which persons discussed the topic and how frequently.
5. Show related topics (co-occurrence signal) so users can navigate the concept space.
6. Work entirely from deterministic enricher outputs and core artifacts -- no LLM, no
   database, no new extraction.

---

## Non-Goals

- **Not** automated contradiction detection or stance comparison across persons -- that
  is a future analysis layer, dependent on NLI enricher.
- **Not** a cross-show topic view -- single corpus only.
- **Not** a replacement for the graph view -- the Cytoscape topic node and its edges
  remain. Topic Entity View is a complementary surface.
- **Not** natural language topic search within the view -- topic selection uses the
  existing search panel or graph node click, not a new query input.
- **Not** a new tab in the main navigation -- see UX section below.

---

## User Stories

**As a podcast listener**, I want to click on a topic in the graph or search results
and see a full picture of how that topic has been discussed across all episodes, so I
can understand what the podcast has actually said about it rather than just finding
individual episodes.

**As a corpus explorer**, I want to see which topics are trending up in the corpus and
navigate to the hottest ones, so I can discover what the podcast has been increasingly
focused on lately.

**As a researcher**, I want to see who in the corpus has discussed a topic most and what
their key positions are (with timestamps), so I can identify the authoritative voices on
that subject within the corpus.

---

## User-Facing Requirements

### Entry Points

**FR1.1** -- Clicking a Topic node in the Cytoscape graph opens the Topic Entity View
in the right rail panel (same mechanism as existing node detail, but expanded to full
Topic Entity layout).

**FR1.2** -- Topic pills in episode search results and Library episode rows are
clickable and open the Topic Entity View.

**FR1.3** -- The `temporal_velocity` enricher output in the Dashboard Content
Intelligence section includes clickable topic rows that open the Topic Entity View.

### Topic Header

**FR2.1** -- Topic display name and canonical `topic:{slug}` ID.

**FR2.2** -- Total episode count where this topic appears in the corpus.

**FR2.3** -- Trend badge: `accelerating` / `stable` / `declining` / `insufficient data`
derived from `temporal_velocity` enricher output. Badge uses intent tokens:
`success` for accelerating, `muted` for stable, `warning` for declining.

**FR2.4** -- Date range of first and last appearance in the corpus.

### Timeline Section

**FR3.1** -- A monthly bar chart (Chart.js, `series-1` token) showing episode mention
count per month, derived from `temporal_velocity` enricher output.

**FR3.2** -- Bars are clickable -- clicking a bar filters the Insights section (FR4)
to that month only.

**FR3.3** -- When `sources.gi: true` appears in the bridge for a given episode, the
corresponding bar segment is visually distinguished (GI-enriched vs KG-only) using
`gi` domain token. This indicates months where structured Insights exist vs months
where the topic was mentioned but not deeply analysed.

**FR3.4** -- An insight line below the chart when the data supports a clear takeaway
(e.g. "This topic appeared in 14 episodes in Q1 2026, up from 2 in 2024"). This text
is computed by the viewer from the `temporal_velocity` enricher data (monthly counts
and trend signal) -- it is not a separate enricher output.

### Insights Section

**FR4.1** -- A scrollable list of grounded Insights (`grounded: true`) from GIL that
are linked to this topic via `ABOUT` edges, ordered by episode publish date descending
(most recent first).

**FR4.2** -- Each Insight card shows: insight text, speaker name (if `SPOKEN_BY`
available), episode title and publish date, and the supporting verbatim quote with
a timestamp jump link.

**FR4.3** -- When no grounded Insights exist for this topic (topic is KG-only, no GIL
ABOUT edges), show an honest empty state: "No grounded insights yet for this topic.
Insights appear when the pipeline runs with GIL extraction enabled."

**FR4.4** -- Insight cards are filterable by person (clicking a person chip in the
Persons section filters the Insights list to that person).

**FR4.5** -- Maximum 20 Insights shown by default; "Show more" loads additional results.

### Persons Section

**FR5.1** -- A list of persons who discussed this topic (resolved via bridge
`person:{slug}` -> GIL SPOKEN_BY -> Quote -> SUPPORTED_BY -> Insight -> ABOUT ->
topic), showing display name, Insight count, and most recent episode date.

**FR5.2** -- Persons are sorted by Insight count descending.

**FR5.3** -- Clicking a person chip filters the Insights section (FR4) to that person.
A secondary action (e.g. long-press or dedicated link) opens the Person Landing
(PRD-029) for that person, giving access to their Guest Brief and Position Tracker.

**FR5.4** -- When the `grounding_rate` corpus enricher has run, each person chip
shows a small grounding quality badge: the percentage of their Insights that are
grounded (`grounded: true`) across the corpus. Uses intent tokens: `success` for
>= 80%, `muted` for 50-79%, `warning` for < 50%. When the enricher has not run, the
badge is hidden (graceful degradation).

**FR5.5** -- When diarization data is unavailable, persons may not be attributable
-- show a note: "Speaker attribution requires diarization. Enable pyannote in pipeline
config."

### Related Topics Section

**FR6.1** -- A compact list of topics that co-occur most frequently with this topic,
derived from `topic_cooccurrence` corpus enricher output.

**FR6.2** -- Each related topic shows its display name and co-occurrence count.

**FR6.3** -- Clicking a related topic navigates to that topic's Topic Entity View.

**FR6.4** -- Maximum 8 related topics shown.

### UX Placement

**FR7.1** -- Topic Entity View is presented in the **right rail panel** of the viewer
(same panel used for graph node detail, episode detail in Library). It is not a new
main tab.

**FR7.2** -- The panel header shows "Topic" with the `kg` domain token color, consistent
with KG identity coloring in UXS-001.

**FR7.3** -- A "View in graph" button navigates to the Graph tab and focuses/centers
on this topic's node in the Cytoscape canvas.

**FR7.4** -- A "Search this topic" button prefills the semantic search panel with the
topic label, consistent with existing "Search topic" affordance in Digest (UXS-001).

---

## Data Sources

| Section | Source | Enricher required |
|---|---|---|
| Header (count, trend, dates) | `temporal_velocity` corpus enricher | Yes -- `temporal_velocity` |
| Timeline chart | `temporal_velocity` corpus enricher | Yes -- `temporal_velocity` |
| GI-enriched bar segments | `bridge.json` per episode (`sources.gi`) | No (bridge is core) |
| Insights list | `gi.json` (ABOUT edges -> Insight -> Quote nodes) | No (core artifact) |
| Person attribution | `gi.json` (SPOKEN_BY edges) + `bridge.json` | No (core artifacts) |
| Person grounding badge | `grounding_rate` corpus enricher (RFC-073) | Yes -- `grounding_rate` |
| Related topics | `topic_cooccurrence` corpus enricher | Yes -- `topic_cooccurrence` |

**Graceful degradation:** If enrichers have not run, the view degrades cleanly:
- Missing `temporal_velocity` -> Timeline section shows empty state with a note:
  "Run `podcast enrich` to generate topic analytics."
- Missing `topic_cooccurrence` -> Related Topics section is hidden entirely.
- Missing GIL Insights -> Insights section shows honest empty state (FR4.3).
- Core artifacts only -> Header shows episode count from KG MENTIONS scan; Timeline
  hidden; Insights hidden; Related Topics hidden. View is still useful for basic
  topic identity.

---

## API Requirements

**New endpoint required:**

`GET /api/topics/{topic_id}` -- returns Topic Entity View data assembled from
enricher outputs and core artifacts for a given canonical topic ID (e.g.
`topic:ai-regulation`). Path segments use full canonical IDs, consistent with the
CIL convention established in RFC-072.

Response shape:

```json
{
  "topic_id": "topic:ai-regulation",
  "display_name": "AI Regulation",
  "episode_count": 46,
  "trend": "accelerating",
  "date_range": {
    "earliest": "2023-01-15",
    "latest": "2026-04-10"
  },
  "monthly_counts": {
    "2024-01": 2,
    "2026-03": 14
  },
  "insights": [ ],
  "persons": [ ],
  "related_topics": [ ]
}
```

The endpoint reads from enricher output files and core artifacts on the filesystem --
no database required. This is the **first read-side consumer** of RFC-073 enrichment
outputs, driving the server changes that RFC-073 Section "Server and Viewer
Consumption" explicitly deferred to a follow-up PRD.

**Pagination:** The `insights` array supports `limit` and `offset` query parameters
(default `limit=20`, matching FR4.5). The viewer's "Show more" control fetches the
next page. `persons`, `related_topics`, and `monthly_counts` are returned in full
(small cardinality, no pagination needed).

`GET /api/topics` -- returns a paginated list of all topics in the corpus with
count, trend, and date range. Used to populate topic navigation and the Dashboard
Content Intelligence section.

---

## Success Criteria

1. A user can click a topic node in the graph and see the full Topic Entity View in
   the right rail without leaving the viewer session.
2. The timeline chart correctly reflects the corpus topic history derived from
   `temporal_velocity` enricher output.
3. At least one grounded Insight with a verbatim quote and timestamp is visible for
   any topic that has GIL ABOUT edges in the corpus.
4. Related topics are shown when `topic_cooccurrence` enricher has run.
5. The view degrades gracefully when enrichers have not run -- no broken panels, honest
   empty states.
6. Clicking "View in graph" focuses the Cytoscape canvas on the topic node.
7. Clicking "Search this topic" prefills the search panel correctly.
8. The `kg` domain token is used for Topic identity elements, consistent with UXS-001.
9. Playwright E2E coverage for the Topic Entity View right rail, including empty state
   and enricher-populated state.

---

## References

- `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
- `docs/rfc/RFC-073-enrichment-layer-architecture.md`
- `docs/rfc/RFC-062-gi-kg-viewer-v2.md`
- `docs/uxs/UXS-001-gi-kg-viewer.md`
- `docs/architecture/kg/ontology.md`
- `docs/architecture/gi/ontology.md`
