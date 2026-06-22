# RFC-097 chunk 8 follow-up tickets (drafted 2026-06-21)

The original chunk-8 scope ("viewer Position Tracker + Person Profile + edge
styling") was scope-cut on 2026-06-21 so the v2 foundation PR closes without
a multi-week UI build. The three flagship-view items are queued here as
follow-up tickets. The edge styling work stays in chunk 8 (in-scope).

These are **draft ticket bodies** — operator alone opens GH issues
(`feedback_never_open_gh_issues`). To open: copy each block into
`gh issue create --title "..." --body "..."`.

Order: A is shared infra; B and C depend on A and can ship in parallel
once A is in.

---

## Ticket A — Person Landing shared component

**Title**: `viewer: Person Landing shared component (RFC-097 chunk 8 follow-up)`

**Body**:

Per RFC-097 chunk-8 scope-cut (2026-06-21), the Person Landing shared
component is split into its own follow-up so the v2 foundation PR
closes cleanly. Person Landing is the shared shell that hosts both the
Position Tracker (Ticket B) and Person Profile (Ticket C) views and
provides identity header + topic overview + cross-navigation between
the two surfaces.

### Scope

- Identity header: `Person.name`, `Person.role` (host / guest / mention),
  episodes-appeared-in count, organizations-affiliated chips
- Topic overview: top-N topics this Person discussed, ranked by count of
  `ABOUT (Insight → Topic)` edges where the Insight has a
  `MENTIONS_PERSON` edge to this Person
- Cross-navigation: tabs / pivot between Position Tracker (per Topic
  drill-in) and Person Profile (all-topics view)
- Entry points: open from search results, KG node click on a Person
  node, or direct URL (`/person/<person_id>`)

### Data foundation (already shipped in RFC-097 v2)

- Typed `Person` nodes (chunk 3)
- `MENTIONS_PERSON` edges from Insights (chunk 4)
- `Person` IDs stable via RFC-072 CIL bridge

### Out of scope

- Position Tracker timeline rendering — Ticket B
- Person Profile aggregate views — Ticket C

### Spec references

- PRD-029: `docs/prd/PRD-029-person-profile.md` — owns the shared
  Person Landing component definition
- UXS-001: `docs/uxs/UXS-001-gi-kg-viewer.md` — shared design system

### Acceptance

- Person Landing loads cleanly for any Person id from the prod-v2 corpus
  (99 eps)
- Tabs render Position Tracker placeholder and Person Profile placeholder
  (filled by Tickets B / C)
- Vitest unit coverage on the identity-header + topic-overview components
- Playwright e2e: open Person Landing from search result, switch tabs,
  back navigation

### Sizing

~1–2 days

### Blocked by

- None (data foundation shipped in chunks 1–7)

### Blocks

- Ticket B (Position Tracker view)
- Ticket C (Person Profile view)

---

## Ticket B — Position Tracker view (PRD-028)

**Title**: `viewer: Position Tracker (Person × Topic over time) — RFC-097 chunk 8 follow-up`

**Body**:

Per RFC-097 chunk-8 scope-cut (2026-06-21), the Position Tracker view
is split into its own follow-up. Position Tracker traces how a Person's
stated positions on a Topic evolved across episodes and time — the
flagship use case RFC-072 named but never delivered.

### Scope

- Query: `position_arc(person_id, topic_id)` per RFC-072 §5A
- Timeline rendering: insights ordered by `position_hint` (within-episode
  position) then `Episode.publish_date` (cross-episode time)
- `insight_type` filter: claim / recommendation / observation / question /
  unknown — multi-select chips
- Quote cards: each insight shows its source quote(s) via
  `SUPPORTED_BY → Quote → SPOKEN_BY` chain
- Empty / degraded states per UXS-009
- Entry: from Person Landing (Ticket A) Topic chip → Position Tracker for
  that Person × Topic pair

### Data foundation (already shipped in RFC-097 v2)

- `ABOUT (Insight → Topic)` + `MENTIONS_PERSON (Insight → Person)` edges
  (chunk 4)
- `Insight.insight_type` LLM-classified field (chunk 5)
- `Insight.position_hint` 4-step waterfall (chunk 5)
- `Episode.publish_date` (already shipped)
- `Quote.speaker` via `SPOKEN_BY` (already shipped)

### Out of scope

- Position-change detection / NLI between insights — RFC-088 analysis
  layer territory (v3)
- Multi-corpus cross-show view — multi-corpus aggregation deferred
- Inline contradiction badges — RFC-097 §464 CONTRADICTS edges deferred
  to v3

### Spec references

- PRD-028: `docs/prd/PRD-028-position-tracker.md` — capability spec
- UXS-009: `docs/uxs/UXS-009-position-tracker.md` — visual contract
- RFC-072 §5A: `position_arc` query pattern

### Acceptance

- Position Tracker renders for any (Person, Topic) pair present in the
  prod-v2 corpus (99 eps)
- Timeline correctly orders insights by `position_hint` within an
  episode and by `publish_date` across episodes
- `insight_type` filter chips correctly partition the timeline
- Quote-card grounding chain (Insight ← SUPPORTED_BY ← Quote ← SPOKEN_BY)
  resolves to the canonical Person
- UXS-009 degradation states all reachable in stack-test
- Vitest + Playwright coverage; `ci-ui-full` green

### Sizing

~3–4 days

### Blocked by

- Ticket A (Person Landing)

---

## Ticket C — Person Profile view (PRD-029)

**Title**: `viewer: Person Profile (everything about a Person) — RFC-097 chunk 8 follow-up`

**Body**:

Per RFC-097 chunk-8 scope-cut (2026-06-21), the Person Profile view is
split into its own follow-up. Person Profile is the all-topics aggregate
view for a Person: insights voiced, quotes attributable, topics discussed,
episodes appeared in, organizations associated with — RFC-072's other
flagship use case.

### Scope

- Query: `person_profile(person_id)` per RFC-072 §5B
- Aggregate sections per UXS-010:
  - Insights voiced (grouped by Topic via `ABOUT` edges)
  - Quotes attributable (`SPOKEN_BY` from this Person)
  - Topics discussed (chips ranked by insight count)
  - Episodes appeared in (`SPOKE_IN` edges)
  - Organizations affiliated (`MENTIONS_ORG` co-occurrence + Person
    membership if present)
- Each topic group links to Position Tracker (Ticket B) for that
  Person × Topic pair
- Empty / degraded states per UXS-010

### Data foundation (already shipped in RFC-097 v2)

- Typed `Person` + `Organization` nodes (chunk 3)
- `ABOUT` / `MENTIONS_PERSON` / `MENTIONS_ORG` edges (chunk 4)
- `Insight.insight_type` (chunk 5)
- `SPOKE_IN` (existing) + `SPOKEN_BY` (existing) edges

### Out of scope

- Analytical ranking / curation — RFC-088 analysis layer (v3)
- Multi-corpus aggregation — deferred
- Person ↔ Person relationships (co-host frequency, etc.) — v3 territory

### Spec references

- PRD-029: `docs/prd/PRD-029-person-profile.md` — capability spec
- UXS-010: `docs/uxs/UXS-010-person-profile.md` — visual contract
- RFC-072 §5B: `person_profile` query pattern

### Acceptance

- Person Profile renders for any Person id present in the prod-v2 corpus
- Each section's count matches the underlying edge count (no silent
  truncation; if truncated, "+N more" affordance)
- Topic group → Position Tracker link works
- UXS-010 degradation states reachable
- Vitest + Playwright coverage; `ci-ui-full` green

### Sizing

~3–4 days

### Blocked by

- Ticket A (Person Landing)

---

## After all three land

Both PRDs (PRD-028, PRD-029) carry "v2 closure (RFC-097, 2026-06-20)"
stanzas. When the corresponding ticket merges, update those stanzas:

- PRD-028: replace "viewer surface delivered by RFC-097 v2 (chunk 8)"
  with "viewer surface delivered by #&lt;ticket-B-num&gt;"
- PRD-029: replace same line with "viewer surface delivered by
  #&lt;ticket-A-num&gt; (landing) + #&lt;ticket-C-num&gt; (profile)"

And in `docs/rfc/RFC-097-unified-kg-gi-ontology-v2.md` line 11–12 the
"viewer surface deferred to follow-up ticket — see chunk 8 scope-cut"
annotations get the same ticket-number substitution.
