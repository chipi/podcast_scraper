# PRD-023: Corpus Digest & Library Glance

- **Status**: Implemented (v2.6.0) — shipped per [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md) (`GET /api/corpus/digest`, Digest tab, Library 24h glance, `corpus_digest_api` on health)
- **Authors**: Podcast Scraper Team
- **Related RFCs**:
  - [RFC-068: Corpus Digest API & Viewer](../rfc/RFC-068-corpus-digest-api-viewer.md) — includes **optional visual metadata** on digest rows and topic hits (same catalog source as RFC-067)
  - [RFC-067: Corpus Library](../rfc/RFC-067-corpus-library-api-viewer.md)
  - [RFC-061: Semantic Corpus Search](../rfc/RFC-061-semantic-corpus-search.md)
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Related PRDs**:
  - [PRD-022: Corpus Library](PRD-022-corpus-library-episode-browser.md)
  - [PRD-021: Semantic Corpus Search](PRD-021-semantic-corpus-search.md)
  - [PRD-017: Grounded Insight Layer](PRD-017-grounded-insight-layer.md)
  - [PRD-019: Knowledge Graph Layer](PRD-019-knowledge-graph-layer.md)
- **Related UX specs**:
  - [UXS-002: Corpus Digest](../uxs/UXS-002-corpus-digest.md) -- Digest tab layout,
    topic bands, recent episodes, health discovery
  - [UXS-001: GI/KG Viewer](../uxs/UXS-001-gi-kg-viewer.md) -- shared design system

## Summary

**Corpus Digest** helps users **prioritize** new and recent episodes across **many feeds**
without scanning the full **Corpus Library** (PRD-022). It surfaces a **small, diverse**
set of episodes for a **calendar time window**, shows **summary context** from existing
metadata, exposes **GI/KG readiness**, and (when metadata includes RSS-sourced fields) may show
**optional artwork and duration** consistent with the **Library** (PRD-022 / RFC-067), and
ties in **semantic search** via **global fixed topics** so users can pivot from “what landed”
to “what matches what I care about.” The
experience ships in **two surfaces**: a **Digest** main tab (full view) and a **Library
glance** region (24-hour snapshot, **default expanded**, collapsible for focus).

## Background & Context

At multi-feed scale (on the order of **100 feeds** and **thousands of episodes**, with
**tens of new episodes per day**), the Library answers *inventory* questions well but
does not, by itself, answer *“what should I open first?”* Semantic search (PRD-021) and
graph artifacts (PRD-017, PRD-019) add power after the user chooses a direction; Digest
reduces the **attention cost** of that first choice.

Digest **reuses** the same **filesystem-backed** metadata and **viewer shell** as PRD-022
/RFC-067. It does **not** require “since last visit” or other **persistent read-state** for
MVP: windows are **calendar-based** (simpler, reproducible, testable). RFC-068 specifies
APIs, ranking, topic configuration, and performance budgets.

## Goals

- **G1 — Glance:** From the **Library** tab, a user sees a **default-expanded** **24-hour**
  digest strip: **feed-diverse** rows with enough context to open detail, graph, or search.
- **G2 — Full Digest:** A **Digest** main tab offers a **longer default window** (e.g.
  **last 7 days**, user-selectable presets) with **semantic topic** sections driven by a
  **global fixed topic list** (versioned defaults; empty topics hidden or de-emphasized).
- **G3 — GI/KG value:** Rows show **honest signals**: at minimum **`has_gi` / `has_kg`**
  (and paths for handoff); optional **lightweight counts** when cheap to obtain without
  full graph render (RFC-068 defines).
- **G4 — Handoffs:** Same **explicit handoffs** as Library: **Open GI/KG** (Graph tab),
  **Search in corpus** (Search panel) with **meaningful** feed/query context per RFC-068.
- **G5 — Phasing:** Requirements are **layered** (L0 metadata → L1 semantic → L2 GI/KG
  signals) so implementation can ship **incrementally** without renegotiating product
  intent.

## Non-Goals

- **No “since last opened” / unread persistence** for MVP (calendar windows only).
- **No auto-generated narrative “daily brief”** prose from an LLM in MVP (summaries come
  from existing metadata bullets where present).
- **No per-user or per-corpus topic lists** in MVP (global fixed topics only; overrides
  are a later phase).
- **No mobile-first layouts**; desktop baseline per UXS-001/UXS-002.
- **No cross-episode KG analytics** (entity trending, sentics, etc.) in MVP beyond what
  RFC-068 lists for L2.
- **Generic HTTP image proxy** for arbitrary remote URLs without ingest — out of scope
  (same as PRD-022). **Optional corpus-local artwork** (RFC-067 Phase 4): when metadata
  includes verified `image_local_relpath` fields and files exist, digest APIs expose the same
  `feed_image_local_relpath` / `episode_image_local_relpath` as Library rows; the viewer
  prefers same-origin `/api/corpus/binary` URLs. Default remains URL-only when download was
  not run (`download_podcast_artwork` off or paths missing).

## Personas

- **Corpus operator / analyst:** Runs `podcast serve`, holds large corpora; needs a fast
  **orientation** surface before deep graph or search work.
- **Developer:** Validates pipeline output; uses Digest to confirm **recent episodes**
  and **artifact presence** without paging the entire Library.

## User Stories

- _As an operator, I can open **Digest** and see **recent episodes across feeds** so that
  one noisy show does not hide everything else._
- _As an operator, I can see **which episodes have GI/KG** ready so that I open the graph
  only when artifacts exist._
- _As an operator, I can jump from a **topic** into **semantic search** scoped to my
  corpus so that I explore beyond the digest list._
- _As an operator, I can see a **24-hour glance** inside **Library** so that I do not
  leave browse mode to notice what is new._
- _As an operator, I can **collapse** the glance region so that it does not compete with
  the feed/episode columns when I want focus._

## Functional Requirements

### FR1 — Two surfaces, one product intent

- **FR1.1:** **Digest tab** (label **Digest**) provides the **full** digest experience
  (window presets, topic bands, richer layout) per UXS-002 and RFC-068.
- **FR1.2:** **Library glance** is a **collapsible** region **above** the Library’s
  feed/episode layout; **default state = expanded**.
- **FR1.3:** Library glance is **fixed to the last rolling 24 hours** (calendar), not the
  Digest tab’s selected window.
- **FR1.4:** A control (e.g. **“Open full Digest”**) switches the main tab to **Digest**
  without losing corpus path context.

### FR2 — Time windows (calendar-only MVP)

- **FR2.1:** Digest tab supports at minimum **24h** and **7d** presets plus optional
  **`since YYYY-MM-DD`** alignment with Library filtering conventions (RFC-068).
- **FR2.2:** No server or client persistence of “last seen” for MVP.

### FR3 — Feed diversity

- **FR3.1:** Both surfaces apply a **documented diversity policy** so the visible set is
  not dominated by a single `feed_id` (exact algorithm in RFC-068; e.g. cap per feed after
  recency sort).
- **FR3.2:** Library glance shows a **small cap** (RFC-068 suggests order of **5–8**
  rows); Digest tab shows a **larger cap** appropriate to the window (RFC-068 specifies
  defaults and maxima).

### FR4 — Metadata layer (L0)

- **FR4.1:** Each digest row includes identifiers needed for Library/detail handoff:
  `metadata_relative_path`, `feed_id`, `episode_title`, `publish_date` when parseable.
- **FR4.2:** Show **summary title** and/or **first one or two bullets** when present
  (truncate per UXS-002 density rules).
- **FR4.3 (visual metadata):** When the corpus catalog exposes optional **feed/episode
  image URLs**, optional **verified local artwork relpaths** (`feed_image_local_relpath`,
  `episode_image_local_relpath` — same semantics as RFC-067 list rows), **duration**, and
  **episode number** (RFC-067 / RFC-068), digest **rows** and **topic hits** that resolve to
  a catalog row **include** the same optional fields when present; the **Digest** tab and
  **Library glance** render compact thumbnails ( **`PodcastCover`** prefers local binary URL
  when `corpusPath` + local path are set, else hotlinked URL) and badges **degraded** when
  data is absent (placeholders).

### FR5 — Semantic layer (L1) — Digest tab MVP

- **FR5.1:** **Global fixed topics** (strings) live in **versioned** server configuration
  (RFC-068 path); this PRD does not mandate default list contents.
- **FR5.2:** For each topic with **at least one hit** in the **current Digest window**,
  show a **topic band** (heading + episode rows or search handoff). Topics with **zero**
  hits are **hidden** for MVP (RFC-068 may allow muted placeholders later).
- **FR5.3:** Topic hits use **existing semantic search** infrastructure (PRD-021 /
  RFC-061) with **strict scope** (corpus path + time window) and **timeouts** per RFC-068.
- **FR5.4:** **Library glance** is **not required** to run per-topic search in MVP **if**
  latency budget would duplicate work; it **must** offer a path to the Digest tab for
  topic exploration (FR1.4). If a **single batched** digest API serves both without
  violating budgets, RFC-068 may promote L1 to the glance in a minor phase.

### FR6 — GI/KG layer (L2)

- **FR6.1:** Every row exposes **`has_gi`**, **`has_kg`**, and relative paths consistent
  with Library detail (RFC-067).
- **FR6.2:** **Optional** `gi_node_count` / `kg_node_count` (and similar) **only if**
  RFC-068 confirms a **cheap** extraction path; otherwise omit in MVP.

### FR7 — Handoffs

- **FR7.1:** **Open GI / Open KG** (or combined) match Library behavior: load artifacts,
  switch to **Graph** tab.
- **FR7.2:** **Search in corpus** sets Search panel context (feed, query) per RFC-067
  patterns; topic bands may pre-fill **topic query** or linked episode context per
  RFC-068.

## Layer model (complexity by surface)

| Layer | Description                         | Digest tab (MVP)      | Library glance (MVP)   |
| ----- | ----------------------------------- | --------------------- | ------------------------ |
| **L0** | Metadata slice + diversity + bullets | Required              | Required                 |
| **L1** | Semantic topic bands              | Required              | Optional (path to tab)   |
| **L2** | GI/KG badges; optional counts     | Required (badges)     | Required (badges)        |

## Success Metrics

- A user reaches **Graph** or **Search** from a digest row in **≤ 3 clicks** under normal
  conditions (same bar as PRD-022).
- Digest tab **initial load** meets the **latency budget** in RFC-068 on a reference corpus
  size (or degrades gracefully with partial results / explicit timeouts).
- Operators can **orient** to new episodes **without** full Library paging for daily
  checks (qualitative feedback optional).

## Dependencies

- [PRD-022: Corpus Library & Episode Browser](PRD-022-corpus-library-episode-browser.md)
- [RFC-067: Corpus Library — Catalog API & Viewer](../rfc/RFC-067-corpus-library-api-viewer.md)
- [PRD-021: Semantic Corpus Search](PRD-021-semantic-corpus-search.md)
- [RFC-061: Semantic Corpus Search](../rfc/RFC-061-semantic-corpus-search.md)
- [PRD-017: Grounded Insight Layer](PRD-017-grounded-insight-layer.md), [PRD-019: Knowledge
  Graph Layer](PRD-019-knowledge-graph-layer.md)
- [RFC-062: GI/KG Viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- [UXS-002: Corpus Digest](../uxs/UXS-002-corpus-digest.md)
- [UXS-001: GI/KG Viewer](../uxs/UXS-001-gi-kg-viewer.md)

## Constraints & Assumptions

**Constraints:**

- **Local tool posture:** Same trust model as Library; no auth/multi-tenant requirements.
- **Path safety:** All paths resolved under corpus root per RFC-067 / artifacts routes.

**Assumptions:**

- Metadata discovery and summary shapes match PRD-004 / PRD-005 and RFC-067.
- Semantic index may be **absent**; Digest **degrades** (L1 hidden or message) without
  breaking L0/L2.

## Design Considerations

### Digest tab default window

- **Decision:** Default **7d** on the Digest tab; Library glance remains **24h** only.
- **Rationale:** Aligns “full digest” with weekly review while keeping Library glance
  minimal.

### Global fixed topics

- **Decision:** Single **install-wide** default list for MVP (PR-reviewed).
- **Rationale:** Simplest ops and docs; Phase 2 may add per-corpus overrides.

## Related Work

- [RFC-068: Corpus Digest — API & Viewer](../rfc/RFC-068-corpus-digest-api-viewer.md)
- [UXS-002: Corpus Digest](../uxs/UXS-002-corpus-digest.md)
- [UXS-001](../uxs/UXS-001-gi-kg-viewer.md)

## Release Checklist

- [x] PRD reviewed
- [x] RFC-068 completed (implementation shipped)
- [x] UXS-002 updated (Digest + glance + health discovery)
- [x] Viewer implementation + tests per RFC-068 (unit, integration, Vitest, Playwright `digest.spec.ts`)
- [x] E2E surface map updated for new tab/region
