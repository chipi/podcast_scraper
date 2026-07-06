# PRD-037: Discovery

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.7 (Phase P1)
- **Parent PRD**: `docs/prd/PRD-035-learning-platform.md`
- **Depends on**: PRD-036 (identity, per-user library, scrape-on-demand)
- **Downstream**: PRD-038 (Catalog) — user lands here after adding a podcast
- **Supersedes**: `PRD-027-platform-discovery.md` (deleted draft)

---

## Summary

Discovery is the user's entry point for adding content. It provides search over an open podcast
catalog (Podcast Index to start), a personal library view, and scrape-on-demand. It is the top of the
`Discover → Catalog → Player` loop. Sources are pluggable via a `DiscoverySource` abstraction so new
sources (OPML, manual RSS) can be added without touching core logic.

> **Phasing note (2026-07-06).** This PRD is delivered in two phases, **curated first, user
> self-serve second** — not all at once in v2.7. **Phase 1 (now):** the corpus is operator-curated;
> the shared **ingestion primitive** (ingest one feed/episode → job → deduped merge, #1069) grows
> it, and the search / library / personalization surfaces run over what's ingested (already shipped:
> entity search #1097, ranking #1098 / #1139, home + search #1090). **Phase 2 (later epic, gated on
> real persistence + the PWA + real users):** the consumer self-serve surface — Podcast Index
> `DiscoverySource` (FR1), user-triggered scrape (FR2 / FR4), and the guardrail implementations
> (rate / quota / cost / abuse). The FRs below describe the full Phase-2 target; the
> `DiscoverySource` seam (`app_content_source`) is scaffolded now so Phase 2 adds no API reshape.
> See `docs/wip/player/1069-SCRAPE-ON-DEMAND-SCOPE-ANALYSIS.md` for the decision + build plan.

## Background & Context

- Without discovery, users must hand-supply RSS URLs — technical-only and hostile. Podcast Index is a
  free, open catalog (~4M shows) with artwork, metadata, and RSS URLs — the natural OSS fit.
- Library management (which shows a user follows, which episodes are ready vs requestable) lives here.
- Per PRD-035 Principle 3, the library is a *per-user* subscription overlay; the underlying episode
  artifacts are shared. Adding a show another user already added triggers no new scrape.

## Goals

- Find any podcast by name/keyword without knowing its RSS URL.
- Add a podcast to the user's personal library in one action.
- Show the user's library with per-podcast episode availability.
- Request scrape of specific episodes or full feeds, globally deduped.
- Establish `DiscoverySource` as the canonical, pluggable discovery interface.

## Non-Goals

- Not a browse-by-category directory (search only in v2.7).
- Not OPML import or manual RSS entry (later `DiscoverySource` implementations).
- Not editorial curation. Personalised recommendations come from Consolidation (PRD-041), not here.
- Not social discovery.

## Personas

Primary: **casual listener** (find a show fast) and **active learner** (add the shows they learn from).

## User Stories

- _As a new user, I can search a podcast by name and add it to my library in one tap._
- _As a returning user, I can see my library and which episodes are ready vs need scraping._
- _As an active listener, I can request a specific episode (or whole feed) be scraped and watch progress._

## Functional Requirements

### FR1: Search

- **FR1.1**: A prominent search input calls the active `DiscoverySource.search()`.
- **FR1.2**: Results show artwork, name, publisher, episode count, short description, and an
  "Add to Library" action.
- **FR1.3**: Shows already in the user's library render "In Library" + a link to their Catalog view.
- **FR1.3a**: Results may surface **related topics** from topic clustering (RFC-075) so users discover
  adjacent shows/episodes by theme. This is discovery, not personalisation (which stays in PRD-041).
- **FR1.4**: Results paginate via "Load more" (no full reload).
- **FR1.5–1.7**: Empty, no-results, and error states are non-blocking; the existing library stays usable.

### FR2: Library

- **FR2.1**: The user's library lists added podcasts with artwork, name, and availability summary
  (e.g. "12 ready, 3 pending").
- **FR2.2**: Each row links to the podcast's Catalog view (PRD-038).
- **FR2.3**: "Remove from library" (confirmed) removes only the user's subscription, never shared artifacts.
- **FR2.4**: Empty-library state guides the user to search.

### FR3: Adding a podcast

- **FR3.1**: "Add to Library" subscribes the user and triggers an initial feed scan to populate episodes.
- **FR3.2**: After adding, the user sees the podcast's Catalog view; already-scraped episodes show Ready,
  others show Request.
- **FR3.3**: Adding a show already in the shared catalog adds only the subscription — no new scrape.

### FR4: Scrape-on-demand

- **FR4.1**: "Request" on an unprocessed episode submits `POST /api/scrape` (PRD-036 FR5).
- **FR4.2**: Requests are globally deduped — an in-progress job is shown rather than re-queued.
- **FR4.3**: "Request all" queues all unprocessed episodes of a podcast.
- **FR4.4**: Progress shows pipeline stage (Transcribing…, Extracting insights…) via
  `GET /api/scrape/status/{job_id}`.
- **FR4.5**: On completion the episode flips to Ready without a manual refresh.

## DiscoverySource abstraction

```python
class DiscoverySource:
    id: str            # "podcast_index"
    display_name: str  # "Podcast Index"
    def search(self, query: str, page: int) -> list[PodcastResult]: ...
    def resolve_feed(self, result: PodcastResult) -> str: ...
```

A source selector appears only when >1 source is configured. v2.7 ships Podcast Index only.

## API summary

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/discovery/search` | Proxy to active source (`q`, `page`); API key stays server-side |
| `GET` | `/api/user/library` | User's library + availability counts |
| `POST` | `/api/user/library` | Add podcast by feed URL |
| `DELETE` | `/api/user/library/{podcast_id}` | Remove subscription |
| `POST` | `/api/scrape` | Request scrape (PRD-036) |
| `GET` | `/api/scrape/status/{job_id}` | Progress (PRD-036) |

## Success Metrics

- Search a major show by name → accurate results with artwork/description.
- Add from search → one tap → land in Catalog.
- A show already in the shared catalog is available instantly (no duplicate scrape).
- Scrape progress updates inline without a refresh.
- Adding a second `DiscoverySource` requires only a new implementation — no API/UI routing changes.
- Podcast Index API key never appears in frontend network traffic.

## Dependencies

- PRD-036 (identity, per-user library, scrape API).
- Podcast Index API key (`.env`, server-side only).
- RFC-065 status.json for progress.

## Open Questions

- Per-user scrape quota / abuse controls (PRD-035 OQ4).

## References

- `docs/prd/PRD-035-learning-platform.md`, `PRD-036-foundation-identity.md`, `PRD-038-catalog.md`
- <https://podcastindex.org/> , <https://podcastindex-org.github.io/docs-api/>
- `docs/rfc/RFC-065-agent-observable-instrumentation.md`
