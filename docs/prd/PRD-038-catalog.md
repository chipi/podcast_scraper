# PRD-038: Catalog

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.7 (Phase P1)
- **Parent PRD**: `docs/prd/PRD-035-learning-platform.md`
- **Depends on**: PRD-036 (read API), PRD-037 (library, scrape)
- **Downstream**: PRD-039 (Player)
- **Supersedes**: `PRD-029-platform-catalog.md` (deleted draft)

---

## Summary

The Catalog is the user's view of available content — the bridge from Discovery to Player. It shows
which episodes are ready, pending, or unscraped, and surfaces enough per-episode context (summary,
topics, insight count, duration) to choose what to play. Two levels: the **podcast view** (one show)
and the **global catalog** (all episodes across the user's library, newest first).

## Background & Context

- After adding a show, users need to browse episodes, understand readiness, and navigate into the player.
- The Catalog is the natural home for surfacing corpus-level signals (topics, insight counts, summary
  previews) so users choose informed — before committing to playback.
- All enrichment is optional per Principle: cards degrade cleanly to core metadata when artifacts are absent.

## Goals

- Show the library's episodes with accurate Ready / Pending / Not-scraped status.
- Give enough per-episode context to choose without opening the player.
- Be the navigation hub between Discovery and Player.
- Surface scrape progress inline for pending episodes.
- Degrade gracefully when enrichment is absent.

## Non-Goals

- Not a search interface (search is its own surface; cross-corpus episode search is later).
- Not a playback surface (cards only; playback is exclusively the Player).
- Not a recommendation engine — ordering is chronological in v2.7 (personalised ordering is PRD-041).
- Not a download manager (bridge-only; no local downloads).

## Personas

**Casual listener** (what's ready to play?) and **active learner** (which episode is worth my time?).

## User Stories

- _As a listener, I can see all episodes of a show I follow with clear ready/pending status._
- _As a listener, I can judge an episode (summary, topics, duration) without opening it._
- _As a browser, I can see recent episodes across my whole library to find something to play._

## Functional Requirements

### FR1: Global catalog view

- **FR1.1**: A Home / All-Episodes view across all library podcasts, publish-date descending.
- **FR1.2**: Each card: artwork, podcast name, episode title, publish date, duration, status badge.
- **FR1.3**: Ready → links to Player; Not-scraped → inline "Request"; Pending → inline progress.
- **FR1.4**: Pending progress sourced from RFC-065 status.json via `GET /api/scrape/status/{job_id}`.
- **FR1.5**: Paginated / infinite scroll, 20 per page.

### FR2: Podcast view

- **FR2.1**: Tapping a library podcast shows its header (artwork, name, publisher, description) + episode list.
- **FR2.2**: Episodes sorted publish-date descending, same card format as FR1.2.
- **FR2.3**: "Request all" queues all unprocessed episodes (PRD-037 FR4.3).
- **FR2.4**: Header shows totals (e.g. "48 episodes · 12 ready · 2 pending").

### FR3: Episode card — enriched state

- **FR3.1**: 1–2 sentence summary preview from the summary artifact.
- **FR3.2**: Up to 5 topic chips (KG), truncated with "+N more".
- **FR3.3**: Speaker count when diarization present (e.g. "2 speakers").
- **FR3.4**: GIL insight count when present (e.g. "7 insights") as a depth signal.

### FR4: Episode card — degraded state

- **FR4.1**: Core-only cards (title, date, duration, status) omit absent fields — no broken/empty panels.
- **FR4.2**: Audio-available-but-no-transcript shows "Transcript pending"; still playable (audio-only).

### FR5: Navigation

- **FR5.1**: Ready card → Player (PRD-039), passing the episode slug.
- **FR5.2**: Breadcrumbs: Player → Catalog (podcast) → Discovery (library); back always available.
- **FR5.3**: From global view, tapping podcast name/artwork → that podcast's Catalog view.

## API summary

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/episodes` | Episodes across library (`page`, `status`) — PRD-036 FR3.1 |
| `GET` | `/api/podcasts/{id}/episodes` | One podcast's episodes (`page`) |
| `GET` | `/api/episodes/{slug}` | Detail + artifact flags — PRD-036 FR3.2 |

Episode summary shape (per PRD-036): `slug, title, podcast_name, publish_date, duration_seconds,
status, artifacts{transcript,summary,gi,kg}, summary_preview, topics[], speaker_count, insight_count`.
Artifact flags drive graceful degradation.

## Success Metrics

- Library → a podcast's episode list in one tap.
- Status always accurate; pending → ready flips without a full reload.
- Cards show enrichment when present and degrade cleanly when absent.
- Request a scrape directly from a card without navigating away.
- Catalog → Player → back preserves scroll position.

## Dependencies

- PRD-036 (read API, status), PRD-037 (library, scrape).
- RFC-065 status.json.

## Open Questions

- Cross-corpus episode search surface (deferred — separate PRD).

## References

- `docs/prd/PRD-035-learning-platform.md`, `PRD-037-discovery.md`, `PRD-039-player.md`
- `docs/rfc/RFC-065-agent-observable-instrumentation.md`
