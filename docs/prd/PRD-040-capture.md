# PRD-040: Capture (highlights + notes)

- **Status**: Shipped — v2.7 / Epic P2 (#1112; children #1114–#1119)
- **Authors**: Marko
- **Target Release**: v2.7 (Phase P2)
- **Parent PRD**: `docs/prd/PRD-035-learning-platform.md`
- **Depends on**: PRD-036 / RFC-098 §3 (per-user files), PRD-039 (capture entry points)
- **Feeds**: PRD-041 (Consolidation)

> **As-shipped note (v2.7).** This PRD is reconciled with what landed. The data model (FR3) shipped
> as specified. Deltas to call out, honestly: transcript capture shipped at **segment (line)
> granularity** — saving a transcript line as a span with `char_start/end` covering the whole
> segment — not yet Kindle-style **sub-segment character selection** (FR1.2); the highlight `color`
> field exists in the model + API but **no colour picker is surfaced yet** (FR1.4), so the global
> view's colour/topic **filters** (FR4.2) are not yet built; and **Markdown export shipped** (it had
> been a v2.7 non-goal — see the revised Non-Goals). Everything else in FR1–FR4 is live. See
> **§"As shipped"** for the FR→code map.

---

## Summary

Capture turns a moment of listening into a durable, grounded artifact. While playing — or reading the
transcript — a user can **highlight** a transcript span or an insight (Kindle-style) and attach a
**note**. Every capture is grounded: it carries the episode slug, timestamp, and transcript character
offsets, so it can be replayed, cited, and later woven into the user's personal knowledge corpus
(PRD-041). Capture must be effortless — one interaction in the common case — or it won't happen.

## Background & Context

- Listening alone doesn't make knowledge stick (PRD-035 thesis). Capture is the first half of the
  retention loop: marking what mattered *in the moment*. Consolidation (PRD-041) is the second half.
- The pipeline already grounds insights and quotes to exact offsets/timestamps (RFC-049). Capture reuses
  that grounding so a user highlight is as precise and replayable as a pipeline quote.
- Captures are per-user overlay (Principle 3): private, attached to shared episode artifacts.

## Goals

- Capture a highlight in ≤1 interaction during playback ("mark this moment").
- Highlight arbitrary transcript spans (Kindle-style selection) and whole insights.
- Attach freeform notes to an episode, a highlight, or an insight.
- Ground every capture (slug + timestamp + char offsets + optional speaker) for replay and citation.
- Let users review and manage their captures per-episode and globally.

## Non-Goals

- Not collaborative/shared highlights (no social in v2.7).
- Not rich-text/markdown editing beyond plain notes (keep it frictionless; richer formats later).
- Not third-party integration (Readwise/Obsidian/etc.) in v2.7 — candidate follow-up. **(Plain
  Markdown export of all highlights + notes *did* ship — see API summary; the integrations remain out.)**
- Not auto-highlighting — capture is user-driven (pipeline insights are a *separate*, suggested layer).

## Personas

**Active learner** (mark what resonates, revisit later) and **researcher** (capture citable evidence).

## User Stories

- _As a listener, I can tap once to highlight the moment I'm hearing right now._
- _As a reader, I can select a transcript passage and save it as a highlight._
- _As a learner, I can save a pipeline insight to my highlights with one tap._
- _As a note-taker, I can attach a thought to a highlight or to the whole episode._
- _As a returning user, I can see all my highlights for an episode, and across everything, and jump back
  to the exact audio moment of any of them._

## Functional Requirements

### FR1: Highlighting

- **FR1.1**: During playback, a single control captures the current moment as a highlight anchored to the
  active segment (slug + timestamp + segment id + offsets).
- **FR1.2**: In the transcript, selecting a span (one or more segments) creates a highlight over the exact
  character range, carrying start/end timestamps.
- **FR1.3**: An insight card (PRD-039 FR4.3) has a "save to highlights" action; the saved highlight retains
  the insight→quote grounding (text, quote, speaker, timestamp).
- **FR1.4**: Highlights have an optional colour/label (small fixed palette) for lightweight categorisation.

### FR2: Notes

- **FR2.1**: A note (plain text) can attach to a highlight, an insight, or the episode as a whole.
- **FR2.2**: Notes are editable and deletable by their owner.

### FR3: Grounding & data model

- **FR3.1**: `highlight` records: `id, user, episode_slug, kind(span|moment|insight), start_ms, end_ms,
  char_start, char_end, segment_ids[], quote_text, speaker?, source_insight_id?, color?, created_at`.
- **FR3.1a — Survive re-transcription:** timestamps (`start_ms/end_ms`) are the stable anchor;
  `segment_ids` and char offsets may shift on re-scrape, so a highlight re-anchors by timestamp on read
  (re-locating the nearest segment), retaining `quote_text` for display + drift verification. A re-scrape
  never silently drops a highlight (see RFC-098 §7).
- **FR3.2**: `note` rows: `id, user, target(highlight|insight|episode), target_id, text, created_at,
  updated_at`.
- **FR3.3**: Both are per-user overlay rows (PRD-036 FR2); authz on every access.

### FR4: Review & management

- **FR4.1**: Per-episode highlights list, each with a jump-to-moment link (seeks the player).
- **FR4.2**: A global "My highlights" view across all episodes, filterable by podcast/topic/colour and
  sortable by recency.
- **FR4.3**: Edit/delete any highlight or note.
- **FR4.4**: Highlights and notes are the raw material consumed by Consolidation (PRD-041).

## API summary

All routes are auth-gated and live under the consumer `/api/app/*` namespace (RFC-098), **not** the
`/api/user` · `/api/episodes` paths this PRD originally sketched. Episode scoping is a `?episode=`
query parameter, not a path segment. As shipped (`src/podcast_scraper/server/routes/app_capture.py`):

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/app/highlights` | All highlights; `?episode=<slug>` scopes to one episode |
| `POST` | `/api/app/highlights` | Create a highlight (201; route mints `id` + `created_at`) |
| `PATCH` | `/api/app/highlights/{id}` | Edit `color` / `quote_text` (404 if absent) |
| `DELETE` | `/api/app/highlights/{id}` | Remove a highlight; returns the remaining list |
| `GET` | `/api/app/notes` | All notes; `?target=&target_id=` scopes them |
| `POST` | `/api/app/notes` | Create a note (target + target_id; `text` min length 1) |
| `PATCH` | `/api/app/notes/{id}` | Edit a note's text (404 if absent) |
| `DELETE` | `/api/app/notes/{id}` | Remove a note; returns the remaining list |
| `GET` | `/api/app/highlights/export.md` | Markdown export of all highlights + attached notes |

## As shipped (v2.7 / Epic P2)

| FR | Shipped as | Where |
| --- | --- | --- |
| FR1.1 moment | One-tap "mark this moment" in the player hero (tags the active speaker) | `app/src/views/PlayerView.vue`, `#1116` |
| FR1.2 span | Save a transcript **line** as a span (segment-granular; char range = whole segment). Sub-segment character selection is a future refinement. | `app/src/components/TranscriptList.vue` (`canCapture`), `#1116` |
| FR1.3 insight | "Save to highlights" on each Knowledge-panel insight card (keeps `source_insight_id` grounding) | `app/src/components/KnowledgePanel.vue`, `#1116` |
| FR1.4 colour | `color` field exists in the model + `PATCH` API; **no colour picker surfaced yet** | model only |
| FR2 notes | Add / edit / delete plain-text notes per highlight in the Library view | `app/src/views/HighlightsView.vue`, `#1117` |
| FR3 grounding | `highlight` + `note` records exactly per FR3.1/FR3.2; per-user JSON files | `src/podcast_scraper/server/app_user_state.py`, `#1114` |
| FR3.1a re-anchor | `reanchor_highlight()` re-locates positional fields by timestamp; a drifted span is flagged (`anchor_status`), never dropped (RFC-098 §7) | `app_user_state.py`, `#1114` |
| FR4.1 per-episode | Highlights grouped by episode with jump-to-moment (`?t=`) | `HighlightsView.vue`, `#1117` |
| FR4.2 global view | Global Library "Highlights" tab (grouped by episode). **Podcast/topic/colour filters not yet surfaced.** | `LibraryView.vue`, `#1117` |
| FR4.3 edit/delete | Delete highlights; add/edit/delete notes | `#1117` |
| FR4.4 → Consolidation | Highlights + notes are the per-user corpus P3 (PRD-041) reads | feeds PRD-041 |
| Export | `GET /highlights/export.md` + a Library "Export Markdown" link | `app_capture_export.py`, `#1115` |

**Tests:** unit (store + re-anchor + Markdown renderer), integration (route CRUD + export over a
fixture corpus), component (capture surfaces + Highlights view), and an e2e covering the full
listen→capture→review loop on the committed validation corpus (`#1114`–`#1118`).

## Success Metrics

- A highlight is captured in ≤1 interaction during playback and appears immediately in "My highlights".
- Every highlight jumps back to within 0.5s of its original moment.
- Highlights survive across sessions and devices (per-user persistence).
- A saved insight retains full grounding (quote + speaker + timestamp).

## Dependencies

- PRD-036 (per-user store, grounding data from artifacts).
- PRD-039 (capture entry points in the player/transcript/insight cards).

## Open Questions

- ~~Export to external tools (Readwise/Obsidian/Markdown)~~ — **Markdown export shipped** in v2.7;
  third-party integrations (Readwise/Obsidian) remain a follow-up PRD.
- Tag taxonomy: free tags vs. fixed colours/labels. The `color` field exists but no picker shipped;
  surfacing colour (+ the global filters that depend on it) is the open v2.7+ refinement.
- Sub-segment character-range selection in the transcript (FR1.2) — shipped at line granularity;
  Kindle-style mid-segment selection is the remaining refinement.

## References

- `docs/prd/PRD-035-learning-platform.md`, `PRD-039-player.md`, `PRD-041-consolidation.md`
- `docs/rfc/RFC-049-grounded-insight-layer-core.md`
