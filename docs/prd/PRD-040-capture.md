# PRD-040: Capture (highlights + notes)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.7 (Phase P2)
- **Parent PRD**: `docs/prd/PRD-035-learning-platform.md`
- **Depends on**: PRD-036 (per-user store), PRD-039 (capture entry points)
- **Feeds**: PRD-041 (Consolidation)

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
- Not export/integration (Readwise/Obsidian/etc.) in v2.7 — candidate follow-up.
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
- **FR3.1**: `highlight` rows: `id, user, episode_slug, kind(span|moment|insight), start_ms, end_ms,
  char_start, char_end, segment_ids[], quote_text, speaker?, source_insight_id?, color?, created_at`.
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

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/user/highlights` | All highlights (filter: `podcast`, `topic`, `color`) |
| `GET` | `/api/episodes/{slug}/highlights` | Highlights for one episode |
| `POST` | `/api/episodes/{slug}/highlights` | Create a highlight |
| `PATCH`/`DELETE` | `/api/user/highlights/{id}` | Edit/remove a highlight |
| `POST` | `/api/user/notes` | Create a note (target + target_id) |
| `PATCH`/`DELETE` | `/api/user/notes/{id}` | Edit/remove a note |

## Success Metrics

- A highlight is captured in ≤1 interaction during playback and appears immediately in "My highlights".
- Every highlight jumps back to within 0.5s of its original moment.
- Highlights survive across sessions and devices (per-user persistence).
- A saved insight retains full grounding (quote + speaker + timestamp).

## Dependencies

- PRD-036 (per-user store, grounding data from artifacts).
- PRD-039 (capture entry points in the player/transcript/insight cards).

## Open Questions

- Export to external tools (Readwise/Obsidian/Markdown) — follow-up PRD?
- Tag taxonomy: free tags vs. fixed colours/labels only in v2.7.

## References

- `docs/prd/PRD-035-learning-platform.md`, `PRD-039-player.md`, `PRD-041-consolidation.md`
- `docs/rfc/RFC-049-grounded-insight-layer-core.md`
