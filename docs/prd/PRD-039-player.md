# PRD-039: Player

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.7 (Phase P1)
- **Parent PRD**: `docs/prd/PRD-035-learning-platform.md`
- **Depends on**: PRD-036 (segments, insights, entities, enclosure, playback state)
- **Upstream**: PRD-038 (Catalog) · **Feeds**: PRD-040 (Capture), PRD-041 (Consolidation)
- **Related UX spec**: `docs/uxs/UXS-011-consumer-learning-app.md` (Editorial Bold; Player visual contract)
- **Related RFC**: `docs/rfc/RFC-099-learning-platform-consumer-client.md` (client behaviour, MVP slice)
- **Supersedes**: `PRD-030-platform-player.md` (deleted draft)

---

## Summary

The Player is the primary value surface. It plays podcast audio **streamed from the original host**
with real-time transcript synchronisation — the current spoken segment is highlighted and the
transcript auto-scrolls; any line is tappable to seek. A collapsible **Knowledge Panel** surfaces the
pipeline's artifacts inline (summary, entities, grounded insights, in-episode grounded search). A **queue** lets the
user line up episodes — Spotify-grade — and queuing an unprocessed episode kicks off scrape-on-demand
so it's ready by the time it's reached.

## Background & Context

- Transcript-synced playback is the clearest differentiator from standard players: only a player backed
  by a structured pipeline can highlight the exact sentence being spoken and surface who-said-what,
  grounded in verbatim quotes + timestamps.
- The Knowledge Panel makes pipeline output visible without leaving the player. It is minimal and
  collapsed by default — one tap to reveal.
- Per Principle 4, audio comes from the original enclosure URL (PRD-036 FR4); we never serve a copy.

## Goals

1. Spotify-grade playback: queue, controls, speed, resume — fast and polished on mobile.
2. Transcript-synced playback as the core, unmistakable differentiator.
3. Surface insights, entities, and summary in the Knowledge Panel without cluttering playback.
4. Grounded in-episode search (retrieval over transcript + insights; no request-time LLM, D6).
5. Persist playback position for resume across sessions and devices.
6. Degrade gracefully: full audio + transcript even without GIL/KG artifacts.

## Non-Goals

- Not background/lock-screen audio with native media-session in v2.7 (north-star).
- Not a full GI/KG viewer — the Knowledge Panel is a minimal inline surface; deep graph exploration
  stays in the operator viewer (RFC-062).
- Not social (no comments, clip-sharing, reactions).
- Not transcript editing/correction.
- Not chunked playback before processing completes (deferred advanced mode).

> **Changed from the old draft**: a **queue is now in scope** (Spotify vibe), and audio is **bridged
> from the original host**, never streamed from a stored copy.

## Personas

**Active listener** (follow along, jump back), **researcher** (grounded in-episode search, jump-to-moment),
**casual listener** (queue and play; never opens the panel).

## User Stories

- _As a listener, I can queue several episodes and have them play in order._
- _As a listener, I can follow the transcript as it plays and tap any line to jump there._
- _As a curious listener, I can see the entities and insights for this episode without leaving the player._
- _As a researcher, I can search within the episode and jump to the exact grounded passages that answer my question._
- _As a returning listener, I can resume where I left off — on any device._

## Functional Requirements

### FR1: Layout

- **FR1.1**: Full-viewport on mobile; two-column on desktop (transcript main + collapsible Knowledge Panel).
- **FR1.2**: Episode title + podcast name at the top.
- **FR1.3**: Back navigation returns to Catalog; playback continues where technically feasible
  (pause-on-navigate acceptable for v2.7).

### FR2: Queue & playback controls

- **FR2.1**: Standard controls: play/pause, scrub with elapsed/total, skip-back 15s, skip-forward 30s.
- **FR2.2**: Speed selector: 0.75× / 1× / 1.25× / 1.5× / 2×.
- **FR2.3**: **Queue**: add/remove/reorder episodes; auto-advance to the next on completion; "play next"
  and "add to queue" actions from Catalog cards (PRD-038).
- **FR2.4**: Queuing an unprocessed episode triggers scrape-on-demand (PRD-036 FR5) and shows its
  progress in the queue; it becomes playable when Ready.
- **FR2.5**: Playback position auto-saves to `PUT /api/app/playback/{slug}` every ~10s and on pause;
  restored via `GET` on open.
- **FR2.6**: A "Resume from X:XX" prompt appears for episodes with a saved position.

### FR3: Transcript panel

- **FR3.1**: Full transcript as a scrollable list of `segments.json` segments.
- **FR3.2**: The playing segment is highlighted and auto-scrolled into view; auto-scroll pauses on manual
  scroll and re-enables after ~5s idle.
- **FR3.3**: Tapping a segment seeks audio to its start.
- **FR3.4**: Segment timestamps shown on hover/tap (desktop) or always (mobile), muted.
- **FR3.5**: Speaker labels per segment when diarization present — canonical `person:{slug}` (RFC-072)
  when available, else raw label.
- **FR3.6**: No transcript → panel offers "Request processing" (audio-only mode still plays).
- **FR3.7** _(retrospective, shipped #1091)_: **Grounded-quote highlighting** — transcript segments
  backing a grounded insight (matched by timeline) get a `●` marker + underline; tapping one
  **opens the Insights panel and centre-scrolls to that insight** (the transcript↔insight bridge).
- **FR3.8** _(retrospective, shipped #1091)_: **Manual sync nudge** — a `Sync −/+` control lets the
  listener shift the transcript↔audio alignment to compensate for ad-insertion drift in the bridged
  stream (the played audio carries dynamic ads not in our transcribed copy). Persisted per episode.

### FR4: Insights panel (collapsible; collapsed on mobile, open on wide desktop)

> **Retrospective (shipped #1091):** the panel is titled **"Insights"** in the UI (matching the dock
> button + cards). Topics and People are **merged** into one compact, expandable "Topics & People"
> row; tapping a chip **searches the corpus** for that term (the originally-specified person→insight
> filter was dropped).

- **FR4.1 Summary**: 2–4 sentence episode summary, shown first when available.
- **FR4.2 Topics & People** _(merged)_: KG topics + persons as chips in one compact, expandable row
  (`+N …` to reveal the rest). Tapping a chip navigates to **corpus search** for that term. _(Epic 3
  layers cluster-first ordering + entity cards on top — PRD-043.)_
- **FR4.3 Insights**: grounded GIL insights as cards — insight text, supporting quote (verbatim),
  speaker if attributed, and a timestamp link that seeks on tap. A `●` **grounded marker**
  distinguishes insights with a timestamped quote from ungrounded claims. Max 5; "Show all" expands.
- **FR4.5 Ask / find in this episode**: a search input scoped to this episode. On submit, the backend runs
  hybrid retrieval (RFC-090) over the episode's segments + insights and returns the most relevant **grounded
  passages** — verbatim text, speaker, and a timestamp link that seeks on tap. **No request-time LLM** (D6):
  results are extractive grounded matches, not generated prose — nothing to hallucinate, no provider or
  credentials in the player. A generative-answer layer on top is a parked future option.
- **FR4.6 Graceful degradation**: each section is independently optional (no summary→hidden,
  no KG→Topics/Persons hidden, no GIL→Insights hidden, no transcript→in-episode search hidden). All absent →
  panel explains intelligence requires GIL/KG processing.

### FR5: Capture entry points (→ PRD-040)

- **FR5.1**: A one-tap "highlight current moment" control during playback, and a highlight action on any
  transcript segment and any insight card. (Storage/behaviour defined in PRD-040.)

## Data sources

| Panel section | Source | Required |
| --- | --- | --- |
| Transcript | `segments.json` (PRD-036 FR3.3) | Yes |
| Audio | original enclosure URL (PRD-036 FR4) | Yes |
| Summary | summary artifact / metadata | No |
| Topics / Persons | `kg.json` (+ `bridge.json`) | No |
| Insights | `gi.json` (Insight + Quote + ABOUT) | No |
| In-episode search | episode segments + insights (hybrid search index) | No |

## API summary

All endpoints live under the **consumer namespace `/api/app/*`** (RFC-098), isolated from the operator
API. The paths below are the **shipped Epic-1 surface** — the Player MVP is fully served (no net-new
server endpoints required).

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/app/episodes/{slug}/segments` | Transcript sync (PRD-036) |
| `GET` | `/api/app/episodes/{slug}/insights` | Knowledge Panel insights |
| `GET` | `/api/app/episodes/{slug}/entities` | Knowledge Panel entities |
| `GET` | `/api/app/episodes/{slug}/audio-source` | Original enclosure URL (PRD-036, RFC-100) |
| `GET`/`PUT` | `/api/app/playback/{slug}` | Resume position |
| `GET`/`PUT` | `/api/app/queue` | Queue read/update |
| `GET` | `/api/app/episodes/{slug}/search` | Episode-scoped grounded retrieval (no LLM) |

## Success Metrics

1. The playing segment is highlighted and in view at all times with no perceptible lag.
2. Tapping a line seeks within 0.5s of the segment start.
3. Playback position restores correctly on re-open, including a different device.
4. ≥1 grounded insight with a tappable timestamp is visible for any GIL-processed episode.
5. In-episode search returns ranked grounded passages with jump-to-moment links within ~1s; no request-time LLM.
6. Player is fully functional (audio + transcript) for episodes with no GIL/KG — no broken panels.
7. Queue auto-advances; a queued unprocessed episode becomes playable after its scrape completes.

## Dependencies

- PRD-036 (segments, insights, entities, enclosure, playback/queue state, scrape).
- The hybrid search index (RFC-090) over the corpus; in-episode search filters it to the episode.

## Open Questions

- Native media-session / background audio (north-star; depends on app shell decision PRD-035 D1).
- Whether/when to add an optional generative-answer layer on top of retrieval (parked per D6; would
  reintroduce a request-time LLM and the CI-stub requirement).

## References

- `docs/prd/PRD-035-learning-platform.md`, `PRD-038-catalog.md`, `PRD-040-capture.md`
- `docs/rfc/RFC-049-grounded-insight-layer-core.md`, `RFC-055-knowledge-graph-layer-core.md`,
  `RFC-072-canonical-identity-layer-cross-layer-bridge.md`
