# RFC-126: Discover Rails, Reasons, Paging and Reels

- **Status**: Implemented in part (branch `feat/player-1.0.4`; epic #2306)
- **Authors**: Marko
- **Stakeholders**: player (Discover, Explore, the player store), server (`/api/app/discover/*`),
  analytics (Umami), the validation corpus
- **Related PRDs**:
  - `docs/prd/PRD-048-discover-and-explore.md` — the requirements this RFC designs
- **Related ADRs**:
  - `docs/adr/ADR-162-mine-means-the-listeners-own-world.md` — one definition of Mine (amended)
  - `docs/adr/ADR-163-the-jobs-of-discover.md` — the jobs, the extractive reason, cold start, the
    reel (amended)
  - `docs/adr/ADR-164-discover-is-a-taste.md` — server-paged rails, one card per episode
  - `docs/adr/ADR-165-validation-corpus-grounds-every-insight.md` — the fixture these are proved on
- **Working plan**: `docs/wip/DISCOVER-ARC-PLAN-2026-10-10.md`

## Abstract

Discover is built from **rails**: ranked lists of episode cards, each card carrying an extractive
**reason**. The server owns each rail's full ranked list and the page's one-card-per-episode rule,
and pages every rail (`GET /discover/rails`, `GET /discover/rails/{rail}`). A **topic reel** stitches
the Moments picker's clips about one topic across episodes and plays them in the player store.
Everything is built from what is on disk: no LLM, no fetch.

## Problem Statement

The previous Discover asked each rail for a fixed list in the browser, so an episode appeared on
several rails, "more" could only reveal what was already loaded, and cold start was decided in the
client from a second request. Reasons quoted the first quote of the first insight, which in real
and fixture data is often a greeting or an ad read.

## Goals

1. One request renders the whole Discover page; one card per episode across it.
2. "Show more" pages a rail from the server, consistent with the first page.
3. A reason's quote is transcript text that backs an insight.
4. "Less like this", "nothing already heard" and cold start apply on the server, on every page.
5. A topic reel plays clips across episodes without moving a resume point or counting as a listen.

## Constraints & Assumptions

- Read paths are disk-only: GI, KG, bridge, storylines, the search sidecar, user state files.
- Mine is `world_episode_set` / `personal_entity_ids` (ADR-162), never re-derived per rail.
- Pre-launch: no backward compatibility for the removed per-rail endpoints.

## Design & Implementation

### 1. The reason (`app_reasons`)

```text
Reason { kind, quote?: ReasonQuote, subject?: ReasonSubject, capture?: ReasonCapture,
         events?: int }
kind ∈ rising_topic | most_engaged | captured | thread | person | another_view
```

The server sends parts; the client writes the words (localised, no generated sentence).

**Quote choice** (`best_grounded_quote`): across one episode's insights, the quote sharing the most
content words (longer than three letters) with any insight, provided it shares at least a third of
its own (`MIN_SUPPORT`). It is credited to the insight it backs. With no backing quote anywhere, the
first insight's first timed quote is used, so no card is lost.

### 2. The rails (`app_discover_rails`)

| Rail | Builder | Ranked by |
| --- | --- | --- |
| `quotes` | `trending_episodes` | rising topics interleaved with most-engaged (distinct listeners) |
| `captured` | `captured_rail` | captures newest first, round-robin, linked by topic/person/storyline |
| `thread` | `thread_rail` | followed storylines, published after the last listen, newest first |
| `people` | `people_rail` | speakers heard in 2+ episodes, most heard first, other shows only |
| `another_view` | `another_view_rail` | the listener's top non-umbrella topics, unheard speakers |

`rail_cards(ctx, rail)` returns the full ranked list, at most `RAIL_MAX` (12):

1. the builder's list (asked for twice `RAIL_MAX`);
2. less what the listener asked to see less of (`suggest_less.json`);
3. for `quotes` under Mine with an empty world: everyone's list, `fallback = true`.

Under Mine nothing from Everyone is mixed in (ADR-163 §7, superseded 2026-10-11), and no rail offers
an episode the listener heard, captured from or saved. The quote rail takes the rising topics of
their world through any such episode about the topic (`topics_anywhere`, `exclude`); its most
engaged half stays inside their world.

Everyone's trending list is computed once per request (`RailContext.everyone`).

### 3. Paging (`ADR-164`)

```text
GET /api/app/discover/rails?scope=mine|corpus&first=2
  → { scope, rails: [ { rail, items[≤first], total, fallback } ... ] }   # page order

GET /api/app/discover/rails/{rail}?scope&offset&limit&exclude=<slug>...
  → { rail, items, total, fallback }
```

`first_pages` walks the rails in page order. Each rail's page is its list less the slugs already
shown by the rails above; a rail with nothing left is omitted. A later page passes `exclude` = the
slugs the rails above showed first, so `offset` counts within the same deduplicated list. The
contract is tested as "first page + next page == the same rail asked from the top".

Explore's Trends page the same way (2026-10-11): `GET /trending?kind&scope&window&sort&offset&limit`
→ `{items, total}`. The server ranks the whole list once (cached for Everyone), orders it by `sort`
(`score` = velocity × volume, `rising` = velocity, `trending` = volume) and slices it; the client
keeps rows and `total` together so a late answer for another scope cannot pair one list's rows with
the other's count.

### 4. Topic reels (`app_goal_reel`)

For a topic, every episode carrying it (search sidecar), the Moments candidates
(`score_candidates`: clip windows, scores) whose insight is ABOUT the topic and whose opening quote
backs the insight. Breadth first: every episode's best clip before any episode's second, at most two
per episode, clips added while they fit `minutes`. The response reports `total_seconds`, which is
what the reel actually runs.

The player store's `startTopicReel(topic, moments, resolve)` loads each clip's episode through a
resolver the shell supplies (`resolveNextUpForSlug`), then seeks. `savePosition` and the armed
listen already skip while any reel runs, so no resume point moves and no listen is logged until
"Keep listening to this episode". ✕ stops without seeking back.

**Themes and storylines** (2026-10-11): `reel_subject` turns a `tc:` or `thc:` id into its member
topics, less any umbrella member (≥ 90% of episodes); a clip qualifies when its insight is ABOUT any
of them. `GET /discover/reel?subject=` takes any of the three and reports `subject_kind` and
`subject_label`. `/discover/reel-topics` offers topics, themes and storylines taking turns (four at
most), keyed on the reel's clips so a theme whose reel is its one topic's reel is offered once. A
`ReelDoor` on topic cards and theme and storyline pages opens the reel once it has two clips.

**No repeats** (2026-10-11): within an episode, a clip that restates a stronger one (content-word
overlap ≥ `ECHO_SIMILARITY`, either way) is folded into it, its topics with it. A subject never
takes two clips of one episode that play the same seconds (two insights grounded in one segment).

### 4b. Learn X this week (`app_learning_path`, 2026-10-11)

A seven-day path (operator's choice: one a day, progress on the server, no reminders). `build_path`
takes the subject's strongest episodes by their best clip (`subject_episodes`, the reel's ranking),
at most `PATH_DAYS` (7), at least `MIN_STEPS` (3), and orders them by publish date, oldest first.
`start_path` freezes the steps into `learning_paths.json` with `started_at`; step `k` opens when
`k` whole days (UTC) have passed, or earlier when the listener asks: once the newest open day is
done, "Next day now" opens the next (`ahead` counts days opened early; operator 2026-10-11, "a day a
day is the rhythm, not a wall"). Days opened early **carry forward** (operator 2026-10-12, review N6):
several days can be done in one sitting and the next day continues from there, so opening three early
on day 1 leaves day 5 open on day 2 (`test_learning_path_days.py`). Up to `MAX_PATHS` (3) at once, one per subject; a fourth is
refused (409 `at_cap`) unless the request names one to `replace`. Variety
(operator 2026-10-11): at most two days per speaker and two per show, the next-best episodes filling
the gaps (`MAX_PER_SPEAKER`, `MAX_PER_SHOW`).

```text
GET    /api/app/discover/learn?scope   → { paths, candidates: [{subject_id, kind, label, steps}], max_paths }
POST   /api/app/discover/learn {subject_id, replace?}       → path (422: too few; 409: at_cap)
GET    /api/app/discover/learn/paths/{subject_id}           → path (404: not learning it)
POST   /api/app/discover/learn/paths/{subject_id}/steps/{i}/done → path (409: not open yet)
POST   /api/app/discover/learn/paths/{subject_id}/next      → path (409: newest open day not done)
DELETE /api/app/discover/learn/paths/{subject_id}           → 204
GET    /api/app/discover/learn/history → { paths: [{subject_id, kind, label, started_at, ended_at, days, done, outcome}] }
PUT    /api/app/discover/reel/progress {subject_id, index, clips, minutes} → the place
DELETE /api/app/discover/reel/progress?subject=             → 204 (finished or ✕; only that one)
GET    /api/app/discover/in-progress   → { paths, reel: {subject, index, clips, episode, quote} | null }
```

**Finished** (operator 2026-10-14): the last day done sets the path's `finished_at` and writes it to
the learning history (`learning_history.json`, with paths left early). A finished path stays one of
the listener's paths for a week (its week says "All N days done ✓"; it leaves Home's In progress at
once — `/discover/in-progress` lists only unfinished paths), counts toward neither the cap nor what Discover stops offering, and starting its subject
again starts afresh; a week after its last day only the history has it
(`FINISHED_PATH_KEEP_S`, hidden on read; no other row is rewritten). Profile → Stats lists the
finished paths from `GET /discover/learn/history` ("Paths you finished").

A day's **Play** (operator 2026-10-14) opens `/episode/<slug>?learn=<subject>&day=<n>`: the day's
clip plays as a one-clip reel and stops at its end; heard to the end (80% of it actually played) the
app does the day through `…/steps/{i}/done`, and the end card offers Back to Learn or Keep listening.
"Play all" on the week plays every day not done back to back, doing each and opening the next.

**In progress** (operator 2026-10-11): Home's door into what was started reads
`/discover/in-progress` — every path, and the last short listen left unfinished
(`reel_progress.json`: the app shell writes each clip change of a topic reel once something was
heard — it played, or the listener moved past clip 1 — the last clip's end or ✕ clears it, naming
the reel so another device's newer one stays; "Keep listening" keeps it; a newer short listen
replaces it). A path's day whose episode left the corpus is not shown and counts as done for
"Next day now"; `next_opens_at` says when the next day opens by itself (24 h periods from the
start, not calendar days). Everyone's trending for the Short listens row and the Learn offers is
the cached list `/trending` uses (`corpus_trending`), not five fresh builds per Discover load.

**What was heard is not offered** (operator 2026-10-14: Discover proposes what is still to hear): the Short listens row leaves out a reel whose clips the listener has heard to their end (`REEL_HEARD_SHARE`, 80% of them, from the clip-heard events `POST /me/moments-heard` records), and comes back only when new clips make it a new listen; the Learn offers leave out a subject whose path was finished (it can still be started from its subject page). Where the listener goes on purpose, it is said rather than hidden: a subject page's short-listen door reads "heard ✓" (each reel clip carries `heard`), its Learn door "✓ Learned · learn again", and the Brief's "Play N moments" says how many were heard (`GET /me/moments-heard?slug=`, which also ticks the Moments list).

Candidates are themes and storylines taking turns, three at most, none already being learned, each
only with a path, and none at the cap: under
Mine those of the listener's world (momentum restricted to `personal_entity_ids`, less "Less like
this"), under Everyone the rising ones. Telemetry: `learn_path { action: start | step_done | open_next | stop,
kind, day }`.

### 5. Cold start and Mine

`world_episode_set` adds `followed_interest_episodes`: the episodes of a followed topic or person,
or of a followed theme's or storyline's member topics, from the search sidecar. An interest reaching
at least 90% of episodes is an umbrella and adds nothing. Cold start is an empty world.

### 6. Telemetry

`discover_rail_shown { rail, items, scope }` once per first page;
`discover_rail_tap { rail, reason_kind, action: open|play_from|less, rank }`.
Save, queue and Play moments are the shared episode actions and are not rail taps.

The reel, Learn and Explore (2026-10-11; every event checked end to end into the dev Umami site by
`e2e/telemetry/10-reel-learn-explore.spec.ts`, accepted, never naming a subject or episode):

```text
reel_open        { source: discover_row|subject_page|in_progress|board, kind }
reel_action      { action: start|next|prev|jump|keep|keep_start|done, kind, clips: CountBucket, clip?: RankBucket }
learn_path       { action: start|open|step_play|step_done|open_next|stop, kind, day?, source?: discover|subject_page|in_progress }
trends_show_more { kind, sort, shown: CountBucket }
catalog_filter   { list: episodes, value: all|following|mine }
```

`kind` is `topic|theme|storyline`, or `board` for a board's own short listen (RFC-119 amendment
2026-10-12, #2340), which opens from the board with `source: board`. The board's id is never sent.

### 7. Scale (2026-10-11: 2,500 episodes on prod, 10,000 planned)

Reels and paths read transcripts, so they are built to cost per episode once, not per request:

- **Per-episode clips** (`episode_clips`): every clip a reel could play from an episode, scored
  once and kept without text (about 2.7 KB an episode). The cache key is that episode's own GI,
  metadata and segments files, so an ingest scores only new or changed episodes. Scoring one costs
  about 5 ms on real transcripts (prod-v2), mostly reading its segments.
- **Per-subject lists** (`subject_episodes`) are rebuilt from those on each corpus change: about
  15 µs an episode, about 25 ms for a subject covering 1,600 episodes.
- **Text only for what plays.** The Short listens chips and the Learn offers count clips and
  days from the text-free lists (`reel_selection`, `path_plan`). Only an opened reel or a started
  path reads its clips in full (`hydrate`).
- **Warmed in the background.** The cache warmer scores every episode after startup and after
  each ingest, last of its steps, so readers do not wait on it.

`scripts/tools/bench_discover.py` times every Discover/Explore endpoint, cold and warm, on a
corpus.

## Key Decisions

- The server owns ranking, dedup and paging (ADR-164).
- Support, not position, chooses a quote; the fallback keeps cards.
- "Another view" never claims disagreement (the corpus records agreement only).
- A topic reel is a separate page and never a listen.

## Alternatives Considered

- **Client-side reveal of a longer list.** Rejected by the operator: "Show more" must page.
- **Client-side dedup across independently loaded rails.** Rejected: it depends on load order, and
  the telemetry would count cards that are not visible.
- **One endpoint per rail with sequential loading.** Rejected: the latency is the sum of all rails.

## Testing Strategy

- Pinned evals per rail on the committed fixture (`test_app_*_rail.py`).
- `test_app_discover_rails.py`: page order, one card per episode, paging continues the page, cold
  start, scope.
- Unit tests for the quote choice, `take_turns`, the rail component and the page.
- e2e: each rail, "Show more" issues `offset=2` and the page stays one card per episode, the reel
  loads another episode for the next clip, telemetry against dev Umami.

## Rollout & Monitoring

Ships with the player release that carries the branch. Rails are judged in Umami by the rule in
the plan: captures per tap and cross-show share primary, tap rate secondary, five participants
minimum.

## Open Questions

- "Less like this": remove or down-weight; Discover only or app-wide.
- Rail latency at prod corpus size (measured only on the 55-episode fixture: 17–66 ms in-process
  for the thread rail).

## References

- `docs/wip/discovery-competitive-analysis-2026-10-09.md`
- `docs/wip/DISCOVER-ARC-PLAN-2026-10-10.md`
