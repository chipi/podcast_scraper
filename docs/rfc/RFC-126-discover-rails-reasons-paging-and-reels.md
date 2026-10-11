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
4. "Less like this", the outside share and cold start apply on the server, on every page.
5. A topic reel plays clips across episodes without moving a resume point or counting as a listen.

## Constraints & Assumptions

- Read paths are disk-only: GI, KG, bridge, storylines, the search sidecar, user state files.
- Mine is `world_episode_set` / `personal_entity_ids` (ADR-162), never re-derived per rail.
- Pre-launch: no backward compatibility for the removed per-rail endpoints.

## Design & Implementation

### 1. The reason (`app_reasons`)

```text
Reason { kind, quote?: ReasonQuote, subject?: ReasonSubject, capture?: ReasonCapture,
         events?: int, outside_habits: bool }
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
3. mixed with the outside share (`mix_outside_habits`: every fifth slot, labelled; none without
   habits);
4. for `quotes` under Mine with an empty world: everyone's list, `fallback = true`.

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

**Planned:** reels for themes and storylines (clips about any member topic); "Learn X this week" as
an ordered path through a theme (one episode a day, oldest claim first, each with its best clip).

### 5. Cold start and Mine

`world_episode_set` adds `followed_interest_episodes`: the episodes of a followed topic or person,
or of a followed theme's or storyline's member topics, from the search sidecar. An interest reaching
at least 90% of episodes is an umbrella and adds nothing. Cold start is an empty world.

### 6. Telemetry

`discover_rail_shown { rail, items, outside, scope }` once per first page;
`discover_rail_tap { rail, reason_kind, action: open|play_from|less, rank, outside_habits }`.
Save, queue and Play moments are the shared episode actions and are not rail taps.

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
