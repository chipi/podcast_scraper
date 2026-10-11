# Discover arc — plan (from the competitive scan)

Branch `feat/player-1.0.4`. The rules every step follows are **[ADR-163](../adr/ADR-163-the-jobs-of-discover.md)**
(decided 2026-10-10). The evidence is `discovery-competitive-analysis-2026-10-09.md`; the scan as a
page is https://claude.ai/artifact/HRXJkuKB6sqZdiFjma4pez.

Every rail ships with:
- a server endpoint with unit and integration tests;
- an eval where ranking matters (real corpus, frozen cases);
- UI in the real Discover (no side-by-side preview);
- its job and its "why" kind stated in UXS-012;
- per-rail measurement (Step 1).

## Step 0 — decisions ✅ (2026-10-10)

The scan's eight open questions were answered and recorded as ADR-163:
- **Unit:** episode-first.
- **Why:** quote and time only.
- **Momentum:** captures and completion.
- **Cold start:** Everyone plus an optional picker.
- **Serendipity:** a labelled ~1-in-5 share per rail.
- **Trends:** move to Explore.
- **Previews:** a card opens the player at the moment.
- **Curation:** none for now.

The Discover 2 preview was removed (UI only; its APIs stay).

## Step 1 — the foundation (unblocks every rail)

1. **The "why" contract** ✅ (2026-10-10).
   - **Server:** `app_reasons.Reason` / `AppReason` (kind, subject, the listener's capture, the
     grounded quote with episode and time, events, `outside_habits`) and `grounded_quote`, which
     never falls back to an insight's own text.
   - **App:** `RecommendationWhy` writes the words and offers "▶ Play from" (`?t=&play=1`).
   - **First adopter:** `GET /trending/episodes`, which also stopped quoting insight paraphrases
     when there is no quote.
2. **Per-rail measurement** ✅ (contract, 2026-10-11).
   - **The split** (advisor review 2026-10-11): who reads a signal decides where it is stored.
     - What only the operator reads is usage analytics, in Umami, under the listener's opt-out.
     - What a feature reads for that user stays server-side as product state, governed by Clear
       listening history.
     - A signal can be both, written twice by different writers. A feature never reads the
       analytics store.
     - Rail impressions, taps and "captured after a rail" are analytics. Completion, captures and
       saves are product state (momentum reads them).
   - **The contract** (`services/analytics.ts`, compiler-checked):
     - one `Source` per rail (`discover_quotes`, `discover_captured`, `discover_thread`,
       `discover_people`, `discover_another_view`, `discover_trending_shows`, `explore_trends`);
     - `discover_rail_shown {rail, items, outside, scope}`, once per rail per mount, only when it
       rendered a card;
     - `discover_rail_tap {rail, reason_kind, action: open|play_from|moments, rank, outside_habits}`;
     - `capture_created.source`, how the capture screen was reached.
   - **Wiring:**
     - `useRailTelemetry(rail)` reports `shown`/`tapped` and marks the next navigation with the
       rail's source (`provenance.markNextNavigation`).
     - The router no longer resets the source on a query-only change, so a capture made in the
       player after `?t=` rewrites still credits the rail. Same visit only; no cross-session join.
   - Removed: the unused ranking-experiment telemetry (`app_ranking_telemetry`, `POST
     /discover/click`).
   - **How a rail is judged** — written before the first rail ships, so "kept or dropped on data"
     is true:
     - primary: `capture_created` with the rail's source ÷ `discover_rail_tap` for the rail
       (captures per tap), and the cross-show share of its opens (`episode_open.from_followed_show
       = false`);
     - secondary: tap rate (`discover_rail_tap` ÷ `discover_rail_shown`);
     - **floor**: a rail is not judged until it was shown to at least **5 distinct participants**.
       Rails are compared with each other, not against an absolute bar. Blockers and the opt-out
       under-count every rail alike.
   - Each rail's definition of done includes wiring these events.
3. **Momentum by distinct listeners** ✅ (2026-10-11). Implement ADR-163 §5 as distinct listeners per signal, not
   a weighted event sum, in `app_engagement_series`:
   - **Signals:** finished (playback 75/95), captured (highlights, notes), saved and opened.
   - **Floors:** listener-based, at least 2 distinct listeners.
   - **The card:** `most_engaged.events` becomes the listener count.
   - An ADR-163 amendment line records it.
4. **The serendipity helper** ✅ (2026-10-11; `app_serendipity`) (ADR-163 §7), moved up from "Also": the first rail needs it.
   - "Outside" means the show is not in the listener's shows and no topic or person is in their
     world (ADR-162).
   - Under Everyone there are no habits, so no label.

## Step 2 — the first rails

1. **Quote cards** ✅ (2026-10-11) (scan #2) — rising topics and the most engaged, as quote +
   "Play from 23:14", on `GET /trending/episodes`.
   - **On Discover:** `QuoteCardsRail` leads, with `RecommendationWhy` and
     `useRailTelemetry('quotes')`.
   - **Mine:** every 5th card is the labelled outside-habits share (`mix_outside_habits`).
   - **Fix:** perspective insights now carry their supporting quote (`cil_queries` annotates
     `_quote_text`). Without it the extractive rule left rising-topic cards with nothing to quote,
     so they never appeared.
   - **Explore** ✅ (`/explore`): Trends and the Episodes · Shows catalog moved there (operator:
     the catalog too). Old `/browse` links to them redirect; Discover's header links to it.
   - The Umami telemetry spec checks the rail contract end to end.
2. **More from what you captured** ✅ (2026-10-11; `app_captured_rail`, `GET /discover/captured`, the
   generic `RecommendationRail`; eval pinned in `test_app_captured_rail.py`) (scan #3, unique) — highlights, notes and saves seed episodes on
   other shows, by the same topic, person or storyline. Server-side, not the preview's per-highlight
   searches.
3. **Continue the thread** ✅ (2026-10-11; `app_thread_rail`, `GET /discover/thread`; eval pinned in
   `test_app_thread_rail.py`) (scan #4) — new episodes in the storylines the listener follows:
   about one of the storyline's topics, not heard, captured or saved, published after their last
   listen in it (all of them before a first listen), newest first, storylines taking turns.

## Step 3 — more rails

- **People you keep hearing** ✅ (2026-10-11; `app_people_rail`, `GET /discover/people`; eval pinned
  in `test_app_people_rail.py`) (scan #5) — hosts and guests in two or more of the listener's
  episodes (never people merely mentioned), on shows they neither follow nor heard them on.
- **Quotes back an insight** ✅ (2026-10-11; `best_grounded_quote`) — every rail's quote is one that
  shares words with an insight of its episode, no longer the first in the list (greetings, ad
  reads). Residual: where the GI has nothing better by that speaker, the greeting stays.
- **Sample an episode** ✅ (2026-10-11) — every recommendation card carries the shared "▶ Moments"
  door (`MomentsLink`, only when `has_moments`), which opens the episode's Moments reel
  (`?moments=1`); its tap is measured as `action: moments`. Surfaced a real bug, fixed: a `play()`
  that a pause overtook (AbortError) showed "Couldn't load the audio" over a playing episode.
- **Hear another view** ✅ (2026-10-11; `app_another_view_rail`, `GET /discover/another-view`; eval
  pinned in `test_app_another_view_rail.py`) (scan #7) — a speaker the listener has not heard on
  one of their top topics. "Another", not "contrasting": `topic_consensus` keeps agreement pairs
  only, so contrast cannot be claimed; agreeing speakers are skipped. Umbrella topics (in ≥ 90% of
  episodes) skipped.
- **Rails take turns properly** ✅ (`take_turns`) — a list whose next episode another list took
  gives its following one, so overlapping rails still fill (the earlier loop skipped the turn).
- **Mentioned in this episode** (scan #8) — already present, nothing built: the player's Brief
  lists the episode's topics and people as chips that open their entity cards
  (`KnowledgePanel.vue`, `kp-topic-chip` / `kp-person-chip`, with host/guest roles).
- **Goal-shaped listening** (scan #6) — "15 minutes on X", "Learn X this week".

## Step 4 — cold start and control

Operator decisions 2026-10-11 (all the recommended options): goal-shaped = a cross-episode reel;
"Suggest less" hides the card's reason subject; picked interests widen Mine.

- The optional topic / person picker (seeds Mine) ✅ — the existing `InterestsPicker`, opened from
  Discover's cold-start note (`trigger: discover_cold_start`). Follows widen Mine
  (`followed_interest_episodes` in `world_episode_set`; umbrellas excepted). ADR-162/163 amended.
- The Everyone fallback, labelled, until history exists ✅ — gated on an EMPTY WORLD, not an
  empty rail. Surfaced and fixed: with no habits, Mine filled with Everyone's cards each marked
  "Outside your usual"; now no habits → no outside share.
- "Suggest less" — hides the card's reason subject (topic / person / storyline) everywhere.

## Also

- **Explore surface:** Trends and the entity dashboards move off Discover (ADR-163 §1), with a clear
  way in from Discover.

## Cold start and evals

- **Rails with no listener data:** `rising_topic` is content momentum, dense from day one.
  `most_engaged` and the captures-based rails sit under their listener floors for a while.
- **Thin data:** a rail with nothing to show hides itself; floors are not lowered to fill the page.
  Under Everyone, Discover shows the content-led rails.
- **Evals:** each rail's eval is a frozen per-user fixture → expected top-k episodes with reason
  kinds. Deterministic, disk-only, no LLM.

## Not decided yet

- The order inside Step 2, if measurement suggests otherwise.
- Whether Explore also gets a nav tab (today: a link in Discover's header).
- What the picker offers and how many picks it asks for.
