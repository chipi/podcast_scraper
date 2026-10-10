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
     - `discover_rail_tap {rail, reason_kind, action: open|play_from, rank, outside_habits}`;
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
4. **The serendipity helper** (ADR-163 §7), moved up from "Also": the first rail needs it.
   - "Outside" means the show is not in the listener's shows and no topic or person is in their
     world (ADR-162).
   - Under Everyone there are no habits, so no label.

## Step 2 — the first rails

1. **Quote cards** (scan #2) — rising topics and the most engaged, as quote + "Play from 23:14";
   builds on `GET /trending/episodes`. **Trends move to Explore in the same step** (named first),
   so Discover never carries three popularity surfaces at once.
2. **More from what you captured** (scan #3, unique) — highlights, notes and saves seed episodes on
   other shows, by the same topic, person or storyline. Server-side, not the preview's per-highlight
   searches.
3. **Continue the thread** (scan #4) — new episodes in the storylines the listener follows, in time
   order.

## Step 3 — more rails

- **People you keep hearing** (scan #5) — recurring real speakers (not people merely mentioned) on
  shows the listener does not follow.
- **Sample an episode** — its best moments as a short taste, reusing the Moments picker.
- **Hear another view** (scan #7) — a contrasting speaker on a topic the listener engages with.
- **Mentioned in this episode** (scan #8) — people and topics, linked onward.
- **Goal-shaped listening** (scan #6) — "15 minutes on X", "Learn X this week".

## Step 4 — cold start and control

- The optional topic / person picker (seeds Mine).
- "Suggest less".
- The Everyone fallback, labelled, until history exists.

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
- What the Explore surface is called, and where its entry sits.
- What the picker offers and how many picks it asks for.
