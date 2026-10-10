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

1. **The "why" contract.** One server shape every recommendation carries: the reason kind, the
   listener's capture when there is one, and the grounded quote with its episode and time. A
   recommendation without one says so. Shared by every rail endpoint, rendered by one component.
2. **Per-rail measurement.** Events shown, opened, played from the card, and captured afterwards,
   per rail and per reason kind, in the existing analytics pipe. A small dashboard view to compare
   rails.
3. **Momentum on captures and completion.** Re-weight the momentum signal behind
   `GET /trending/episodes` (and trending shows) as ADR-163 §5 says; keep "rising against its own
   history".

## Step 2 — the first rails

1. **Quote cards** (scan #2) — rising topics and the most engaged, as quote + "Play from 23:14";
   builds on `GET /trending/episodes`.
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
- **Serendipity share:** a shared helper every rail uses (ADR-163 §7).

## Not decided yet

- The order inside Step 2, if measurement suggests otherwise.
- What the Explore surface is called, and where its entry sits.
- What the picker offers and how many picks it asks for.
