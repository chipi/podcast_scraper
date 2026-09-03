# Collaborative filtering: the signal audit (#1923)

**Status:** audit complete, no model built — which is what the issue asks for first.
**Date:** 2026-09-03. **Branch:** `feat/player-offline-downloads`.

#1923 says the first piece of work is "an honest audit, not a model", and names three questions.
This answers them, plus two the audit turned up on its own.

---

## Q1. What is captured today?

Every per-user signal, and whether it can carry collaborative weight:

| Signal | File | Shape | Usable for CF? |
|---|---|---|---|
| Episode starts | `listen_events.jsonl` | append-only, one line per start, `{ts, slug, feed_id}` | **Yes** — the natural user×item matrix |
| Playback position | `playback.json` | one row per episode ever, `{position_seconds, updated_at, finished}` | Partly — `finished` is a strong positive, position is a weak one |
| Listening time | `listening_daily.json` | per-day seconds (NEW, #1914 Phase 0) | **Yes, and better** — dwell beats a click |
| Interest follows | `interest_events.jsonl` | explicit follows of topics/people | Yes — but it is *stated* taste, which the app already uses |
| Highlights / notes | `highlights.json`, `notes.json` | with `created_at` | **Strongest positive** — capture is costly, so it means something |
| Favourites | `favorites.json` | with `added_at` | Yes |
| Ranking clicks | `ranking_events.jsonl` | discover-feed clicks | Yes, with position bias |

Cross-user aggregation **already exists and is shipped**: `app_engagement_series._iter_user_ids`
walks every user dir, and `app_stats.compute_episode_stats` computes per-episode reach. So CF needs
no new access pattern — the precedent is set.

## Q2. What is missing or distorted?

The issue lists three distortions and notes they are shared with #1914. **All three have since been
fixed by the offline arc (#1925), and this audit is the first place that is written down:**

- ~~"Listen events fire from exactly one call site (`PlayerView`), so queue auto-advance and
  mini-player playback log nothing"~~ — **fixed** (#1924). The logger moved into the player store;
  every start is recorded, including auto-advance, which no view observes.
- ~~"offline listening logs nothing at all"~~ — **fixed** (#1924). Queued on device with the moment
  it happened, flushed on reconnect, deduplicated server-side on `(slug, client_ts)`.
- ~~"opens double-count by design (one per mount)"~~ — **fixed** (#1924). A listen is armed on load
  and fired on the first `play`, so opening an episode page is no longer a listen. This one
  mattered most for CF: page views are *browsing*, and treating them as listening would have made
  the co-occurrence matrix mostly a map of what the UI puts on screen.

**Still distorted, and relevant:**

1. **No negative signal at all.** Nothing records a skip, an abandon, or an episode opened and
   dropped after 20 seconds. Every CF shape below is therefore positive-only (implicit feedback),
   which is normal — but it must be *chosen*, not stumbled into.
2. **`position_seconds` is furthest-reached, not listened.** Already documented in #1914; it means
   "heard fraction" over-reports on a seek. `listening_daily` is the honest replacement but only
   starts accruing from 2026-09-03.
3. **Highlights are sparse by nature.** The strongest signal is also the rarest.

## Q3. How many users are there? — measured, not estimated

Against production (`prod-podcast.tail6d0ed4.ts.net`, reachable over the tailnet):

- `/api/app/podcasts` → **24 shows**.
- `/api/app/episodes?limit=200` → **20 episode rows returned**.
- `/api/app/episodes/{slug}/stats` for **all 20** → `{"listeners":0,"opens":0}` — every one.

**Caveat, stated because it changes the reading:** that endpoint's own docstring notes counts are
zero when no app data dir is configured, and I could not distinguish "no users" from "no app data
dir" remotely — SSH to the prod host is key-denied for my account. What is certain is that the
shipped cross-user endpoint reports **no interaction data whatsoever** in production. The consumer
app is also still behind a coming-soon gate, so a zero user base is the *expected* state, not a
malfunction.

### The cold-start floor, with arithmetic

For item-item co-occurrence, the expected number of users who heard **both** items in a pair is
roughly `U · (E/N)²`, where `U` = users, `E` = episodes each user hears, `N` = catalogue size. For
a pair to be worth showing you want that to reach ~5.

| Level | N (items) | E per user | Users needed for ~5 co-occurrences |
|---|---|---|---|
| Episode-level | ~680 | 20 | **~5,700** |
| Episode-level | ~680 | 50 | ~925 |
| **Topic-level** | ~200 | 30 | **~220** |
| Topic-level | ~200 | 50 | ~80 |

This is the quantitative version of the hunch already in the issue — *"sparsity at the topic level
is far lower… possibly the right first surface rather than the second."* It is right, and the gap
is an order of magnitude, not a rounding difference. **Episode-level CF is out of reach until the
user base is in the thousands. Topic-level becomes meaningful in the low hundreds.**

## Q4 (new). A privacy defect exists TODAY, before any CF is built

`GET /api/app/episodes/{slug}/stats` is **unauthenticated by design** ("Public (no auth) — returns
only anonymous aggregate counts"). That reasoning holds at scale and fails at small N:

- With one user in the system, `listeners: 1` on an episode says *that user listened to it*.
- Anyone who can name a user and reach the endpoint can enumerate the catalogue and reconstruct a
  substantial part of that person's listening history.

Listening history is sensitive. This is not hypothetical at the current user count — it is the
current user count that makes it acute. **Recommendation: suppress below a k-anonymity floor**
(return `listeners: null` under, say, 5) rather than reporting exact small counts. Cheap, and it
must land *before* any surface that makes cross-user data more visible.

## Q5 (new). The signal that would actually be differentiating is not being written

The blend the issue calls "the thing no competitor with only one of the two can copy" — semantic
similarity × co-listening — needs co-listening at the **topic** level, which the KG can already
resolve. But nothing today writes a per-user topic-exposure log; topic interest is *derived* on
read from the episode set, time-decayed. Deriving is fine for one user's profile and wrong for
co-occurrence, because the decay makes the matrix depend on *when it was computed*.

## Recommendation

1. **Do not build a model now.** At zero measured interactions, any CF output is noise wearing a
   confident label — which is exactly what the issue set out to avoid.
2. **Land the k-anonymity floor on episode stats now.** It is a real defect, it is small, and it
   is a prerequisite for everything else here.
3. **Write the topic-exposure log now** (undecayed, per user, append-only), for the same reason
   #1914 Phase 0 shipped ahead of recaps: *a day not recorded is gone*. Recording is cheap;
   retrofitting history is impossible.
4. **Revisit the model at ~200 real users**, topic-level first, with the honest-empty pattern the
   repo already uses (`scope=mine` search returns zero-coverage rather than a global fallback).

## What this audit did NOT do

- **No model, no evaluation, no offline metrics.** There is nothing to evaluate against.
- **No prod user count.** SSH is key-denied; only the aggregate endpoint was reachable.
- **No exact prod catalogue size.** `limit=200` returned 20 rows; whether that is the catalogue or
  a server-side cap was not established, so the arithmetic above uses ~680 from the corpus notes
  rather than a measured N.
- **No decision on negative signal.** Flagged as a choice to make, not made.
