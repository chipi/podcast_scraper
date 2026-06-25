# PRD-042: Home (Learning Hub)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.7 (Phase P1)
- **Parent PRD**: `docs/prd/PRD-035-learning-platform.md`
- **Depends on**: PRD-036 (read API), PRD-038 (Catalog), PRD-021/RFC-090 (hybrid search)
- **Adjacent**: PRD-037 (Discovery — add *new* podcasts), PRD-041 (Consolidation — recommendations/recall)
- **Related UX spec**: `docs/uxs/UXS-012-consumer-home.md` (to be written after the design is worked out)
- **Related RFC**: `docs/rfc/RFC-099-learning-platform-consumer-client.md` (§Home & corpus search)
- **Related issue**: GitHub #1090

---

## Summary

Home is the consumer app's **launch surface** — the entry the user sees first. It is **not** a
list and **not** a store: it is an orientation hub for the user's *growing knowledge corpus*.
Its jobs are to **orient** (what's new), **resume** (continue listening), **discover-within**
(recommended + **search your whole library**), and **route** into the Player / Catalog / Shows.
It is where the product thesis — passive listening becoming a growing, queryable personal
knowledge corpus — first becomes visible and actionable.

The full catalog moves to its own surface (`/catalog`, "Browse all"); Home links into it.

## Background & Context

- The MVP made the raw catalog the landing page. That reads as a "list", not a place you
  *return to learn*. Home should orient and resume, not dump a list.
- The corpus-wide grounded search (`GET /api/app/search`, RFC-090) exists but has no consumer
  surface. Home is its natural front door — "ask your library / find any moment across
  everything you've heard" — distinct from the in-episode Ask (PRD-039 FR4.5).
- Recommendations are owned by Consolidation (PRD-041); Home **surfaces** them (a v1 heuristic
  until PRD-041 lands), it does not own the recommendation engine.
- Discovery (PRD-037) is about adding *new* podcasts; Home is the home for what you *already
  have*. They are different surfaces.

## Goals

- Give a returning user an immediate "where was I / what's new" orientation.
- Make the personal knowledge corpus **searchable** from the front door (corpus-wide grounded
  search), with jump-to-episode-and-moment.
- Surface recommended and recent content that leads into the Player in one tap.
- Be the navigation hub: Home → Catalog / Shows / Player / Search.
- Degrade gracefully: every section is independent and hides when empty or signed-out.

## Non-Goals

- Not Discovery — adding new podcasts/feeds stays in PRD-037.
- Not the recommendation *engine* — Home renders recommendations; PRD-041 produces them.
- Not editorial/CMS curation — "featured" is a deterministic heuristic, not hand-curated.
- Not a generative answer surface — corpus search is extractive grounded passages (D6, no
  request-time LLM), same contract as in-episode search.

## Personas

**Returning learner** (resume + what's new), **researcher** (search the whole corpus, jump to
moments), **browser** (shows + featured → something to play).

## User Stories

- _As a returning user, I land on Home and can resume what I was listening to._
- _As a learner, I can search across everything I've heard and jump to the exact moment._
- _As a browser, I see what's new and what's recommended, and play in one tap._
- _As a user, I can get to my full catalog and my shows from Home._

## Functional Requirements

### FR1: Orientation hero + corpus search entry

- **FR1.1**: A hero region establishing identity and the primary action.
- **FR1.2**: A prominent **"Ask your library"** search entry → corpus-wide grounded search
  results (FR5). Whether the search bar *is* the hero or sits within it is a UXS decision.

### FR2: Continue listening (auth)

- **FR2.1**: In-progress episodes (saved playback positions), most-recent first, with a resume
  affordance (e.g. "12:04 / 48:00") → Player resumes at that position.
- **FR2.2**: Hidden when signed-out or when there is no history.

### FR3: What's new

- **FR3.1**: The newest episodes across the library (a short rail), newest-first → Player.
- **FR3.2**: "Browse all →" links to the full Catalog (`/catalog`).

### FR4: Recommended (v1 heuristic)

- **FR4.1**: A small set of suggested episodes ("more like what you've been hearing" /
  topic-adjacent), v1 heuristic over existing similarity; full personalisation is PRD-041.
- **FR4.2**: Clearly degradable — hidden when no signal (e.g. signed-out / no history / no index).

### FR5: Corpus-wide grounded search (new surface)

- **FR5.1**: A search results surface (`/search?q=`) over the **whole library** via the hybrid
  search index (RFC-090), returning **grounded passages** — verbatim text, source episode +
  speaker, and a timestamp that **opens the Player at that moment**. No request-time LLM (D6).
- **FR5.2**: Graceful when the index is unavailable (200 + empty, message), same as in-episode.

### FR6: Your shows

- **FR6.1**: The podcasts in the user's library as a grid → that show's Catalog view.

### FR7: Featured / spotlight (optional)

- **FR7.1**: One deterministic spotlight episode (e.g. newest with rich insights) with a play CTA.

### FR8: Listening stats / "your knowledge" (later)

- **FR8.1**: Aggregate signals (episodes, hours, topics) — a Consolidation-adjacent later add.

## API summary

| Method | Path | Description | Status |
| --- | --- | --- | --- |
| `GET` | `/api/app/episodes` | What's new / spotlight (newest) | Shipped |
| `GET` | `/api/app/search?q=` | Corpus-wide grounded search (FR5) | Shipped (#1068) |
| `GET` | `/api/app/episodes/{slug}/related` | Recommended seed (peers) | Shipped (#1084) |
| `GET` | `/api/app/podcasts` | Shows list (FR6) | **Net-new** |
| `GET` | `/api/app/playback` | Saved positions for Continue (FR2) | **Net-new (auth)** |

## Success Metrics

- A returning user can resume from Home in one tap.
- Corpus search returns grounded passages with jump-to-moment within ~1s; no request-time LLM.
- Home → Catalog / Shows / Player / Search all reachable in one tap.
- Every section degrades cleanly (signed-out, empty, no index) — no broken panels.

## Open Questions

- **Organizing principle / hero**: **Resolved (2026-06-24)** — **adaptive hero**: resume-state
  (Continue) when signed-in with in-progress history, else discover-state ("Ask your library" +
  Featured); the corpus search is prominent in **both** states. Worked out via the two mockups
  (`docs/wip/player/mockups/home-*`); contract in UXS-012, behaviour in RFC-099.
- Which sections ship in Home v1 vs. iterate (Continue/Recommended/Stats need auth + new
  endpoints; What's-new/Shows/Featured are cheap).
- Recommended heuristic for v1 (peers-of-recent-plays vs top-by-topic) until PRD-041.

## References

- `docs/prd/PRD-035-learning-platform.md`, `PRD-038-catalog.md`, `PRD-039-player.md`,
  `PRD-037-discovery.md`, `PRD-041-consolidation.md`
- `docs/rfc/RFC-099-learning-platform-consumer-client.md`, `docs/rfc/RFC-090-*` (hybrid search)
