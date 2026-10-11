# ADR-163: The jobs of Discover — episode-first rails, each with a grounded "why"

- **Status**: Accepted
- **Date**: 2026-10-10
- **Authors**: Marko Dragoljevic
- **See Also**: [ADR-162](ADR-162-mine-means-the-listeners-own-world.md) ("Mine"),
  [UXS-012](../uxs/UXS-012-consumer-home.md) (Discover), the competitive scan
  `docs/wip/discovery-competitive-analysis-2026-10-09.md`

## Context & Problem Statement

Discover grew surface by surface: trending shows, Trends (rising topics, themes, storylines,
people), the entity dashboards and search. It answers several questions at once — what is
happening in podcasts, what is popular, what to hear next — and the popularity-shaped surfaces
overlap. A side-by-side preview of a reworked page (Discover 2) was built and then removed
(2026-10-10): the redesign starts from evidence, not from a preview.

The competitive scan (2026-10-09, 12 apps) found:

- **Explaining why is the best-evidenced lever.** Spotify users were up to 4× more likely to click
  a recommendation with an explanation.
- **Discovery moved to episodes and inside them** (Castro, YouTube, Spotify): chapters, mentions,
  sampling without committing.
- **Recommenders optimise "not your usual habit"** (Spotify GLIDE: up to +14.3% new-show
  discovery).
- **No app recommends from what a listener captured**, and none backs a recommendation with the
  quote that justifies it. We hold both: every insight has a quote and a time; listeners keep
  highlights, notes and saves.
- **Anti-patterns:** popularity loops, several rails doing one job, recommendations inside a "mine"
  surface, explanations the content cannot back, a social graph before demand.

The scan ended with eight open questions. The operator answered them on 2026-10-10. This record
fixes those answers so every later rail is built against the same rules.

## Decision

1. **Discover has one job: what to hear next.** Everything on it recommends something to listen to.
   **Trends and the entity dashboards move to their own Explore surface** — what podcasts are
   talking about is a different question from what to hear next.
2. **Episode-first.** Rails recommend episodes, and the moments inside them. Shows are reached
   through their episodes; trending shows, while they remain, sit below the episode rails.
3. **Every rail has one job and one kind of "why"**, declared when the rail is built. Two rails
   that would do the same job are one rail.
4. **The "why" is extractive: the listener's own capture and a grounded quote with its time** —
   "Because you highlighted '…' → at 23:14 X says '…'". Never a sentence the transcript cannot
   back: no LLM-written summary line. A recommendation with no grounded reason shows none rather
   than an invented one. *Amended 2026-10-11:* a **quote rail** (its card IS a quote) skips an
   item with no quote; every **other rail** shows the item with no reason. Rails do not each decide
   differently.
5. **Momentum weighs captures and completion over plays.** Highlights, saves and finished listens
   count more than opens and starts, so a small, noisy user base cannot inflate an item by opening
   it. Rising stays relative to the item's own history (it already favours small shows).
   *Amended 2026-10-11 (advisor review):* counted in **distinct listeners**, not events.
   - Each listener counts once per signal per entity per week, weighted: finished (75/95%) 3,
     captured 3, saved 2, opened, subscribed and followed 1 each. A weighted event sum would let
     one listener's three highlights weigh 9 — the inflation this rule exists to prevent.
   - Floors are in listeners: corpus-wide at least 2, and "most engaged" at least 2.
   - A card's count is the listener count.
6. **Cold start: Everyone, labelled, plus an optional picker.** A listener with no history sees
   Everyone's rails, labelled as such ("Everyone", never "For you"). A pick-a-few-topics-or-people
   step, optional and skippable, seeds Mine. A "Suggest less" control lets a listener steer.
   *Amended 2026-10-11 (operator):*
   - **The picker seeds Mine by widening it.** A followed topic or person brings its episodes into
     the listener's world, and a followed theme or storyline brings its member topics' episodes
     (ADR-162 amendment). Umbrella topics are excepted, and so is any interest reaching ≥ 90% of
     episodes, so following one cannot make Mine everyone's.
   - **Cold start is an empty world, not an empty rail.** Only then does Mine's quote rail show
     Everyone's cards, labelled, with "Pick topics". A listener with history whose Mine is quiet
     sees it empty, never told their picks "begin once you listen".
   - **"Suggest less" hides the card's reason**: the topic, person or storyline its "why" names,
     across every rail, undone from Profile › Interests. *Initial setup, 2026-10-11 (operator):* it
     REMOVES (a down-weighted card that still shows reads as being ignored), in BOTH scopes (it is
     the listener's preference about recommendations), on every recommendation surface — Discover's
     rails, the "15 minutes on" row, the Learn offers — and never where the listener asks: Search,
     Explore's Trends and catalogue, the subject's own page. No expiry; the "less" taps per rail are
     measured, and down-weighting or fading is reconsidered on that evidence.
7. ~~**Serendipity: a small, labelled share inside each rail.**~~ *Superseded 2026-10-11
   (operator):* **under Mine everything on the page is the listener's own.** A corpus holds babies,
   history, cars and sport; a listener who cares about none of them is not shown them under Mine,
   labelled or not. Breadth is the Everyone scope, one tap away on the same switch. And because a
   rail says what to hear next, no Mine rail offers an episode the listener already heard, captured
   from or saved; the quote rail reaches any episode about a rising topic of their world, not only
   episodes already in it.
8. **A quote card opens the player at the quote's moment.** No separate in-place clip player: the
   episode plays from the time, with its context and capture at hand, under the same rights as any
   playback. Moments stays the way to hear an episode's best parts back to back.
   *Amended 2026-10-11 (operator: goal-shaped listening is a cross-episode reel):*
   - **"N minutes on a topic"** is a second reel. It takes the strongest clips about one topic from
     different episodes, back to back (the Moments picker's clip windows, breadth first), and lives
     on its own page (`/listen/topic/:id`).
   - Each clip loads its own episode under the same rights as any playback.
   - Like Moments, it never moves a resume point or counts as a listen until "Keep listening to
     this episode" makes it one.
   - A clip must say what its insight says, so no greetings or ad reads.
   - The page states what the reel actually runs, never the budget.
9. **No curation for now.** No editorial lists, no creator Podroll, no social graph. Revisit when
   there are listeners and curators to serve.
10. **Scope is labelled honestly** (ADR-162): a "Latest from shows you follow" rail holds only
    followed shows — a recommendation never appears inside a "mine" surface.

## Consequences

**Positive**

- One rule set for every rail: a new rail states its job and its "why" kind, or it does not ship.
- The differentiator is concrete and checkable: each recommendation carries the quote and the time
  that justify it, built from data no other app has (captures × grounded insights).
- Discover and Explore stop competing for the same space.
- Captures-and-completion momentum is harder to game and matches the product's purpose (keeping
  what you heard).

**Negative**

- Some recommendations will show no reason, where the grounding is missing. Accepted: an honest
  blank over a confabulated reason.
- Moving Trends to Explore changes where returning listeners find them. Explore needs a clear way in
  from Discover.
- Captures are sparse while the user base is small, so momentum may be thin at first and lean on
  completion.
- Without a share from outside, Mine can narrow listening; breadth depends on the listener
  switching to Everyone.

**Neutral**

- The existing APIs stay as building blocks: `GET /trending/episodes` (quote cards from rising
  topics and the most engaged), `/interests/derived`, storylines, person cards.
- Measurement comes first among the follow-ups: each rail reports shown, opened, played and
  captured, so rails are kept or dropped on data rather than feel.

## Alternatives Considered

- **Show-first, or shows and episodes equally** — rejected: the evidence (Castro, YouTube,
  Spotify) and our own data (insights and moments are per episode) point to episodes.
- **A grounded quote plus an LLM-written one-liner** — more readable, but any sentence the
  transcript cannot back undercuts the differentiator; it also needs an LLM call per card. A named
  reason only (Pocket Casts' "Because you follow X") — rejected as weaker than the quote we have.
- **Momentum from plays and opens** (today's signal), or from the corpus alone — rejected: plays are
  noisy at our size, and corpus-only momentum ignores what listeners actually keep.
- **Everyone with no picker, or a required onboarding** — rejected: the first leaves Mine empty
  longer than needed; the second puts a gate before any value.
- **A separate "Something different" rail, or no serendipity** — rejected at first: a separate rail
  is easily skipped; none narrows listening (Spotify's own diversity research). The labelled
  one-in-five share chosen instead was removed on 2026-10-11 (§7): under Mine the operator wants
  only the listener's own world.
- **Trends staying on Discover, or folded into the rails' reasons only** — rejected: two jobs on one
  page, or a useful surface lost.
- **Playing clips in place on the card, or both** — rejected for now: a second player and its rights
  questions, where opening the episode at the moment does the job.
- **Creators' Podroll tags or our own editorial picks** — deferred, not rejected: cheap (Podroll) or
  curator-bound (editorial), both worth revisiting with an audience.
