# PRD-048: Discover and Explore — what to hear next, and looking around

**Status:** Implemented in part (branch `feat/player-1.0.4`; epic #2306)
**Date:** 2026-10-11
**Related:** ADR-162 (what "Mine" means), ADR-163 (the jobs of Discover), RFC-126 (the design),
ADR-164 (Discover is a taste), ADR-165 (grounded validation corpus), UXS-011, UXS-012

## Summary

Discover answers one question: **what should I hear next?** Every recommendation is an episode, and
each one says why it is there in words the transcript can back: the listener's own capture, and a
quote with the time it is said. Explore is where a listener looks around: what is rising, and the
whole catalogue. Both pages are a taste, not a scroll: a few cards per section, and more on request,
paged from the server.

## Background & Context

The competitive scan (`docs/wip/discovery-competitive-analysis-2026-10-09.md`) found that no
podcast app recommends from what a listener highlighted, and that a grounded "why" is rare.
Recommendation surfaces elsewhere fail in known ways: popularity feedback loops, several rails
doing the same job, recommendations inside "mine" surfaces, and confabulated explanations.

Before this work, Discover mixed four jobs on one page: trends, the catalogue, trending shows and
search. Its recommendations were show-first and did not say why.

## Goals

1. Every Discover recommendation is an episode with a grounded reason.
2. Recommendations come from what the listener captured, follows and keeps hearing, not only from
   popularity.
3. Under Mine everything on the page is the listener's own; breadth is the Everyone scope.
4. A new listener sees something useful at once, labelled as everyone's, and can make it theirs.
5. The listener can steer ("Less like this") and undo.
6. Neither page asks for a long scroll.
7. Every rail is measured, so it can be kept, changed or dropped on evidence.

## Non-Goals

- **Curation** (editorial lists, creator picks, a social graph): ADR-163 §9.
- **LLM-written reasons.** A reason is extractive or absent.
- **Claimed disagreement.** "Hear another view" is a different speaker; the corpus records agreement
  pairs only, so contrast is not claimed.
- **A separate in-place clip player** for quotes (ADR-163 §8). The cross-episode reel is its own page.

## Personas

- **The regular listener** has history: captures, follows, listening. Discover should feel like
  theirs.
- **The new listener** has nothing yet. Discover should still be worth opening, and say honestly
  whose picks they are.
- **The curious browser** wants to look around a topic or the catalogue: Explore.

## User Stories

- As a listener, I see episodes worth hearing next, each saying why, and I can play from the quote.
- As a listener who highlighted something, I see episodes elsewhere that continue it.
- As a listener who follows a storyline, I see what is new in it since I last listened.
- As a listener, I meet the people I keep hearing on shows I don't follow yet.
- As a listener, I hear a different speaker on a topic I care about.
- As a listener with fifteen minutes, I hear the best clips on a topic across episodes.
- As a new listener, I see everyone's picks, labelled, and can pick a few topics to make them mine.
- As a listener, I can say "Less like this", undo it at once, or undo it later in Profile.
- As a listener, I can save, queue or play any recommended episode, and open its show.
- As a listener on a phone, I reach search in one tap, and Explore from the tab bar.

## Functional Requirements

### FR1: The reason ("why")

Every card carries a reason (`app_reasons.Reason`): its kind, its subject (topic, person, storyline
or theme), the listener's capture where one seeded it, and a quote with its time. The quote is
transcript text that backs an insight of its episode (`best_grounded_quote`); a greeting or an ad
read never stands in. With no backing quote, a quote rail skips the item; other rails show it with
no quote (ADR-163 §4).

### FR2: The rails (Discover)

In page order:

| Rail | Who sees it | Job |
| --- | --- | --- |
| Worth hearing next | everyone | rising topics and the most engaged episodes (distinct listeners) |
| More from what you captured | signed in, Mine | episodes elsewhere linked to a highlight, note or save |
| Continue the thread | signed in, Mine | new in the storylines the listener follows |
| People you keep hearing | signed in, Mine | recurring hosts and guests, on other shows |
| Hear another view | signed in, Mine | a speaker not yet heard on one of their topics |

Under Mine every card is the listener's own: nothing from Everyone is mixed in (ADR-163 §7,
superseded 2026-10-11), and no rail offers an episode they already heard, captured from or saved.

### FR3: A taste, not a scroll

Each rail shows two cards. "Show more (N)" fetches the next page from the server; N is what the
server says is left. An episode appears once on the page: the highest rail keeps it (ADR-164).
Explore's Trends work the same way: three rows on a phone and five on desktop, then the next page
from the server, ordered there by the active sort.

### FR4: Cold start

Under Mine with nothing in the listener's world, the first rail shows everyone's cards under the
note "Everyone's picks — yours begin once you listen, save or pick a few topics", with "Pick topics"
(the interests picker). Followed topics, people, themes and storylines widen Mine (ADR-162).

### FR5: "Less like this"

On a card whose reason names a subject, one tap hides that subject from every Discover rail and
from the "15 minutes on" row, with Undo in place. Profile › Interests › Suggested less lists what is
hidden and undoes it.

### FR6: Card actions

Every card has the standard episode actions (save, queue, ⋯ with Play moments and add to board, and
download) and "Play from" the quote. The whole card opens the episode in the player, paused; the
show is named, not linked — the player leads to the show (operator 2026-10-11, UXS-014).

### FR7: Minutes on a topic

"15 minutes on" a topic plays the strongest clips about a topic from different episodes, back to back,
and says what it actually runs. It is reachable from Discover's "15 minutes on" row and from topic,
theme and storyline cards.

### FR7b: Learn X this week

A signed-in listener can start a seven-day path through a theme or storyline from Discover: one
step a day, each an episode and its best clip on the subject, oldest first. Discover shows "Day 3
of 7" and the step to do now; `/learn` shows the whole week and stops the path. A day opens on its
day and can be done late, never early. One path at a time. No reminders (operator 2026-10-11).

### FR8: Explore

Explore has its own place in the navigation. It holds Trends, each kind as a short list with
"Show more" paged from the server, and a taste of the catalogue: the newest four episodes or six
shows, with "See all" to the full lists (`/catalog`, `/shows`). Mine applies to Trends and Search;
the catalogue is everyone's, and its full lists carry a "Which episodes" / "Which shows" filter
(All shows, Following, Mine) for the listener's own.

### FR9: Search

Search leads Explore (operator 2026-10-11: looking for something is Explore's job; Discover is being
told what to hear next), so search is one tap from the Explore tab on a phone.

## Success Metrics

Measured in Umami (`discover_rail_shown`, `discover_rail_tap`, `capture_created.source`), per rail;
the reel, Learn and Explore by `reel_open`, `reel_action`, `learn_path`, `trends_show_more`,
`explore_see_all` and `catalog_filter` (RFC-126 §6):

- **Primary:** captures per tap, and the cross-show share of opens.
- **Secondary:** tap rate.
- **Floor:** five distinct participants before a rail is judged.

## Dependencies

- The GI/KG corpus with timed, speaker-attributed quotes; topic perspectives; storylines; the topic
  consensus enricher.
- Umami (usage analytics, under the listener's opt-out).
- The Moments picker (clip windows).

## Risks

- **Thin data.** Small corpora give short rails and short reels. Mitigation: rails hide when empty,
  and reels state their real length.
- **Misaligned quotes in the corpus.** Mitigation: the support rule (FR1) and the grounded
  validation corpus (ADR-165).
- **Performance at scale.** Rails are measured only on the 55-episode fixture so far.

## Open questions

- Whether "Less like this" should later fade or down-weight rather than remove (initial setup in
  ADR-163 §6: remove, both scopes, recommendation surfaces only; decided on the measured taps).
