# ADR-162: "Mine" means the listener's own world — one meaning per kind of item, one switch

- **Status**: Accepted
- **Date**: 2026-10-09
- **Authors**: Marko Dragoljevic
- **See Also**: [UXS-012](../uxs/UXS-012-consumer-home.md) (`TrendingScopeButton`)

## Context & Problem Statement

The player offers "Mine" in several places, and each had grown its own meaning and its own control:

- **Trends** (topics, themes, storylines, people) had a Mine ⇄ Everyone icon on the Trends row. Its
  "mine" was strictly the listener's own world (2026-10-07): followed, saved, or met in an episode
  they heard or captured from.
- **Trending shows** had no switch on Discover and always ranked everyone. The server's "mine" set
  held **no shows at all**, so `scope=mine` for shows was always empty.
- **Search** had its own "All / Mine" tabs, kept in the URL, where "mine" meant episodes heard or
  captured from only — not saved episodes, not the shows the listener follows.

Three controls, three definitions. A listener who flipped one could not predict the others.

## Decision

1. **"Mine" is the listener's own world**: what they follow, what is in their Library (saved), and
   what they listened to or captured from. It is defined **once per kind of item**:

   | Item | In "mine" when the listener… | Server |
   | --- | --- | --- |
   | Show | follows it, saved it, or heard / captured from / saved one of its episodes | `personal_show_ids` |
   | Episode (search) | heard or captured from it, saved it, or it belongs to a show they follow | `world_episode_set` |
   | Topic, person, theme, storyline (Trends) | follows it, saved it, or it appears in an episode they heard, captured from, or saved | `personal_entity_ids` |

2. **Following a show widens search, not Trends.** A followed show's episodes are the listener's to
   search; every topic such a show has ever covered is not "theirs". Counting those would make Mine
   indistinguishable from Everyone for anyone following a few busy shows, and cost one knowledge-
   graph load per episode of every followed show.
3. **Saving an episode counts like hearing it** for shows and Trends — but is kept out of
   `derived_interest_counts`, which also feeds the discover ranker: a bookmark is weaker evidence of
   taste than a listen, and the ranker's behaviour is not part of this decision.
4. **One switch.** Discover carries one Mine ⇄ Everyone switch in its header (`discover-scope`);
   trending shows, Trends and a search started on the page follow it. The Search results' switch
   reads and writes the **same** remembered choice (`useTrendingScope`, the `lp.trendingScope`
   preference, synced across devices). **Mine is the default** for a signed-in listener; signed out
   there is no Mine and the switch is hidden. An explicit `?scope=` in a search URL wins.
5. **An empty Mine says so** and offers everyone's (`discovery-mine-empty`,
   `trending-shows-mine-empty`, `search-mine-empty` → "Search every episode instead") — a section
   that vanishes while the switch is lit reads as broken, and with Mine the default, an empty Mine is
   a new listener's first search.

## Consequences

**Positive**

- One mental model: flip Mine anywhere and every "mine" surface agrees.
- Trending shows under Mine works at all (it was always empty).
- Search Mine finds what the listener follows and saved, not only what they already heard.

**Negative**

- Search Mine returns more than before for people who follow shows — results they used to get only
  under All. Accepted by the operator (2026-10-09).
- A fresh account sees empty-but-explained Mine sections by default until it follows, saves or
  listens to something.

**Neutral**

- Relational cards' `scope=mine` (a person or topic card scoped to the listener) still reads heard ∪
  captured episodes; this decision did not change it. A later change that does should extend the
  table above rather than define a fourth meaning.
- Each "mine" list is cached per scope on the device (`home.trendingshows.mine` / `.corpus`), so one
  never hydrates from or stands in for the other.

## Alternatives Considered

- **"For you" for trending shows** — everyone's rising shows boosted by the listener's interests,
  leaving out followed shows. Better for discovery, but a second meaning of "mine"; rejected in
  favour of one definition. Personalised discovery belongs to a section that says so, not to Mine.
- **A switch per section** (as before). Rejected: the operator asked for one page-level control.
- **Search Mine follows Discover only at launch** (independent afterwards). Rejected: the operator
  wanted the two synced.
