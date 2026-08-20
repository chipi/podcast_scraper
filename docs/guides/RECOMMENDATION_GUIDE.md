# How we decide what you should listen to next

This guide explains the whole recommendation system: what it recommends, how it decides, why each
part exists, and what it deliberately does not do. It assumes no prior knowledge of the codebase.

If you only read one thing, read [The formula](#the-formula) and
[What each signal is for](#what-each-signal-is-for).

To *see* it work rather than read about it, run:

```bash
python scripts/eval/score/rank_scenarios_v1.py
```

That prints a table of what a given person's feed looks like under each signal, on a corpus built
to make the differences visible. Every claim in this guide is checkable against it.

---

## The three questions

The product answers three different "what next?" questions, and they are not the same question.
Confusing them is the single most common source of bugs in this area.

| Surface | Question | Draws from |
| --- | --- | --- |
| **Discover** | "What should I listen to *next*?" | The whole corpus, ranked |
| **Your Week** / digest email | "What happened since I was last here?" | Your own captures + follows |
| **Revisit** | "What did I already learn that I should see again?" | Only your own captures |

Discover looks outward at episodes you have not heard. The other two look inward at what you
already did. Everything below is about **Discover** unless a section says otherwise.

---

## The pipeline, end to end

```text
  corpus on disk
        │        pipeline output: metadata + transcripts + KG + GI per episode
        ▼
  catalog rows                       corpus_catalog.build_catalog_rows_cumulative
        │        one row per episode: title, publish date, which artifacts exist
        ▼
  candidate pool                     app_discover_view.build_discover_pool
        │        the newest 4 × limit episodes, PLUS any episode matching an interest
        ▼
  scoring                            app_discover_view.rank_discover
        │        one number per candidate (the formula below)
        ▼
  the feed                           top `limit`, highest score first
```

Two things are worth pausing on.

**The pool is not the corpus.** Scoring runs over a bounded candidate set, not every episode you
could theoretically be shown. The bound is recency-based, so without the interest union a
well-matching but older episode could never surface however high it scored — it would never be a
candidate in the first place. That union is why a niche follow can pull a nine-day-old episode to
the top of the feed.

**The catalog is rebuilt per request.** There is no persisted index. That is why the pool has to
stay small, and it is the main scaling constraint in this area today.

---

## The formula

For each candidate episode:

```text
    score = (significance / feed_mean) × (1 + affinity + trend + recency)
```

Read it as: **a base worth, normalised against its own show, multiplied by how much this
particular person should care.**

Written out:

```text
    base        = significance(episode)              # how much we know about this episode
    normalised  = base / mean(significance of that FEED's episodes)
    affinity    = w_a × (1 − 0.5 ^ (explicit + 0.5 × derived))
    trend       = w_t × (min(topic_velocity, cap) − 1)
    recency     = w_r × 2 ^ (−age_days / half_life)

    score       = normalised × (1 + affinity + trend + recency)
```

Every term is multiplicative on a base of 1, so a signal that does not apply contributes exactly
nothing rather than dragging the score toward zero.

---

## What each signal is for

### Significance — "how much do we know about this episode?"

`+2` if the episode has grounded insights, `+1` if it has a knowledge graph, `+0.2` per summary
bullet up to five.

**This measures pipeline coverage, not content quality.** An episode is not better because our
enrichment ran well on it. Left raw, that is a real bias: a thin, perfectly relevant episode from a
sparsely-processed show loses to a richly-enriched but irrelevant one.

**The fix is the `/ feed_mean` division.** Each episode is compared against *its own show's*
average, so a show we happen to enrich poorly is not punished for it — only within-show variation
survives. You can watch this: in the scenario table, the `follows the sparse show` persona gets the
least-enriched feed in the corpus at the top of their feed, because they follow it.

### Interest affinity — "did they ask for this?"

The dominant personalisation signal, and the one with the most history.

It used to be `matched / len(interests)` — a **ratio**. Following a second topic halved the boost
the first one earned, so the system rewarded staying narrow and punished exactly the engagement it
exists to encourage. Somebody who followed ten things got a tenth of the personalisation of
somebody who followed one.

It is now a **saturating** curve: `1 − 0.5^contribution`. Each additional match adds strictly less
than the one before (1 match → 0.5, 2 → 0.75, 3 → 0.875), so a broad-interest user cannot swamp the
base signal — but no match ever *reduces* another's contribution.

Two kinds of interest feed it:

* **Explicit** — what you followed in the picker or from an entity card.
* **Derived** — inferred from what you have heard and captured. These enter at **half weight**
  (`derived_ratio: 0.5`), so an inference can raise an episode but never outvote something you
  actually asked for.

**Not every entity is followable, deliberately.** A person id that identifies somebody only
*inside one episode* is excluded from both kinds — an unresolved diarization voice
(`person:speaker-{episode}-{n}`, #1b) and a bare first name with no surname anywhere in that
episode (`person:unresolved-{name}-{episode}`, #1685). Both are filtered by
`is_unresolved_speaker_placeholder` inside `entities_from_kg`, which is the single source for
entity cards, ranking rows **and** derived interests — so they cannot be followed by hand or
minted into a profile by listening.

This matters for affinity specifically. Before #1685 these were minted as GLOBAL ids, so
`person:jensen` pooled every Jensen in the corpus into one followable token. Production measured
**208 occurrences of 172 such ids**, of which **196 had no full name anywhere in their episode** —
hollow tokens that either lead nowhere or, worse, attach one person's statements to another's
name. Following one added affinity to an incoherent set of episodes. Where the episode *does*
contain exactly one matching full name (12 of the 208), the reference is healed to the real
person instead, so the mention strengthens that person's signal rather than splitting off from it
(`bare_name_heal`, on by default).

### Recency — "is this current?"

A graded boost decaying with a **730-day half-life**, measured from the newest episode in the pool
rather than from wall-clock now.

Before this it was only a tie-break, which meant any non-empty interest set sorted the entire pool
by score — following one topic reshuffled even the 90% of the feed that had nothing to do with it,
and "newest first" quietly became "best-enriched first".

A 30-day half-life was tried first and was **completely inert** on the validation corpus: its
925-day span meant the second-newest episode already scored 0.014, so every candidate but the
newest was flattened to nothing.

**What the half-life encodes: when podcast content goes stale — not how big the corpus is.**
That distinction is the whole point. 365 was fitted to the corpus we had at the time, which means
it needs re-tuning every time the archive deepens. The target content window is 2-4 years, with a
tail out to about a decade:

| age | 365d | **730d** | 1095d |
| --- | --- | --- | --- |
| 1 year | 0.50 | **0.71** | 0.79 |
| 2 years | 0.25 | **0.50** | 0.63 |
| 4 years | 0.06 | **0.25** | 0.40 |
| 10 years | 0.00 | **0.03** | 0.10 |

At 365 a four-year-old episode scores 0.06 — effectively excluded from the freshness signal while
sitting inside the window we care about. At 730 it scores 0.25: still competing, clearly aged. A
two-year episode keeps half its freshness. 1095 was rejected as too generous; a decade-old episode
should win on relevance or not at all.

The personalisation cost is about one point: 730d measured 97.2% vs 365d's 98.4% on the same
corpus, for the "does a follow flatten the feed's sense of time" check.

**Keep the scale in mind.** Recency's entire range is worth ~0.5 in the score, while a single
followed interest is worth 3.0. This tunes the shelf for someone with no strong interests; it does
not drive a personalised feed.

### Trend velocity — "is this heating up?"

Off by default. It reads a topic-momentum enrichment and boosts episodes on topics that are
accelerating. It stays off because it has never been tuned against real engagement — the same
reason the whole personalisation path shipped behind a flag.

---

## Interests: explicit, derived, and forgetting

Derived interests are the people and topics that recur across the episodes you have heard or
captured — ranked, top-8, feeding Discover exactly as an explicit follow does. That is what lets
personalisation work for somebody who never opens the picker.

Two properties matter and both were bugs first:

**It reads your most RECENT engagement, not an alphabetical slice.** The selection used to be
`sorted(slugs)[:40]`, and slugs are `{feed-slug}-{hash}` — so the sort grouped by show, and past 40
episodes the profile froze on whichever shows happened to be spelled first. New listening stopped
moving it at all.

**It can forget.** Each occurrence now decays with a **90-day half-life**, aged from your own most
recent engagement. Without that it was a pure accumulator with no term that could shrink: somebody
whose taste had moved on kept being recommended the taste they left, and the more they had listened
beforehand, the longer the escape took.

The 90 days is the largest half-life that still fully recovers a taste shift — measured across
30/60/90/180/365 on a user with twelve old episodes and four recent ones. Shorter buys nothing on
that case and costs a light listener dearly: at 30 days, somebody who hears one episode a week has
their oldest engagement weighted `0.002`, which is deletion, not decay.

`count` in the API stays the plain episode tally, because that is what the UI says out loud; the
new `weight` field is what the order is actually built from.

---

## What a brand-new user sees

**Pure recency, and no signal can change it.**

`rank_discover` returns the pool unscored when the interest set is empty. The entire ranking
apparatus is dormant until the first follow. This is visible in the scenario table as an entire
block of `= unchanged` rows, and it is a product fact rather than a tuning accident: with nothing
to personalise against, "newest first" is the honest answer.

---

## The other two surfaces

### Revisit — spaced repetition over your own captures

Captures come back on a ladder: **2 days → 7 → 30 → 90**. A highlight is due when
`now − last_seen ≥ ladder[times_shown]`.

Three surfaces show revisit content — the Revisit tab, the Your Week card, and the digest email —
and all three now apply **the same graph gate**: a capture whose episode has no knowledge graph is
withheld everywhere. Previously the tab showed captures the other two silently dropped, which read
as a bug and made an empty Your Week impossible to explain from inside the app.

An episode without a knowledge graph is a **pipeline defect**, not a normal state. It is logged at
runtime and fails corpus validation at build time.

**Arriving at the player with `?revisit=<id>` advances the ladder.** All three surfaces carry that
marker, so consuming revisit through the email advances your schedule exactly as using the inbox
does. Before that, the only thing in the entire product that advanced a ladder was the inbox's
dismiss button — so somebody who genuinely went back and listened never progressed, and the digest
re-sent the same five items indefinitely.

### Your Week and the digest email are the same payload

Both are built by one assembler. The email is a **reminder of the page you would see anyway**, so
they cannot diverge; the in-app version adds artwork and a backfilled title for rendering, but the
items are the same items.

---

## Current tuning

| Signal | Enabled | Weight | Parameters |
| --- | --- | --- | --- |
| `significance` | yes | 1.0 | `gi_bonus 2.0`, `kg_bonus 1.0`, `bullet_step 0.2`, `bullet_cap 5` |
| `interest_affinity` | yes | 4.0 | `derived_ratio 0.5`, `cap 1.0` |
| `recency` | yes | 0.5 | `half_life_days 730` |

And the admission policy, which is in the same config but is not a scoring signal — it decides
which candidates enter ranking at all:

| signal | params |
| --- | --- |
| `discover_pool` | `corpus_share 0.15`, `page_multiple 4`, `max_candidates 400`, `min_limit_for_share 5` |

**Why it lives with the weights.** Every signal above re-orders the candidates; this one decides
who is in the room. No weight can promote an episode the pool excluded, which makes admission the
most consequential parameter here — and it was a module constant nothing could override until
2026-08-19, which is how it stayed a fixed 48 episodes while the corpus grew to 678. A tuning
sweep has to be able to vary it.

`corpus_share` 0.15 is a judgement call rather than a measurement, chosen against a corpus
expected to grow by an order of magnitude.
| `trend_velocity` | **no** | 0.4 | `cap 1.5` |

Affinity's weight is 4.0 rather than 2.0 because saturation makes one match worth `0.5` of the
weight — so 4.0 restores a single matched interest to exactly the ×2 boost it has always had. The
number changed; the behaviour did not.

Operators can retune all of this at runtime via `PUT /api/app/ranking-config`. The offline eval
scores **the stored config**, not the shipped default, so a deployment that has retuned is measured
on what it actually runs.

---

## How we know it works

Two mechanisms, doing different jobs.

**The gate** — `scripts/eval/score/rank_discover_v1.py` scores seeded personas against gold
"relevant shows" and reports nDCG@10 against a recency baseline. It fails the build if
personalisation does not measurably beat plain recency. It answers *did we break it?*

**The observer** — `scripts/eval/score/rank_scenarios_v1.py` prints what each signal does on a
corpus built for the purpose. It answers *what does it actually do?*

The observer exists because the realism fixture cannot answer that second question — and that is
not a flaw in it. Realism means keeping the confounds in; observation means removing every confound
but one. On the validation corpus, three separate ranking questions turned out to be
**undiscriminable**: the recency half-life (its 925-day span made 14/30/90 days identical), the
coverage bias (uniform enrichment makes it invisible by construction), and a ranking order whose
count-order happened to coincide with alphabetical order.

---

## What this system deliberately does NOT do

* **No diversification.** The loop is `heard → derived interests → ranked higher → heard`. Decay
  bounds how long a stale interest persists; nothing actively widens the profile. Filter-bubble
  risk is low today by accident of bounds (top-8 interests, a recency-capped pool, ranking that
  re-orders but never filters) rather than by design.
* **No collaborative filtering.** Nothing uses other users' behaviour. Every signal is derived from
  this user and this corpus.
* **No LLM at request time.** Ranking is arithmetic over pre-computed artifacts.
* **No engagement feedback.** Impressions and clicks are recorded but read by nothing. When either
  starts feeding ranking, the absence of diversification stops being theoretical.

---

## Where the code lives

| Concern | File |
| --- | --- |
| Scoring, pool, all signal maths | `src/podcast_scraper/server/app_discover_view.py` |
| Signal registry + shipped tuning | `src/podcast_scraper/server/app_ranking_config.py` |
| Derived interests (the one definition) | `src/podcast_scraper/server/app_user_corpus.py` |
| Spaced-repetition ladder | `src/podcast_scraper/server/app_resurfacing.py` |
| Your Week / digest assembly | `src/podcast_scraper/server/app_digest_personal.py` |
| The gate | `scripts/eval/score/rank_discover_v1.py` |
| The observer | `scripts/eval/score/rank_scenarios_v1.py` |
