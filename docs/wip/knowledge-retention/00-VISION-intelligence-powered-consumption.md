# VISION — Intelligence-Powered Information Consumption

- **Status**: Draft — the north star for the Knowledge Retention package
- **Layer**: internal (strategic; states the mission and the bet)
- **Read**: first, before the handoff index. Everything else in the package is *what* and
  *how*; this is *why*.

---

## The north star

**Turn information consumption from something ephemeral into something that compounds.**
Today you listen, you absorb a little, and it's gone. We want every hour of listening to
deposit into a **living, personal model of the world that stays true as the world moves** —
so that after 50 or 100 podcasts you don't have a blur and a graveyard of bookmarks, you have
a sharper, current, navigable understanding that makes the *next* hour worth more than the
last.

Consumption that builds an asset, not consumption that passes through.

---

## The problem we're attacking

Knowledge from media dies three ways, and every existing tool fights only the first:

1. **Decay** — you forget it.
2. **Staleness** — the world moved; what you retained is now wrong or incomplete.
3. **Isolation** — it never connected to anything, so it can't be retrieved or built on.

"Second brain" products are ~90% decay. But for understanding the world through media, the
real killer is **staleness** — and beneath it, **narrative capture**: positions are
constructed, contested, and quietly revised, and a static memory of "what someone said once"
becomes a frozen, often misleading snapshot. The deeper purpose of this product is
**objectivization** — helping a person see through narratives by exposing how they're built,
where they conflict, and how they shift over time. Retention and objectivization are the same
mission at two scales: keep an individual's model true, by keeping it honest about conflict
and change.

---

## The bet (the one idea everything follows from)

**Move the unit of value from the *episode* to a *persistent personal knowledge state*.**

Every consumption product treats the episode as the unit — it streams, you consume it, it's
gone, and a save button just decorates the flow. We make the unit a knowledge state that
episodes *write into*. Once you do that, two goals that normally fight on a roadmap —
**remembering** what mattered and **tracking** how it evolves — become the same thing: recall
is *reading* the state, tracking is *keeping it current*.

---

## How it works (at altitude — specs elsewhere)

- **Three layers.** A shared, evolving corpus (**L0**); a thin per-user overlay of typed,
  weighted **pointers** into it (**L1**) — *no copied content, just "this mattered to me,
  here"*; and a personal subgraph (**L2**) **derived live** from the two. Because the personal
  layer stores pointers, not copies, "keep it alive" is free — a pointer always resolves to
  the current corpus. **L2 is never stale by construction;** only your *awareness* of it goes
  stale, which is exactly what the product repairs.
- **Five operations** on that state: **deposit, connect, resurface, reconcile, advise.**
- **A flywheel.** Each deposit densifies the subgraph → recommendations steer the next listen
  toward it → that listen deposits more → reconciliation keeps it true. The episode becomes an
  *input that makes the state more valuable*. That is "lasting value, not one-time-and-gone,"
  expressed as a loop.

---

## Why this is ours to win

The differentiated value lives where nobody else can follow: **keeping the model true**
(staleness) and **feeding it conflict and divergence** (objectivization). Both require what we
already have and a generic tool does not — **canonical identity, cross-show synthesis, and
conflict/evolution detection over a live corpus.** We're not building a memory app; we're
scoping corpus-level intelligence down to one person's retained slice. A bookmark of a string
can be resurfaced; only a *typed pointer into a live graph* can be reconciled. That single
architectural choice is the moat.

---

## The arc of ambition

Podcast is the **first domain**, not the destination. The same machine — retain, keep true,
expose conflict — generalises into a **media / journalism objectivization** layer, and beyond
that into a **generic intelligence platform with industry verticals.** Media is the long-term
passion; the retention model is how we earn the right to build it, by proving the core loop on
a domain we can own end-to-end.

---

## Non-negotiables (violate these and it becomes a lesser, different product)

1. **The personal layer holds pointers, never content.** The moment it stores copies, it goes
   stale and becomes a graveyard.
2. **L2 is derived, never stored as truth.** Live view over a live corpus.
3. **Smallness is the value.** Capture is intentional and decays; a hoard that resurfaces junk
   is failure, not completeness.
4. **Faithfulness over cleverness.** Every insight grounds back to source; resolution accuracy
   beats sophistication.
5. **Advice is a projection of *your* model, not a popularity filter** — and it surfaces what
   *challenges* you, not just what flatters you. The anti-filter-bubble is the point.
6. **Divergence is measured from content, never imposed from labels.** An objectivization
   platform that imports ideological bias into its own corpus would betray its reason to exist.
7. **The corpus is curated for shape, not size** — overlap *and* divergence — because value is
   gated on shape.

---

## What success looks like

- **Felt:** a user opens their map after 80 episodes and recognises it as *their mind* — and
  trusts it, because it tells them when what they believed got contradicted.
- **Measurable:** the flywheel compounds (subgraph density grows faster per episode with the
  loop on than off); the working set stays sharp, not bloated; reconciliation surfaces real
  conflict, not noise. All provable *before real users* via simulation.
- **Mission:** the product measurably makes a person harder to mislead — they see narratives
  forming, conflicting, and shifting, instead of absorbing whichever they heard last.

---

## How this package serves the vision (closing the loop)

| Vision element | Realised by |
|---|---|
| The model, the flywheel, the surfaces | **PRD-034** (product) + **RFC-081** (substrate) + **UXS** |
| "Keep it true" at scale, simply | **RFC-081** reconciliation inversion + two-dial abstraction |
| Build it without losing the contracts | **PLAN** (milestones, the invariant + R1 + cost tests) |
| See, debug, and tune the invisible machine | **PRD-035 / RFC-086** operator control room |
| Prove the value **without real users** | **PRD-036 / RFC-087** simulation harness |
| Acquire the **right-shaped** corpus the value depends on | **PRD-037 / RFC-088** corpus scout |

Read top-to-bottom, the package is one intent: **make consumption compound into a true,
personal, navigable model of the world — and prove it, tune it, and feed it before a single
real user arrives.**
