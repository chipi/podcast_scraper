# RFC-128: Durable Storylines — a storyline is a thing that exists, not the output of one run

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 5.5)
- **Stakeholders**: Consumer App, Enrichment (corpus clustering), Server API
- **Related RFCs**:
  - `docs/rfc/RFC-127-board-summaries.md` (the storyline summaries, #2344, ride on the same sidecar)
- **Related ADRs**:
  - `docs/adr/ADR-163-the-jobs-of-discover.md` (storylines as followable subjects)
- **Issue**: #2349
- **Related work**: #2345 (stable ids and reference repair, the stopgap this replaces), #2344

## Abstract

A storyline today is recomputed from scratch on every enrichment run and keeps only what that run's
numbers support. Its name is whichever member topic ranks highest that day, and it vanishes when
its topics stop co-occurring often enough. A listener who follows "prompt engineering" can find it
renamed to "agentic engineering" next week, or gone. This RFC makes a storyline a **durable record**
— an id, a name, a lifecycle and a lineage — that the clustering only **proposes membership for**
and never creates or deletes on a whim. It is proven before it ships by **replaying the clustering
month by month over the real production history** and measuring how storylines live and die under
the old and the new rule.

## Problem Statement

Operator, 2026-10-14: "I really don't like this idea of storylines disappearing … if I decided to
follow prompt engineering, I want that to be the basis for all kinds of recommendation … how would
this work in a news world, like crisis in Hormuz, or war in Ukraine … that storyline might change
over time, but it's not going to disappear."

Measured on production (read-only):

- 99 storylines; 18 span only 2 episodes, 13 span 3, 12 span 4 (2026-10-13).
- Over three days of snapshots, 10 storylines vanished: 3 renamed as they grew, 3 absorbed into
  others, 4 dissolved two-topic pairs (2026-10-13).
- 3 of the 6 accounts follow 18 themes and storylines that a re-clustering renamed, so the follows
  point at nothing (2026-10-14).

The stopgap in #2345 keeps an id when at least half the members stay and maps the rest to successors. That repairs
references after the fact, but the storyline still disappears, changes its name and loses its
followers whenever one run's numbers say so.

**Themes are not storylines.** A theme (`tc:`) groups topics that mean similar things (embedding
similarity); it is fine for themes to regroup as the vocabulary grows. A storyline (`thc:`) groups
topics discussed together in the same episodes (co-occurrence lift) — the shape of a news thread.
This RFC is about storylines only.

## Proposal

### 1. Identity separate from membership

A **storyline registry** (`enrichments/storyline_registry.json`) holds every storyline ever created:
`id`, `name`, `aliases`, `state`, `created_at`, the weighted topic set (each topic with first and
last seen and a recency-weighted strength), and `lineage` (`merged_into`, `split_from`). The
clustering stays as it is and produces **proposals**: groups of topics discussed together this run.
A reconcile step matches proposals to registry entries and updates membership. It never deletes an
entry.

### 2. A lifecycle, not deletion

`emerging → active → quiet → archived`. A storyline with no new episode for N weeks goes quiet; a
new episode on its topics makes it active again. "Crisis in Hormuz" is quiet between flare-ups and
is the same storyline at the next one. Followers keep it in every state; Discover ranks quiet ones
down, never out.

### 3. Sticky assignment, then a high bar to create

A run first attaches episodes and topics to existing storylines (by overlap with the registry's
weighted topic set). Only what fits none seeds a **candidate**, and a candidate becomes a storyline
only past a bar: proposed values, to be set by the replay — at least 3 topics, 5 episodes, recurring
across at least 2 weeks. Today 43 of 99 storylines span fewer than 5 episodes.

### 4. Membership drifts, identity stays

Topic strengths decay with recency, so over years "war in Ukraine" can change every topic it
started with and keep its id, its followers and its page.

### 5. A name that sticks

The name is chosen when the storyline is created and changes only past a higher bar (a sustained
shift over several runs). The old name stays as an alias: search finds it, and the page can say
"now also: agentic engineering". When a new thread grows its own momentum, it becomes a child
storyline linked to its parent instead of a rename.

### 6. Merges and splits keep lineage

Two storylines merge only when they overlap heavily over several consecutive runs; the absorbed one
redirects and its followers follow the survivor. A split creates children; the follow goes to the
main continuation, and the listener is offered the others.

### 7. What a follow means

Following a storyline follows the registry entry. Recommendations, reels, Learn paths and summaries
use its current membership, so a follow never silently stops meaning anything.

## Proof: the replay

Before any of this ships (the `replay-proven-fix` procedure):

1. **Corpus by date.** From the production corpus (read-only copy), for every month in its history,
   the episodes published up to that month.
2. **Old rule.** Run today's clustering on each month's corpus, as a nightly would have.
3. **New rule.** Run the clustering plus the reconcile against a registry carried month to month.
4. **Measure, old against new:**
   - storylines alive after 3, 6 and 12 months; how many vanish, and why (renamed, absorbed,
     dissolved);
   - name changes per storyline per year;
   - storylines created that span 2 episodes;
   - for a simulated follower of each storyline at each month, where the follow ends up 6 months
     later (same storyline, a successor, nothing);
   - named threads the operator chooses (e.g. "prompt engineering", "AI and the economy"): their
     id and name over time.
5. **Faithfulness.** The replay's last month reproduces today's production storylines under the old
   rule; a mutation (break the reconcile) must move the numbers.

The thresholds in §2, §3 and §5 are set from this replay, not guessed.

## Migration

The registry is seeded from today's storylines (ids kept). #2345's `id_history` maps every id the
registry ever retired, so stored references keep resolving. No user state is rewritten by this
change; the reference pass keeps running.

## Not in scope

- Themes (`tc:`): they stay a per-run semantic grouping.
- Editorial curation of storylines (naming by hand, pinning); possible later on top of the registry.

## Open questions

- The quiet-period N and the creation bar: from the replay.
- Whether a quiet storyline is offered to new listeners at all, or only kept for followers.
- How a child storyline appears on its parent's page.
