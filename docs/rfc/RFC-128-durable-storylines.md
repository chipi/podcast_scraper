# RFC-128: Durable Storylines — a storyline is a thing that exists, not the output of one run

- **Status**: Implemented (2026-10-14), not yet deployed
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

A **storyline registry** (carried as `data.registry` in `enrichments/topic_theme_clusters.json`)
holds every storyline ever created: `id`, `name`, `aliases`, `anchor`, `created_at`,
`last_active_at`, and the recency-weighted topic set. Lineage is `id_history` (`successors`,
`gone`); explicit `merged_into` / `split_from` records are not built (Open questions). The
clustering stays as it is and produces **proposals**: groups of topics discussed together this run.
A reconcile step matches proposals to registry entries and updates membership. It never deletes an
entry.

### 2. A lifecycle, not deletion

`emerging → active → quiet → archived`. A storyline with no new episode for **90 days** goes quiet; a
new episode on its topics makes it active again. "Crisis in Hormuz" is quiet between flare-ups and
is the same storyline at the next one. Followers keep it in every state; Discover ranks quiet ones
down, never out.

### 3. Sticky assignment, then a bar to create

A run first attaches episodes and topics to existing storylines (by overlap with the registry's
weighted topic set). Only what fits none seeds a **candidate**, and a candidate becomes a storyline
only past a bar, set by the replay below: **at least 2 topics and 5 episodes, proposed across at
least 28 days**. Today 45 of 99 storylines span fewer than 5 episodes. A replay with "3 topics"
kept prompt engineering (12 episodes, two topics) out: most real threads here are two topics
discussed together.

### 4. Membership drifts, identity stays

Topic strengths decay with recency, so over years "war in Ukraine" can change every topic it
started with and keep its id, its followers and its page.

### 5. A name that sticks

The name is chosen when the storyline is created and changes only when another label has led its
proposals for **60 days** running. The old name stays as an alias: search finds it, and the page can say
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

## Replay results (2026-10-14)

Production corpus, read-only copy: 2,635 served episodes, clustered month by month from 2024-11 to
2026-10 (episodes published up to each month). Faithfulness: today's clustering on the copy
reproduces production's 99 storylines member for member and id for id.

| | Today's rule | New rule, bar 2/5/2 | Bar 2/3/2 \* | Bar 3/5/2 \* |
| --- | --- | --- | --- | --- |
| Storylines at 2026-10 | 99 | 58 | 83 | 31 |
| Spanning fewer than 5 episodes | 45 | 1 | 24 | 0 |
| Same id, new name, during the history | 2 (24 storylines renamed by a new id, up to 4 times each) | 4 | 5 | 2 |
| A follow made 2026-04 is on the same storyline at 2026-10 | 25 of 34 | 19 of 19 | 25 of 25 | 12 of 12 |
| "Prompt engineering" | yes | yes (since 2026-04) | yes | no (a candidate) |
| Episodes in some storyline | 795 | 681 | 746 | 561 |
| Topics in two storylines; widest storyline | 0; 16 topics | 0; 16 topics | — | — |

\* Measured with the prototype, before the two fixes below; not re-run. The first two columns are
`scripts/eval/replay_storylines.py` on the copy, after them.

**Found on the copy after the first implementation, both fixed and now tested:**

- *Anchors moved.* The first run after deploy seeded every topic at the same strength, so 22 of 99
  storyline pages would have moved to another anchor topic. The registry now keeps each anchor.
- *Topics shared.* Attaching a proposal copied all its topics into the storyline, including ones
  another storyline held: by 2026-10, 25 topics sat in two storylines and some storylines widened
  (3 → 13 topics). Each topic is now published in one storyline only, as today's clustering does.

Today's rule renames as the corpus grows, back and forth: one 44-episode thread went "nominal gdp
targeting → inflation expectations → nominal gdp targeting → federal reserve policy → inflation
expectations". Under the registry it keeps one id and renames once. Chosen: **2/5/2**.

**Coverage is limited by topics, not by this rule.** 30% of episodes belong to any storyline today:
22,884 topics, 92% of them on one episode only, and 70% of episodes have no topic pair seen
together in two or more episodes, which is exactly the set in no storyline. Tracked in #2351.

## Implementation (2026-10-14)

- `search/storyline_registry.py` — `reconcile(proposals, prior, …, now)`: attach at a 0.5 topic
  share, topic strength ×0.85 per 30 days (held at 0.2), bar 2 topics / 5 episodes / 28 days,
  rename after 60 days, quiet after 90 days. The anchor a page routes by is kept in the
  registry (the pre-registry anchor, or the most-episode topic at creation) and moves only when its
  topic leaves; the member is marked `"anchor": true`. Each topic is published in one storyline
  only (the one it anchors, else the one where it is strongest); a storyline whose topics all
  faded into others is recorded as their successor in `id_history`, for the reference pass.
- `search/storyline_history.py` — the one-time history seed (Migration).
- `enrichers/topic_theme_clusters.py` (1.3.0) — this run's clusters are proposals; the registry is
  carried in `topic_theme_clusters.json` (`registry`, plus `state`, `aliases`, `created_at`,
  `last_active_at` per storyline, and each member's `strength`); super-themes are computed over the
  registry's storylines. Each member's `lift_to_cluster` keeps its meaning: its co-occurrence lift
  to the other members published.
- **The registry cannot be lost silently.** The executor writes every envelope atomically; the
  enricher also keeps the registry's last-good copy (`enrichments/storyline_registry.json`) and
  continues from it if the artifact is unreadable. Each run logs one line,
  `storylines: registry health=ok|bad source=… storylines=… quiet=… shared_topics=… vanished=…`
  (also `registry_health` in the artifact); `bad` (the copy was used, the registry started over, a
  topic in two storylines, a storyline gone with no successor or gone record) logs at ERROR and
  fires the alert `enrich-common-storyline-registry-bad`.
- `search/storylines.py` — quiet storylines are offered after active ones on the Home rail. The
  other surfaces need no rule: momentum and trending already fall with no new episode, and a
  storyline's page, its 15 minutes, an episode's or a show's storylines, and search keep showing a
  quiet one, which is the point.
- **Shown to listeners:** the storyline page says "Also called …" (its aliases) and "Quiet since
  …"; searching an old name opens the storyline under its current one (entity resolution and the
  search index); the card flags the anchor the page is routed by.
- `scripts/eval/replay_storylines.py` — the replay above, driving the production enricher (old
  rule: each month from scratch; new rule: each month reconciled against the previous one).

## Migration

Operator, 2026-10-14: build the registry from history, and keep every storyline published today.

`python -m podcast_scraper.search.storyline_history --corpus <root> [--apply | --undo]`, once,
after the deploy: it replays the registry month by month over the corpus (no model; seconds on
production),
then merges the result with the storylines published today. A replayed storyline that shares half
of its topics, or of a current storyline's, with a current one folds into it: the current one takes
the replayed history (created, last active, topic strengths; a different replayed name becomes an
alias) and keeps its own id, name and anchor. A current storyline with no replayed match is dated
by its episodes; a replayed one with no current match is added. It never writes the artifact:
`--apply` writes a seed (`enrichments/storyline_registry.seed.json`, its id the hash of the
registry) that the next enricher run takes in, so it cannot race a run, and the same seed twice is
a no-op. `--undo` withdraws a seed not yet taken, or seeds back the registry a taken one replaced.
Dry run by default.

Simulated on the production copy, seed then two nightly runs (`health=ok`, `source=seed` then
`artifact`): 99 of 99 ids, names and anchors unchanged, and the card flags the routed anchor on all
99, so no page, follow or save moves; 17 quiet; no topic in two storylines; all 59 replayed
storylines folded in (none added); 6 storylines trade a few topics with each other; 47 candidates
carried on, so new storylines do not wait four weeks from deploy.

Without the seed (a corpus's first run), the registry starts from the previous artifact's
storylines, with today's ids, names and anchors, and, on a corpus with none, from its own clusters.
#2345's `id_history` maps every id the registry retires, so stored references keep resolving. No
user state is rewritten; the reference pass keeps running.

## Not in scope

- Themes (`tc:`): they stay a per-run semantic grouping.
- Editorial curation of storylines (naming by hand, pinning); possible later on top of the registry.

## Open questions

- Matching proposals to storylines by THEME (semantic grouping) rather than exact topic ids, so an
  episode on "risk naming" can join "risk management" (#2351).
- Automatic merges of two storylines that keep proposing together (v1 folds a proposal into its
  best match and leaves the other to go quiet).
- Whether a quiet storyline is offered to new listeners at all, or only kept for followers.
- How a child storyline appears on its parent's page.
