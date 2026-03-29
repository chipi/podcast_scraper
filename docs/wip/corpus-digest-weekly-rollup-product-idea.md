# Corpus digest & weekly rollup — product shaping idea (WIP)

**Status:** Idea notebook — **not** a committed PRD/RFC. Intended for product shaping **on top of**
a stable **core** pipeline (transcripts, metadata, summarization, optional GIL/KG, optional DB
projection).

**Audience:** Maintainers and future PRD/RFC authors when prioritizing “many podcasts, one brain”
workflows.

---

## Problem sketch

Someone subscribes to **10–50 podcasts** and runs `podcast_scraper` regularly. They want to:

- **Navigate** what arrived recently without opening every episode.
- **Consume** quickly (skim) and **dig deep** only where it matters.
- Answer: **“What happened last week across my library?”** without duplicate noise and without
  losing trust when a claim matters.

Core artifacts today are **per episode** (transcript, metadata, summary, optional `gi.json`, future
KG JSON). There is no **first-class, time-scoped, cross-feed digest** contract yet.

---

## Layering on top of core (mental model)

This aligns with the guides’ “stack” narrative:

| Layer | Role | Primary home (today / planned) |
| --- | --- | --- |
| **Summaries** | Fast **consumption** per episode | PRD-005, metadata |
| **KG** | **Navigation** across episodes (entities, themes, links) | PRD-019, RFC-055/056 |
| **GIL** | **Value and trust** — takeaways + optional verbatim evidence | PRD-017, RFC-049/050 |

**Dependency:** Digest features should assume these layers are **stable and versioned** before
building opinionated rollups (schema churn would invalidate weekly reports).

---

## Gaps to close later (product opportunities)

These are **not** promises — they are **shaping axes** for a future PRD/RFC set.

1. **Time-scoped aggregation** — Define “window” (e.g. calendar week, last N days) across **all**
   processed episodes in a corpus or run family; emit **one** digest artifact or query view.

2. **Cross-feed inbox semantics** — “New since last successful run” / per-feed watermarks so users
   reason about **backlog vs delta**, not only folder-per-run.

3. **Story clustering & dedup** — Reduce fatigue when many shows cover the **same** news cycle
   (cluster titles, entities, or embeddings — TBD).

4. **Ranking & time budgets** — Optional “30 minutes this week” views using length, recency,
   novelty, and (later) user interests.

5. **Change detection** — “What’s new on topic X **this week**” vs evergreen KG rollups (delta vs
   cumulative).

6. **Digest output contract** — A small, versioned **document or JSON** shape: sections (e.g.
   headlines by theme, GI-backed bullets, links to episodes). Glue layer above per-episode files.

7. **Presentation** — CLI/HTML/email/Obsidian export is **out of core** unless explicitly scoped;
   the **contract** matters first so integrators can render.

8. **Personalization** — Watchlists, saved entities/topics — complements KG extraction but is
   **user config**, not only graph structure.

---

## Suggested sequencing (when core is stable)

1. **Stable inputs** — Published dates and universal episode identity (e.g. ADR-007) reliable in
   metadata; summaries + optional GI/KG present where flags allow.

2. **Queryable corpus** — PRD-018 / RFC-051 or documented file-scan patterns so “last week” is not
   hand-rolled ad hoc for every user.

3. **Digest v0** — Narrow scope: time filter + sorted list of episodes with summary lead + link;
   no clustering.

4. **Digest v1** — Add theme/entity rollups (KG-assisted), optional GI-highlighted “verified
   bullets,” dedup iteration.

---

## Non-goals (for early digest iterations)

- Replacing **listening** or primary sources for high-stakes decisions.
- Full **recommender system** or social graph.
- **Merging** GIL into KG artifacts (keep contracts separate; join at digest/query layer if
  needed).

---

## Related documents

- [Grounded Insights Guide](../guides/GROUNDED_INSIGHTS_GUIDE.md) — summaries / KG / GI stack
- [Knowledge Graph Guide](../guides/KNOWLEDGE_GRAPH_GUIDE.md) — same stack from KG side
- [PRD-005: Episode summarization](../prd/PRD-005-episode-summarization.md)
- [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md)
- [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md)
- [PRD-018: Database projection](../prd/PRD-018-database-projection-gil-kg.md)
- [ADR-007: Universal episode identity](../adr/ADR-007-universal-episode-identity.md)

---

## Promotion path

When this idea hardens: move content into a **PRD** (what/why) and optionally an **RFC** (digest
schema, CLI `digest` namespace, or export-only tool). Until then, treat this file as **WIP**
notes only.
