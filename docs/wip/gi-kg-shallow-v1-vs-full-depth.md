# GI and KG: shallow v1 vs full depth (WIP note)

**Purpose:** Single place to describe what the **current implementation** delivers
(“shallow v1”) versus what **full depth** would mean per PRD/RFC, for **both**
Grounded Insights (`gi` / GIL) and Knowledge Graph (`kg` / KG).

**Audience:** Implementers, reviewers, and anyone calibrating roadmap vs specs.

**Post–v1 capability backlog (depth work):** [GitHub #466](https://github.com/chipi/podcast_scraper/issues/466)
(NL-style consumption, `kg query` IR, richer aggregation, entity resolution, RFC-053, etc.).

**Normative specs (read these first):**

- GIL: [PRD-017](../prd/PRD-017-grounded-insight-layer.md), [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md), [RFC-050](../rfc/RFC-050-grounded-insight-layer-use-cases.md)
- KG: [PRD-019](../prd/PRD-019-knowledge-graph-layer.md), [RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md), [RFC-056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md)
- Optional relational serving (both): [PRD-018](../prd/PRD-018-database-projection-gil-kg.md), [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md)
- **GIL v1 / #460 (operator record):** [GROUNDED_INSIGHTS_GUIDE § Recorded product decisions (v1, issue 460)](../guides/GROUNDED_INSIGHTS_GUIDE.md#recorded-product-decisions-v1-issue-460)
- **KG v1 (operator record):** [KNOWLEDGE_GRAPH_GUIDE § Recorded product decisions (v1, KG shallow)](../guides/KNOWLEDGE_GRAPH_GUIDE.md#recorded-product-decisions-v1-kg)

---

## Shared framing

| Idea | Shallow v1 (today) | Full depth (later) |
| ---- | ------------------ | ------------------ |
| **Storage** | Per-episode artifacts under a run tree (`*.gi.json`, `*.kg.json`) | Same files **plus** optional Postgres projection (PRD-018 / RFC-051) for SQL / analytics |
| **Consumption** | CLI + scripts scan directories, merge/export locally | Same **plus** services that query DB, cache, or API layers |
| **Identity** | Episode-local or run-scoped IDs | Stronger cross-corpus entity/topic resolution, optional global registries |
| **NL queries** | **`kg`:** no NL consumption in v1 (RFC-056). **`gi query`:** fixed English **patterns** only, not open-ended LLM QA ([recorded decisions](../guides/GROUNDED_INSIGHTS_GUIDE.md#recorded-product-decisions-v1-issue-460)) | Optional NL → structured query for both layers ([#466](https://github.com/chipi/podcast_scraper/issues/466)) |

This note is **not** a commitment to build every “full depth” item; it maps **gap**
so roadmap and docs stay honest.

---

## Grounded Insights (GIL / `gi`)

### What shallow v1 covers today

- **Artifact:** `gi.json` generation in the pipeline when GIL is enabled (RFC-049).
- **CLI:** `gi validate`, `gi inspect`, `gi show-insight`, `gi explore`, `gi query`,
  `gi export` (NDJSON / merged bundle), aligned in spirit with `kg` where it makes sense.
- **`gi query`:** Deterministic pattern matching over English-ish phrases (topics,
  speakers, compound filters, topic leaderboard-style rollups). Responses are **structured
  JSON** built from graph-like walks over the artifact, not free-form answers.
- **`gi explore`:** Filtered views (e.g. by topic/speaker) for “Insight Explorer”-style
  browsing without a separate product UI.
- **Quality gates:** Schema validation scripts, optional **enforce** thresholds
  (e.g. PRD-017-oriented metrics) for CI / operators.

### What “full depth” would add (RFC-050 / PRD-017)

Items below are **spec or product** depth beyond the current deterministic CLI.

1. **Semantic / NL question answering (RFC-050 UC4-style)**  
   Arbitrary questions (“What are the main risks?”) answered with **evidence-backed**
   structured output, likely via an LLM or hybrid **constrained** to `insights[]` /
   `supporting_quotes[]`. Today: only **fixed patterns** match; everything else is out of band.

2. **Richer Insight Explorer UX**  
   Dedicated explorer experience (UI or API) with ranking, facets, and navigation —
   not only terminal JSON from `gi explore` / `gi query`.

3. **Synonyms, taxonomy, fuzzy topic alignment**  
   Substring / label matching and optional `ABOUT` edges are not a managed ontology;
   full depth implies curated or learned topic linking across phrasings.

4. **Cross-episode “intelligence” beyond file scan**  
   Global prioritization, deduplication of near-duplicate insights, cross-show identity
   for speakers — typically needs **corpus indices** or **PRD-018** tables.

5. **Optional join to KG**  
   Linking insights to KG node IDs for “topic/entity in graph ↔ grounded claim” is
   **explicitly deferred** in RFC-056 for v1; full depth could add stable join keys.

6. **Serving layer**  
   HTTP/API, batch jobs, or search index built on top of exports or RFC-051 — out of
   scope for the core CLI-only shallow v1.

---

## Knowledge Graph (KG / `kg`)

### What shallow v1 covers today

- **Artifact:** Per-episode `kg.json` when KG extraction is enabled (RFC-055), with
  schema validation (`kg validate`, published `docs/kg/kg.schema.json`).
- **CLI:** `kg validate`, `kg inspect`, `kg export` (NDJSON / merged), **`kg entities`**
  (cross-episode entity roll-up), **`kg topics`** (within-episode topic pair
  co-occurrence).
- **Quality gates:** KG quality metrics script with optional **enforce** thresholds
  (PRD-019-oriented) for CI / operators.

### What “full depth” would add (RFC-056 / PRD-019)

1. **Natural-language or ad-hoc graph queries**  
   RFC-056 lists **structured access first**; NL translation is post-v1. Full depth
   could add `kg query`-style commands with a defined **intermediate representation**
   (not arbitrary Cypher from users without a safe layer).

2. **Richer cross-episode analytics**  
   Beyond `entities` / `topics`: path queries, community detection, temporal trends,
   co-occurrence **across** episodes for arbitrary edge types — often easier with
   **graph DB** or **materialized tables** (RFC-051).

3. **Entity resolution**  
   PRD-019 non-goals admit imperfect resolution for v1. Full depth tightens **same
   entity across episodes** (canonical IDs, merging rules, human-in-the-loop or ML).

4. **Optional GIL ↔ KG linking**  
   Edges or attributes tying KG nodes to `insight_id` / quotes (RFC-056: optional,
   not required for v1).

5. **Adaptive extraction alignment**  
   RFC-053-style routing (`route_kg_extraction`, episode profiles) tuned for cost /
   quality — orthogonal to consumption but part of “mature” KG operationally.

6. **Downstream products**  
   Search, RAG, visualization, recommendations — consume exports or RFC-051; not
   delivered by the shallow CLI alone.

---

## Side-by-side summary

| Dimension | GI shallow v1 | GI full depth | KG shallow v1 | KG full depth |
| --------- | ------------- | ------------- | ------------- | ------------- |
| Per-episode artifact | Yes | Yes | Yes | Yes |
| Validate / export | Yes | Yes | Yes | Yes |
| Deterministic CLI queries | Pattern-only `gi query` | + NL / constrained QA | `entities`, `topics` rollups | + richer graph query / IR |
| DB-backed consumption | No | RFC-051 | No | RFC-051 |
| Cross-corpus identity | Weak | Stronger | Weak | Stronger resolution |
| Evidence / grounding | Core to GIL | Same contract, richer access | N/A (structure-first) | Optional link to GIL |

---

## Maintenance

When implementation catches up on a row in the tables above, update this note or fold
the paragraph into the user guides (`GROUNDED_INSIGHTS_GUIDE.md`,
`KNOWLEDGE_GRAPH_GUIDE.md`) and trim duplication here.

**Status:** WIP — analysis note, not a normative spec change.
