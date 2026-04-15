# RFC-075 open questions — follow-up notes (WIP)

**Status:** Draft — **Phase 1** (topic cluster line in graph **node rail**) and **Phase 2** (search
**API** query-time join **`metadata.topic_cluster`**; result card line; **Show on graph** camera may
include **`tc:`** compound) are implemented; open question **2** is closed as **canonical JSON only**
with server-side join at search time (no FAISS denormalization).  
**See:** [RFC-075: Corpus Topic Clustering](../rfc/RFC-075-corpus-topic-clustering.md) (§ Open questions).

This note captures discussion so we can decide later without losing context. When a direction is
chosen, promote the outcome into **UXS-004** / **UXS-005** and/or RFC-075, then trim or archive this
file.

### Recommended defaults (draft — for discussion)

These are **not** committed product decisions; they summarize a sensible default if we close the
questions without new constraints.

| Topic | Suggested default |
| ----- | ------------------- |
| **Open question 1** | Treat **B** (themed UX) as the **north star**, delivered **in phases**; keep **primary selection on the leaf** Topic (or insight/quote) until a deliberate **“select cluster”** flow exists. |
| **Open question 2** | Keep **`topic_clusters.json`** (and API passthrough) as the **only canonical** store for membership. **Do not** duplicate cluster ids into FAISS metadata **until** a concrete feature needs per-`kg_topic` cluster at **retrieval** time; if we add that later, **JSON stays canonical** and reindex rules are documented in one pipeline step. |

---

## Open question 1 — Search highlights and focus vs TopicCluster compound parents

### Problem

Semantic search resolves hits to **concrete** graph ids (`topic:…`, insights, etc.). **TopicCluster**
nodes (`tc:…` from **`graph_compound_parent_id`**) are **grouping** chrome: they are not FAISS
`source_id`s. So it is undefined whether **Show on graph**, selection, and highlighting should treat
only the **member Topic**, or also surface the **parent cluster** as part of the story.

### Rough options (from discussion)

| Option | Idea |
| ------ | ---- |
| **A — Minimal** | Highlight and focus **only** leaf nodes the index knows about; **TopicCluster** stays visual-only for layout. |
| **B — Themed UX (longer term)** | Build an explicit **hit → optional cluster context** in the client when `topic_clusters.json` is loaded: show cluster as part of the narrative (chip, compound emphasis, rail copy, optional camera behavior to the compound). |

### Current lean

**Lean toward B** as the north star: it matches user mental models for “same topic family, different
slug” and keeps **TopicCluster** from being decorative-only.

### Suggested guardrails if we pursue B (phased)

1. **Selection truth:** Keep **primary selection** on the **Topic** (or insight/quote) for detail
   rail and keyboard flows unless we add an explicit “select cluster” action. Avoid making
   **TopicCluster** the default focus target without a product decision.
2. **Phase 1 — Client-only:** Given loaded topic-clusters doc + highlighted/selected `topic:id`,
   resolve **`graph_compound_parent_id`** and add **lightweight** UI in the graph rail — done.
3. **Phase 2 — Search + graph:** Server attaches **`metadata.topic_cluster`** on **`kg_topic`** hits
   from **`topic_clusters.json`**; result cards show **Topic cluster:**; **Show on graph** keeps leaf
   selection and widens the camera to include the **`tc:`** parent when resolvable. Further
   grouping or “select cluster” flows remain out of scope until designed.
4. **Phase 3 — Optional:** Client-only joins only if we add features that cannot use the API field;
   not required for the current slice.
5. **Explicit non-goals for an early slice:** e.g. “result list does not group by cluster” until
   designed separately.

### Where to record a final decision

- **UXS-004** (graph) and **UXS-005** (search): short **Search ↔ graph handoff** subsection when
  ready.
- **RFC-075** § Open questions: update or close with a pointer to those UXSs.

---

## Open question 2 — Cluster membership in FAISS metadata vs `topic_clusters.json` only

### Problem

Membership today lives in **`search/topic_clusters.json`** (and optionally **HTTP**). FAISS rows
hold per-vector metadata for retrieval; they do not need cluster ids for search to function.

### Tradeoffs

- **JSON only:** Single place to update when re-clustering; no mandatory reindex. Risk: any feature
  that wants “cluster id per hit” must join client-side or load the JSON (already done for graph).
- **Denormalize into FAISS:** Convenient for query-time joins or future facets; implies **sync**
  discipline and usually **reindex** when clustering changes.

### Resolution (closed)

- **Canonical store:** **`topic_clusters.json`** only; no duplicate cluster ids in FAISS metadata.
- **Search:** **`run_corpus_search`** joins JSON at query time and sets **`metadata.topic_cluster`**
  for **`kg_topic`** rows when the topic is a cluster member — no reindex when clustering changes.
- Future **facets / ranking** that need cluster ids at retrieval time can still read the same JSON
  or this response field; re-evaluate FAISS denormalization only if profiling shows a bottleneck.

---

## Next steps (when revisiting)

1. Confirm or revise the **B** lean, phasing, and **recommended defaults** table above.
2. For **open question 1:** **Phase 1** is reflected in **UXS-004** / **UXS-005** and the node rail
   (`node-detail-topic-cluster-context`). Later phases: **Show on graph** / search emphasis — schedule
   when prioritized.
3. For **open question 2:** unless a feature forces otherwise, **affirm JSON-only** (now stated in
   RFC-075 § Open questions); if denormalization is needed later, add the **data** note + reindex
   contract in the same change that introduces the consumer.
