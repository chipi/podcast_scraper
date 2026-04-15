# RFC-075 holistic review (post-implementation)

Structured pass over **data flow**, **degradation**, and **documentation** after the Corpus
Topic Clustering Layer implementation. See [RFC-075](../rfc/RFC-075-corpus-topic-clustering.md).

## Pass 1: End-to-end data flow

| Stage | Inputs | Outputs / consumers |
| ----- | ------ | --------------------- |
| **Indexing** | `*.kg.json` topics, transcript chunks | FAISS under `<corpus>/search/` with `doc_type: kg_topic` rows (`source_id` = `topic:…`, vectors per chunk/row). |
| **Clustering CLI** | FAISS store + KG labels from `*.kg.json` | `search/topic_clusters.json` (`build_topic_clusters_for_corpus`, greedy average-linkage). Optional `--validate-config`, `--merge-cil-overrides`. |
| **CIL / lift** | `cil_lift_overrides.json` | `topic_id_aliases` merged (file keys win over auto). `load_cil_lift_overrides` → `resolve_id_alias` in transcript lift and search paths. |
| **HTTP** | Corpus root | `GET /api/corpus/topic-clusters` returns JSON or **404** `{ available: false }`. |
| **Viewer** | Merged GI+KG graph + optional cluster JSON | `fetchTopicClustersDocument` after API artifact load; `applyTopicClustersOverlay` sets Cytoscape `parent` on matching **Topic** nodes; **TopicCluster** parent nodes. |

**Consistency:** Clustering does not rewrite `*.kg.json`. Aliases affect lift/search identity resolution; graph overlay is visual-only unless operators also merge aliases.

## Pass 2: Edge cases and failure modes

| Scenario | Behavior |
| -------- | -------- |
| No `topic_clusters.json` | API **404**; viewer skips overlay (`null` doc). |
| No `kg_topic` rows / missing index | `topic-clusters` exits non-zero (`FileNotFoundError` / no rows for validate). |
| Singleton topics (one embedding row) | Omitted from `clusters` in JSON; no auto-alias from clustering for that id. |
| `cil_lift_overrides.json` corrupt on merge | Merge path logs warning and treats as empty top-level object before merge. |
| `topic_clusters.json` invalid JSON on API | **500** with generic detail. |
| Layer filter hides all KG topics | `pruneOrphanTopicClusterParents` removes **TopicCluster** nodes with no visible children; strips stale `parent` on topics. |
| Same topic matched by two clusters | Last cluster in payload wins for `parent` in overlay (data should not duplicate). |

**Open (RFC):** Search-hit highlighting and focus navigation vs compound **TopicCluster** parents — not specified; treat as future UX work.

## Pass 3: Documentation cross-check

| Doc | Status |
| --- | ------ |
| [RFC-075](../rfc/RFC-075-corpus-topic-clustering.md) | Phases 0–4 and HTTP/CLI described; open questions list remains. |
| [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) | Corpus Topic Clustering section + mermaid `CorpusTopicClusters` route. |
| [SERVER_GUIDE.md](../guides/SERVER_GUIDE.md) | `GET /api/corpus/topic-clusters` row in route table. |
| [CLI.md](../api/CLI.md) | `topic-clusters` including `--merge-cil-overrides`. |
| [SEMANTIC_SEARCH_GUIDE.md](../guides/SEMANTIC_SEARCH_GUIDE.md) | CIL overrides + RFC-075 merge note; short **Topic clustering** subsection added for discoverability. |
| [UXS-004](../uxs/UXS-004-graph-exploration.md) | Compound **TopicCluster** contract. |
| `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` (repo root; not under `docs/`) | Graph shell + topic-clusters fetch. |

**Closed follow-ups (this pass):**

- **Acceptance:** `config/acceptance/README.md` (repository root) documents the optional post-run **`topic-clusters`** command (not wired into the acceptance runner by design).
- **Playwright:** `e2e/search-to-graph-mocks.spec.ts` mocks **`GET /api/corpus/topic-clusters`** with schema v2 JSON and asserts **Topic cluster** appears in the **Types** row.
- **Observability:** CLI / module logs include **`schema_version`** on successful writes; HTTP handler emits a **debug** line with **`schema_version`** and cluster count when serving the file.
- **Full-path docs:** [SEMANTIC_SEARCH_GUIDE.md](../guides/SEMANTIC_SEARCH_GUIDE.md), [CLI.md](../api/CLI.md), [GIL_KG_CIL_CROSS_LAYER.md](../guides/GIL_KG_CIL_CROSS_LAYER.md), [UXS-004](../uxs/UXS-004-graph-exploration.md) reference **v2** field names where relevant.

## Pass 4: Second review (consistency audit)

**When:** Follow-up pass after schema v2 field naming, expanded tests, and cross-guide updates.

| Check | Result |
| ----- | ------ |
| **Python readers** | `_cil_alias_target_topic_id` / `_graph_compound_parent_id` prefer v2 keys; v1 legacy keys still read. |
| **CLI `--merge-cil-overrides`** | Derives aliases from the same **`payload`** written to disk (`topic_id_aliases_from_clusters_payload`). |
| **HTTP `GET /api/corpus/topic-clusters`** | Passthrough JSON; **404** / **500** contracts unchanged; **debug** log includes **`schema_version`** when enabled. |
| **Viewer overlay** | `graph_compound_parent_id` ?? `cluster_id`; member **`topic_id`** matched after stripping layer prefixes (`stripLayerPrefixesForCil`). |
| **Doc set** | [SERVER_GUIDE.md](../guides/SERVER_GUIDE.md) route row matches behavior; [GIL_KG_CIL_CROSS_LAYER.md](../guides/GIL_KG_CIL_CROSS_LAYER.md) includes clustering row + test pointers. |
| **Tests** | v2 assertions in server unit/integration; Vitest **`corpusTopicClustersApi`**; Playwright mock + **Types** row; v1 alias test retained in **`test_topic_clusters.py`**. |

**Residual (product / future, not implementation gaps):**

- RFC-075 **Open questions** (search-hit highlighting vs **TopicCluster** parents; optional FAISS metadata vs JSON-only membership) — unchanged.
- **v1-only files on disk:** Supported indefinitely via reader fallbacks; no forced migration.

**Outcome:** No new blocking inconsistencies found in this pass.

## Review outcome

No blocking inconsistencies found between implemented behavior and the above docs for RFC-075.
Remaining product questions live under **Open questions** in RFC-075.
