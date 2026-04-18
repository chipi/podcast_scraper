# RFC-075: Corpus Topic Clustering Layer

- **Status**: Draft
- **Authors**: Engineering (podcast_scraper)
- **Stakeholders**: Viewer / server maintainers, search and CIL owners
- **Related PRDs**:
  - `docs/prd/PRD-021-semantic-corpus-search.md` (embeddings source)
- **Related RFCs**:
  - `docs/rfc/RFC-061-semantic-corpus-search.md` (FAISS, `kg_topic` indexing)
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` (`topic_id_aliases`, CIL)
  - `docs/rfc/RFC-068-corpus-digest-api-viewer.md` (digest surfaces topics)
- **Related UX specs**:
  - `docs/uxs/UXS-004-graph-exploration.md` (graph rendering)
- **Related Documents**:
  - `docs/architecture/ARCHITECTURE.md`
  - `tests/fixtures/search/topic_clusters_validation.example.yaml` (minimal schema example for unit tests only; not a maintained dataset)

**GitHub tracking:**

- [#551](https://github.com/chipi/podcast_scraper/issues/551) — Phase 0: validation dataset
- [#552](https://github.com/chipi/podcast_scraper/issues/552) — Phase 1: clustering module + CLI + tests
- [#553](https://github.com/chipi/podcast_scraper/issues/553) — Phase 2: auto `topic_id_aliases`
- [#554](https://github.com/chipi/podcast_scraper/issues/554) — Phase 3: API + viewer compound nodes
- [#555](https://github.com/chipi/podcast_scraper/issues/555) — Phase 4 (optional): KG prompt nudges
- [#556](https://github.com/chipi/podcast_scraper/issues/556) — Post-implementation: 3-pass holistic review

## Abstract

This RFC defines a **Corpus Topic Clustering Layer**: an optional, well-bounded post-processing step that groups semantically similar KG topic nodes across episodes using embeddings already produced for semantic search. It writes a small corpus-level artifact (`topic_clusters.json`), can feed **auto-generated** `topic_id_aliases` for CIL, and lets the GI/KG viewer render **Cytoscape compound parent nodes** so users see topic families without losing per-episode labels.

**Architecture alignment:** The layer sits **beside** per-episode KG/GI extraction and **beside** CIL: it **reads** KG topic labels and search-time vectors, **does not** replace KG or FAISS, and **does not** require a database. Evolving clustering (threshold, algorithm, secondary signals) should be possible by changing this layer and its artifact only.

**Multiple id lenses:** The design is pragmatic, not a single unified topic ontology. You still have **embedding space** (FAISS), **`topic:…`** ids in KG and index metadata, **`tc:…`** compound parents for viewer layout (**`graph_compound_parent_id`**), and **CIL** merge targets (**`cil_alias_target_topic_id`**) plus hand **overrides** — they stay aligned by convention and tooling. A concise table lives in the [Semantic Search Guide — Id spaces (mental model)](../guides/SEMANTIC_SEARCH_GUIDE.md#id-spaces-mental-model).

## Problem Statement

KG topics use `topic:{slug}` derived from LLM-chosen labels. The same real-world subject often appears under different slugs and labels across episodes and feeds. Example from a 40-episode two-feed corpus: `topic:strait-of-hormuz-conflict`, `topic:strait-of-hormuz`, `topic:strait-of-hormuz-control`, and `topic:marine-traffic-disruption` all relate to Hormuz shipping and tension, but the merged graph shows them as unrelated nodes. CIL timelines and search lift that depend on **exact** `topic:` ids under-count cross-episode continuity.

**Use cases:**

1. **Viewer graph:** Operator loads a multi-episode corpus and sees topic variants grouped visually (compound nodes) with a shared canonical label.
2. **CIL topic timeline:** Queries keyed to a canonical topic id find more episodes once aliases map variants to that id.
3. **Offline operation:** Clustering runs at build or index time; no external API calls at query or view time.

## Goals

1. **Bounded module:** Implement clustering behind a clear interface (inputs: topic id, label, optional description, embedding vector, contributing episode ids; output: cluster assignments and `topic_clusters.json`).
2. **Filesystem artifact:** Emit `<corpus>/search/topic_clusters.json` alongside existing search index files; no new storage backend.
3. **Optional validation YAML:** Operators may pass **any** path to `--validate-config` with expected-same pairs and expected-distinct pairs while tuning clustering; there is **no** committed canonical validation file in `config/`. Unit tests use a **minimal fixture** under `tests/fixtures/search/` for YAML shape only.
4. **Integration hooks:** CLI command to (re)build clusters; optional HTTP route to serve the JSON; viewer consumes it as an overlay (compound parents), without mutating raw `*.kg.json`.
5. **CIL bridge:** Optional generation of `topic_id_aliases` entries from clusters; document merge rules with hand-authored overrides.

## Constraints and assumptions

**Constraints:**

- No query-time calls to external embedding APIs; reuse the same local model as indexing (e.g. `sentence-transformers/all-MiniLM-L6-v2` per RFC-061).
- Backward compatible when `topic_clusters.json` is absent (viewer and APIs degrade gracefully).
- Corpus layout remains filesystem-first; artifact size should stay small (on the order of kilobytes for hundreds of topics).

**Assumptions:**

- `kg_topic` (or equivalent) vectors exist in the FAISS store for topics to cluster, or labels can be re-embedded with the same model for a standalone clustering run.
- Operators accept that pure embedding similarity can confuse geography-qualified labels (e.g. Cuban vs Iranian economic crisis); an optional ad-hoc validation YAML can encode must-not-merge cases while tuning.

## Design and implementation

### 1. Layer placement and boundaries

The **Corpus Topic Clustering Layer** is not "semantic search" itself; it **consumes** embeddings produced for search (and KG topic text from artifacts). Position in the stack:

- **Below:** KG extraction (`*.kg.json`), FAISS indexing (`kg_topic` rows).
- **Peer:** CIL (`cil_lift_overrides.json`, `topic_id_aliases`) for cross-episode identity.
- **Above:** Server routes and viewer graph merge, which read `topic_clusters.json` as read-only overlay data.

**Isolation contract:** Core clustering logic should avoid importing FastAPI or Vue; adapters load FAISS / walk `*.kg.json` and call the pure function or class that returns clusters.

### 2. Algorithm and parameters

- **Similarity:** Cosine similarity on topic embedding vectors (same space as semantic search).
- **Clustering:** Greedy **average-linkage** merging with a configurable **cosine threshold** (starting point 0.75; tune against your corpus using optional `--validate-config` YAML). Repeatedly merge the two clusters whose **mean** pairwise similarity between members is highest, while that mean remains at or above the threshold. This avoids single-linkage chaining (weak A–C links via B).
- **Canonical label per cluster:** The member whose embedding has the **highest average cosine similarity** to all other members in the cluster (centroid-closest label), not merely the shortest string (short labels can be too narrow, e.g. "ICE Deployment").

**Validation YAML (operator-authored, any path):** Uses **`expected_merge_pairs`** (exactly two `topic:` ids that must co-cluster) for CLI checks, because full human “story” groups often fail pairwise embedding thresholds at 0.75 (e.g. Supreme Court vs tariff refunds in the same episode). **`expected_distinct`** lists pairs that must not co-cluster. Optional per-row **`episode_sources`** maps each `topic:` id in that row to one or more **`episode_id`** strings (same as metadata / FAISS `kg_topic` metadata) where that Topic appears in `*.kg.json`; clustering ignores `episode_sources` (human backtracking only).

**Known risk:** Geography-qualified topics ("Cuban Economic Crisis" vs "Economic Crisis in Iran") may score high on similarity. Mitigations: encode as expected-distinct in validation; if needed later, add a secondary signal (e.g. named-entity overlap) without changing the artifact schema version in a breaking way.

### 3. Artifact schema (`topic_clusters.json`)

Versioned JSON written under `<corpus>/search/topic_clusters.json`. Current writers emit **`schema_version`: `"2"`**. Readers accept **v1** payloads (legacy field names below) for older files on disk.

**Two different ids per cluster (do not conflate):**

| Field (v2) | Legacy (v1) | Prefix | Role |
| ---------- | ------------- | ------ | ---- |
| **`graph_compound_parent_id`** | `cluster_id` | `tc:` | **Viewer / Cytoscape only:** id of the compound **TopicCluster** parent node. Never used as a CIL merge target. |
| **`cil_alias_target_topic_id`** | `canonical_topic_id` | `topic:` | **CIL / `topic_id_aliases`:** the topic id non-centroid members should alias **to** when `--merge-cil-overrides` runs. Never used as the graph compound parent id. |

**Human-facing label (unchanged across versions):** **`canonical_label`** — display string for the cluster (centroid-closest member label), not a graph or CIL id.

Illustrative v2 shape:

```json
{
  "schema_version": "2",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "threshold": 0.75,
  "clusters": [
    {
      "graph_compound_parent_id": "tc:strait-of-hormuz",
      "cil_alias_target_topic_id": "topic:strait-of-hormuz",
      "canonical_label": "Strait of Hormuz",
      "member_count": 4,
      "members": [
        {
          "topic_id": "topic:strait-of-hormuz-conflict",
          "label": "Strait of Hormuz Conflict",
          "similarity_to_centroid": 0.95,
          "episode_ids": ["episode-id-or-stem"]
        }
      ]
    }
  ],
  "singletons": 155,
  "topic_count": 194,
  "cluster_count": 12
}
```

- **Members:** List actual `topic:…` ids present in KG graphs; viewer sets Cytoscape `parent` to **`graph_compound_parent_id`** when both cluster parent and member exist in the merged graph.

### 4. CIL: auto-generated `topic_id_aliases`

Optional post-step: map each non-centroid member `topic_id` to the cluster’s **`cil_alias_target_topic_id`** (slug from centroid-closest label). **`podcast_scraper.cli topic-clusters --merge-cil-overrides`** writes **`topic_clusters.json`**, derives aliases, then merges into **`cil_lift_overrides.json`**.

**Precedence:** Existing keys in the file’s **`topic_id_aliases`** win over freshly derived auto entries (hand edits and prior merged runs stay in control). New topic ids only in the auto map are appended. Other top-level keys (**`transcript_char_shift`**, **`entity_id_aliases`**) are preserved.

### 5. Viewer: Cytoscape compound nodes

- Load `topic_clusters.json` (via API or static path) when loading a corpus.
- After normal GI+KG merge, inject a **parent** node per cluster with `type: TopicCluster` (or agreed equivalent) and set `data.parent` on member topic nodes that appear in the graph.
- Style compound parents in `cyGraphStylesheet.ts` (e.g. rounded rectangle, low-opacity fill, dashed border, canonical label).
- **Do not** rewrite raw KG files; overlay only.

### 6. HTTP API

- **`GET /api/corpus/topic-clusters`**: returns `search/topic_clusters.json` for the resolved corpus root (`path` query or server default). **404** JSON body `{ "detail": "...", "available": false }` when the file is absent.

### 7. CLI

- **`topic-clusters`**: rebuild `topic_clusters.json` for a corpus root (`--threshold`, optional `--validate-config`, optional `--merge-cil-overrides`).

## Key decisions

1. **Compound nodes vs collapsing ids:** Use Cytoscape compound parents so per-episode labels remain visible; collapsing everything to one `topic:` id is optional via aliases for CIL/search but not required for the graph overlay.
2. **Centroid-closest canonical label:** Reduces risk of overly narrow canonical strings compared to "shortest label only."
3. **Draft RFC not in index:** Per project convention, Draft RFCs are not listed in `docs/rfc/index.md` until promoted.

## Alternatives considered

1. **Rewrite all topic ids in merge:** Simplifies CIL but loses distinct labels in the graph; rejected for viewer clarity.
2. **Only manual `topic_id_aliases`:** Does not scale with corpus growth; rejected as sole solution.
3. **New database for clusters:** Rejected; filesystem JSON is sufficient at current scale.

## Testing strategy

- **Unit tests:** Pure clustering on small synthetic embedding matrices; canonical label selection.
- **Integration / manual:** Run `topic-clusters --validate-config <your.yaml>` on a corpus whose index contains the listed `topic:` ids; there is no required repo-wide validation dataset.
- **Regression:** When `topic_clusters.json` is missing, server and viewer paths must not error.

## Rollout and monitoring

**Phases:**

1. Clustering module + CLI + tests (fixture YAML for shape only).
2. Auto-alias generation (`topic-clusters --merge-cil-overrides`).
3. API route (`GET /api/corpus/topic-clusters`) + viewer compound `TopicCluster` parents (Cytoscape `parent`).
4. KG extraction prompts nudge **short, stable topic headings** (about 2–8 words; detail in `description`) to improve cross-episode alignment (`kg/llm_extract.py` system prompts + shared Jinja user prompts).

**Success criteria:**

1. When using optional validation YAML locally, constraints pass under chosen threshold (or documented waiver for specific pairs).
2. `topic_clusters.json` size and generation time acceptable on a 40-episode reference corpus.
3. Viewer shows compound grouping for at least one multi-episode Hormuz-style cluster in manual QA.

## Post-implementation holistic review

Completed **four** passes (data flow, edge cases, documentation, second consistency audit) — see
[`docs/wip/rfc-075-holistic-review.md`](../wip/rfc-075-holistic-review.md).

## Autoresearch Findings (2026-04-18, PR feat/pipeline-validation-591)

### Threshold sweep on production corpus (1178 unique topics, 10 feeds)

| Threshold | Clusters | Singletons | Singleton% |
| :-------: | :------: | :--------: | :--------: |
| 0.60 | 208 | 504 | 43% |
| 0.65 | 177 | 655 | 56% |
| **0.70** | **129** | **809** | **69%** |
| 0.75 (old) | 88 | 929 | 79% |
| 0.80 | 56 | 1019 | 87% |

**Default lowered from 0.75 → 0.70**: +47% more clusters (88 → 129) with
manageable max cluster size (19). Manual inspection confirms all new clusters
are legitimate merges (quantum topics cluster, geopolitical topics cluster, etc.).

### KG label quality is the biggest lever for clustering

Issue #580 showed Gemini produces 156-char sentence labels → 90% singletons.
Root cause: sentence-shaped labels have low pairwise embedding similarity.
KG v2 prompt (#590) + label enforcement (#587, 50-char cap) address this.
Recommend re-sweep after deploying v2 prompt on production corpus — optimal
threshold may shift back to 0.75 with cleaner labels.

---

## Open questions

**Viewer / search (cluster parents vs hits):** Phase 1 shows **Topic cluster:** context (canonical
label) in the **graph node rail** when **`topic_clusters.json`** is loaded and the selected node is a
member **Topic**; **selection** stays on the Topic. Optional follow-ups (search-hit emphasis on
compounds, **Show on graph** camera to cluster) remain product decisions — see
[UXS-004](../uxs/UXS-004-graph-exploration.md), [UXS-005](../uxs/UXS-005-semantic-search.md), and
[`docs/wip/wip-rfc-075-open-questions-followup.md`](../wip/wip-rfc-075-open-questions-followup.md).

**Data (FAISS vs JSON):** **Canonical** cluster membership stays in **`topic_clusters.json`** (and
HTTP passthrough) unless a future feature requires denormalized cluster ids on **`kg_topic`** FAISS
rows; see the same WIP note for the recommended default.

**Working notes** (options, phasing, history):
[`docs/wip/wip-rfc-075-open-questions-followup.md`](../wip/wip-rfc-075-open-questions-followup.md).

## References

- `docs/rfc/RFC-061-semantic-corpus-search.md`
- `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
- `docs/architecture/ARCHITECTURE.md`
- `tests/fixtures/search/topic_clusters_validation.example.yaml` (test-only minimal example)
- `docs/wip/wip-topic-clusters-validation-reference.yaml` (optional WIP copy of a richer validation set for local `--validate-config`; not used by CI)
