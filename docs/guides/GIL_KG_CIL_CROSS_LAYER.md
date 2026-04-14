# GIL, KG, and cross-layer (CIL)

This guide ties together the **Grounded Insight Layer (GIL)**, the **Knowledge Graph (KG)**,
and the **Canonical Identity Layer (CIL)**: shared `person:` / `topic:` / `org:` identities,
per-episode **`bridge.json`**, cross-episode **HTTP queries**, and **semantic search lift**
from transcript chunks to structured insights. Use it as a **map**; layer-specific behaviour
stays in the linked guides and RFCs.

---

## How the pieces fit

| Piece | Role | Primary doc |
| ----- | ---- | ----------- |
| **GIL** (`gi.json`) | Evidence-backed insights and verbatim **Quote** nodes (`char_start` / `char_end`, timestamps, optional speaker). | [Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md), [GIL ontology](../architecture/gi/ontology.md) |
| **KG** (`*.kg.json`) | Entities, topics, and relationships for navigation and linking. | [Knowledge Graph Guide](KNOWLEDGE_GRAPH_GUIDE.md), [KG ontology](../architecture/kg/ontology.md) |
| **`bridge.json`** (per episode) | Joins GI and KG surfaces under **one** canonical id per real-world person/topic/org; **`display_name`** for UI. Emitted next to `gi.json` / `kg.json` stems. | [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) |
| **CIL HTTP API** | Read-only queries over on-disk **bridge + gi + kg** (position arc, person profile, topic timeline, id lists). | [Server Guide](SERVER_GUIDE.md) (`/api/persons/*`, `/api/topics/*`) |
| **Semantic search lift** | For **transcript** FAISS hits, optional **`lifted`** block (insight, speaker, topic, quote times) when chunk spans overlap a **Quote** and `bridge.json` resolves names. | [Semantic Search Guide](SEMANTIC_SEARCH_GUIDE.md) |
| **Offset verification** | Confirms **Quote** char ranges overlap **transcript chunk** metadata in the index (same coordinate space). | [Semantic Search Guide — lift and verification](SEMANTIC_SEARCH_GUIDE.md#chunk-to-insight-lift-and-offset-verification-rfc-072--528) |

**Viewer:** The GI/KG SPA loads artifacts via **`GET /api/artifacts`** and merges GI+KG (and bridge-backed dedupe where implemented). See [Development Guide — GI / KG browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype), [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md), [UXS-001](../uxs/UXS-001-gi-kg-viewer.md).

---

## Artifacts on disk

Typical episode workspace (paths vary for multi-feed; see [CORPUS_MULTI_FEED_ARTIFACTS](../api/CORPUS_MULTI_FEED_ARTIFACTS.md)):

- `*.metadata.json` — episode row, provenance paths to `gi.json` / `kg.json`.
- `*.gi.json` — GIL graph.
- `*.kg.json` — KG graph (when `generate_kg` ran).
- `*.bridge.json` — CIL identity map for that episode (when the pipeline emits it).

**Path rules in code:** `src/podcast_scraper/builders/rfc072_artifact_paths.py` (bridge next to metadata; GI/KG siblings of bridge; bridge next to `gi.json`). **GIL edge `type` comparisons:** `src/podcast_scraper/gi/edge_normalization.py`.

The vector index lives at **`<corpus_root>/search/`** (`vectors.faiss`, `metadata.json`, …) when `vector_search` / `index` has run at the **corpus parent** for multi-feed trees.

---

## CLI and Make targets

| Command | Purpose |
| ------- | ------- |
| `python -m podcast_scraper.cli verify-gil-chunk-offsets --output-dir <corpus>` | JSON report: Quote vs transcript chunk overlap (RFC-072 Phase 5 gate). Supports **feed-nested** metadata (`feeds/.../metadata/`) via discovered metadata files. |
| `make verify-gil-offsets-strict` | Same verifier with **`--strict`** and **`--min-overlap-rate`** (default **0.95**). Override corpus: `GIL_OFFSET_VERIFY_DIR=/path/to/run`. |
| `python -m podcast_scraper.cli search …` / `index …` | Semantic search and FAISS maintenance ([Semantic Search Guide](SEMANTIC_SEARCH_GUIDE.md)). |
| `python -m podcast_scraper.cli gi …` / `kg …` | GIL and KG CLIs ([Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md), [Knowledge Graph Guide](KNOWLEDGE_GRAPH_GUIDE.md)). |

---

## HTTP API (summary)

- **`GET /api/health`** — includes **`cil_queries_api`** when CIL routes are mounted.
- **`GET /api/search`** — corpus search; **transcript** hits may include **`lifted`** (dict) per hit when alignment and graph edges allow.
- **`GET /api/persons/{id}/positions|brief|topics`**, **`GET /api/topics/{id}/timeline|persons`** — CIL cross-layer queries ([Server Guide](SERVER_GUIDE.md)).

OpenAPI: **`/docs`** when the server is running.

---

## Testing (where coverage lives)

| Area | Layer | Location / command |
| ---- | ----- | ------------------- |
| GIL pipeline, schema, CLI | Unit + integration + E2E | `tests/unit/gi/`, `tests/integration/`, `tests/e2e/test_gi_cli_e2e.py`; see [Testing Strategy — GIL Testing](../architecture/TESTING_STRATEGY.md#gil-testing-implemented--prd-017-rfc-049050) |
| KG | Unit + E2E | `tests/unit/kg/`, `tests/e2e/test_kg_cli_e2e.py` |
| Bridge builder | Unit | `tests/unit/builders/test_bridge_builder.py` |
| CIL query logic | Unit | `tests/unit/podcast_scraper/server/test_cil_queries.py` |
| CIL HTTP | Integration | `tests/integration/server/test_cil_api.py` |
| Search lift + offset verify | Unit | `tests/unit/podcast_scraper/search/test_transcript_chunk_lift.py`, `test_gil_chunk_offset_verify.py` |
| Bridge integration (pipeline-shaped) | Integration | `tests/integration/test_bridge_integration.py` |
| FastAPI viewer (search, health, library, …) | Integration | `tests/integration/server/test_server_api.py`, `test_viewer_search.py`, … |
| Viewer TS merge / bridge types | Vitest | `make test-ui` |
| Viewer UX | Playwright | `make test-ui-e2e` |

**Quality gates:** `make quality-metrics-ci` (fixtures), optional `make gil-quality-metrics` / `make kg-quality-metrics` on a real run. **Offset strict gate:** `make verify-gil-offsets-strict` when you have an indexed corpus path.

---

## CI and acceptance workflows

- **Pytest** jobs already cover unit/integration/E2E for GIL, KG, server, and search helpers.
- **Vitest / Playwright** cover the Vue viewer (`make test-ui`, `make test-ui-e2e`).
- **Main / release CI** (`.github/workflows/python-app.yml`, job `test-acceptance-fixtures`) runs `make test-acceptance-fixtures-fast`, then **`make verify-gil-offsets-after-acceptance`** on every acceptance `run_*` that has **`search/metadata.json`**. That is **not** the same as scheduled **nightly** (`nightly.yml` runs `make test-nightly`, not the acceptance matrix).
- **Default `make ci-fast`** still skips full acceptance and offset gates; local checks use **`make verify-gil-offsets-strict`** with **`GIL_OFFSET_VERIFY_DIR`** when you have an indexed corpus.

See [Testing Guide — GIL, KG, CIL, and semantic search](TESTING_GUIDE.md#gil-kg-cil-and-semantic-search-validation) and `scripts/acceptance/README.md` (canonical acceptance + CI wording).

---

## Specifications and ADRs

- **RFC-072** — CIL, `bridge.json`, cross-layer query patterns, semantic search lift path.  
  [RFC-072-canonical-identity-layer-cross-layer-bridge.md](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md)
- **ADR-052** — Separate GIL and KG artifacts.  
- **ADR-053** — Grounding contract for evidence-backed insights.  
- **RFC-061 / PRD-021** — Semantic corpus search (FAISS).  
- **RFC-062** — Viewer v2 (FastAPI + Vue).  
- **PRD-017 / PRD-019** — GIL and KG product requirements.

---

## Related guides (read next)

- [Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md)  
- [Knowledge Graph Guide](KNOWLEDGE_GRAPH_GUIDE.md)  
- [Semantic Search Guide](SEMANTIC_SEARCH_GUIDE.md)  
- [Server Guide](SERVER_GUIDE.md)  
- [Testing Guide](TESTING_GUIDE.md)  
- [Testing Strategy](../architecture/TESTING_STRATEGY.md)  
- [Architecture — GIL, KG, and CIL](../architecture/ARCHITECTURE.md#gil-kg-and-canonical-cross-layer-cil)
