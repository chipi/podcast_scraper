# Corpus artifacts and viewer surfaces

Authoritative map of **what the pipeline writes**, **what the API reads**, and **which viewer tabs depend on which routes**. Used by:

- [PROD_RUNBOOK — Code/content compatibility](../guides/PROD_RUNBOOK.md#codecontent-compatibility) (operator decision tree)
- [GitHub #796](https://github.com/chipi/podcast_scraper/issues/796) (`produced_by`, `/api/health` preflight)
- [GitHub #797](https://github.com/chipi/podcast_scraper/issues/797) (`scripts/ops/post_deploy_smoke.sh`)
- [Code/content compatibility matrix](../COMPATIBILITY.md)

Survey baseline: `src/podcast_scraper/server/routes/` and pipeline writers under `src/podcast_scraper/workflow/`, `search/`, `gi/`, `kg/`, `builders/`.

---

## Section 1 — Artifact inventory

| Artifact | Path (relative to corpus root) | Writer | Reader(s) | `schema_version`? | Min code version |
| --- | --- | --- | --- | --- | --- |
| `corpus_manifest.json` | top-level | multi-feed finalize (`write_corpus_manifest`) | metrics routes, `/api/health` preflight (#796) | yes (`1.1.0`) | 2.5+ |
| `corpus_run_summary.json` | top-level | multi-feed finalize | `GET /api/corpus/runs/summary`, metrics | yes (`1.1.0`) | 2.5+ |
| `corpus_incidents.jsonl` | top-level | pipeline incidents | run summary rollup | n/a (JSONL) | 2.5+ |
| `feeds.spec.yaml` | top-level | operator / seed | `GET/PUT /api/feeds`, pipeline | n/a | n/a |
| `viewer_operator.yaml` | top-level | operator | `GET/PUT /api/operator-config`, jobs | n/a | n/a |
| Episode metadata | `feeds/<stable>/run_*/metadata/*.metadata.json` | pipeline | library, digest, detail, graph | partial (episode fields) | 2.0+ |
| Transcripts | `feeds/<stable>/run_*/metadata/*.txt` (and segments JSON when enabled) | pipeline / transcription | episode detail | n/a (text) | 2.0+ |
| GIL artifacts | `feeds/<stable>/run_*/metadata/*.gi.json` | GI pipeline | graph, library, detail, artifacts API | yes (`2.0`) | 2.4+ |
| KG artifacts | `feeds/<stable>/run_*/metadata/*.kg.json` | KG pipeline | graph, library, detail, artifacts API | yes (`1.2`) | 2.4+ |
| Bridge artifacts | `feeds/<stable>/run_*/metadata/*.bridge.json` | CIL bridge step | library filters, node-episodes, graph | implicit (bridge builder) | 2.5+ |
| LanceDB two-tier index | `search/lance_index/` (segment + insight + aux tables) | two-tier index build / migration | `GET /api/search` (hybrid, default) | n/a (embedded DB) | 2.7+ |
| LanceDB index | `search/lance_index/` | embed / index build | `GET /api/search`, explore | n/a (embedded DB) | 2.7+ |
| Index metadata | `search/metadata.json` | `build_two_tier_index` (written next to the LanceDB index) | GIL chunk-offset verifier (char offsets), index stats | n/a | 2.5+ |
| Topic clusters | `search/topic_clusters.json` | post-index cluster build | `GET /api/corpus/topic-clusters`, digest topics, library filter | yes (`2`) | 2.5.5+ |
| Catalog JSON (optional) | `corpus/*.json` (e.g. precomputed catalog exports) | tooling / fixtures | some dashboard paths when present | varies | n/a |

---

## Section 2 — API route → artifact dependency map

| Route | Reads (primary) | Returns empty if missing? | Fails hard if missing? |
| --- | --- | --- | --- |
| `GET /api/health` | optional default corpus `corpus_manifest.json` (#796) | n/a | no |
| `GET /api/artifacts` | `**/*.gi.json`, `**/*.kg.json`, `**/*.bridge.json` | empty `artifacts` list | no |
| `GET /api/artifacts/{path}` | single artifact file | 404 | yes (per file) |
| `GET /api/corpus/feeds` | walks `*.metadata.json` | empty `feeds` | no |
| `GET /api/corpus/episodes` | catalog from metadata (+ optional bridge/clusters) | empty `items` | no |
| `GET /api/corpus/episodes/detail` | metadata, transcript, `.gi.json`, `.kg.json`, bridge | partial fields null | no (per field) |
| `GET /api/corpus/episodes/similar` | LanceDB index + catalog | empty / error in body | soft (search error string) |
| `GET /api/corpus/digest` | catalog rows by publish window | empty `rows` | no |
| `GET /api/corpus/topic-clusters` | `search/topic_clusters.json` | 404 JSON `{available: false}` | yes (404, not 5xx) |
| `GET /api/corpus/persons/top` | GI/KG-derived person index | empty list | no |
| `GET /api/corpus/stats` | catalog | zeroed histograms | no |
| `GET /api/corpus/coverage` | GI/KG presence walk | empty coverage | no |
| `GET /api/corpus/documents/manifest` | `corpus_manifest.json` | 404 | yes |
| `GET /api/corpus/documents/run-summary` | `corpus_run_summary.json` | 404 | yes |
| `GET /api/corpus/runs/summary` | run summary + incidents | partial empty | no |
| `GET /api/search` | `search/lance_index/` (two-tier hybrid) | 200 with `error` field (`no_index` when absent) | no (structured error) |
| `GET /api/explore` | LanceDB index + artifacts | empty graph payload | soft |
| `GET /api/index/stats` | `search/` index files | `available: false` | no |
| `POST /api/index/rebuild` | corpus artifacts | async job errors | yes (job failure) |
| `GET/PUT /api/feeds` | `feeds.spec.yaml` | empty feeds list | no |
| `GET/PUT /api/operator-config` | `viewer_operator.yaml` | defaults | no |
| `POST/GET /api/jobs` | operator YAML + docker | job errors | yes (spawn failure) |
| CIL routes (`/api/persons/*`, `/api/topics/*`) | GI/KG + bridge | empty / 404 per id | per route |

---

## Section 3 — Viewer tab → API route map

| Tab | Primary routes | Secondary routes |
| --- | --- | --- |
| **Library** | `/api/corpus/feeds`, `/api/corpus/episodes` | `/api/corpus/episodes/detail` (rail), `/api/corpus/episodes/similar` |
| **Digest** | `/api/corpus/digest` | `/api/corpus/persons/top`, `/api/search` (topic bands) |
| **Graph** | `/api/artifacts`, `/api/artifacts/{path}` | `/api/corpus/topic-clusters`, `/api/corpus/episodes/detail`, `/api/explore` |
| **Search** | `/api/search` | `/api/index/stats` (index status chip) |
| **Dashboard** | `/api/corpus/stats`, `/api/corpus/runs/summary`, `/api/index/stats` | `/api/corpus/coverage`, `/api/corpus/topic-clusters`, `/api/jobs` (Pipeline sub-tab) |
| **Configuration** (status bar) | `/api/feeds`, `/api/operator-config`, `/api/health` | `/api/jobs`, `/api/scheduled-jobs` |

Post-deploy smoke (`scripts/ops/post_deploy_smoke.sh`) hits the **primary** route for Library, Digest, Graph-relevant corpus reads, and Search, plus `/api/health`.

---

## Section 4 — Migration history (append-only)

| Release | Artifact change | Notes |
| --- | --- | --- |
| 2.6.0 | No required schema bumps | Optional fields only; `produced_by` added to `corpus_manifest.json` (#796). |
| 2.5.2 | `corpus_manifest` 1.0 → 1.1 | Added `cost_rollup` — PR #650. |
| 2.5.0 | Multi-feed corpus artifacts | `corpus_manifest.json`, unified `search/` index (#505 / #506). |
| 2.5.x | `topic_clusters.json` | schema `"2"`; built after LanceDB index exists. |
| 2.4.0 | GIL 2.0, KG 1.2 | RFC-072 canonical identity; read-time migrations in `gil_kg_identity_migrations.py`. |
| 2.4.0 | Bridge artifacts | `*.bridge.json` siblings next to metadata/GI/KG. |

When bumping any `schema_version`, update this table **and** [COMPATIBILITY.md](../COMPATIBILITY.md).
