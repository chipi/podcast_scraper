# Multi-feed corpus artifacts (GitHub #505 / #506)

JSON files written at the **corpus parent** — the directory passed as `output_dir` when running
**two or more feeds** in one batch (CLI repeated `--rss` / positional feeds, or config `feeds` /
`rss_urls` with ≥2 URLs). Single-feed runs do not require these files.

See also [RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md) §7 and
[SEMANTIC_SEARCH_GUIDE.md](../guides/SEMANTIC_SEARCH_GUIDE.md) (unified index at
`<corpus_parent>/search`).

## `corpus_manifest.json`

Rolling **per-feed operational** snapshot (updated each multi-feed finalize).

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `schema_version` | string | yes | Contract version; currently `1.0.0`. |
| `tool_version` | string | yes | `podcast_scraper` package version when written. |
| `corpus_parent` | string | yes | Absolute, normalized corpus parent path. |
| `updated_at` | string | yes | UTC ISO 8601 with `Z` when the manifest was written. |
| `feeds` | array | yes | One object per feed row in the batch (see below). |

**`feeds[]` object:**

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `feed_url` | string | yes | RSS URL for that row. |
| `stable_feed_dir` | string | yes | Directory name under `feeds/` (stable id from URL). |
| `last_run_finished_at` | string | yes | When that feed’s row finished (per-feed timestamp from the runner; ISO 8601 `Z`). |
| `ok` | boolean | yes | Whether the feed pipeline completed without exception. |
| `error` | string or null | yes | Redacted / short error string when `ok` is false; else null. |
| `episodes_processed` | integer | yes | Episodes processed for that feed in this batch. |

## `corpus_run_summary.json`

**Batch-oriented** summary for automation (same timing as manifest; complements structured log line
`corpus_multi_feed_summary`).

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `schema_version` | string | yes | Contract version; currently `1.1.0` (incident rollup on finalize). |
| `corpus_parent` | string | yes | Absolute, normalized corpus parent path. |
| `finished_at` | string | yes | UTC ISO 8601 with `Z` when the summary document was written (batch finalize). |
| `overall_ok` | boolean | yes | True only if every feed row has `ok: true`. |
| `feeds` | array | yes | Per-feed outcomes (see below). |
| `batch_incidents` | object | yes | Rollup of `corpus_incidents.jsonl` lines appended during this multi-feed batch only (byte window). |

**`feeds[]` object:**

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `feed_url` | string | yes | RSS URL. |
| `ok` | boolean | yes | Feed pipeline success. |
| `error` | string or null | yes | Error message when `ok` is false; else null. |
| `episodes_processed` | integer | yes | Count for that feed. |
| `finished_at` | string | optional | Present when the runner recorded a per-feed completion time (ISO 8601 `Z`). |
| `failure_kind` | string | optional | When `ok` is false: `soft` or `hard` (GitHub #559 classification). |
| `episode_incidents_unique` | object | yes | Distinct episode-scoped incident rows in the batch window for this feed URL: `policy`, `soft`, `hard` (see `batch_incidents`). |

**`batch_incidents` object:**

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `log_path` | string | yes | Path to `corpus_incidents.jsonl` used for the rollup. |
| `window_start_offset_bytes` | integer | yes | File size at batch start; lines appended after this offset belong to this batch. |
| `window_end_offset_bytes` | integer | yes | File size when finalize ran. |
| `lines_in_window` | integer | yes | Number of JSON lines parsed in the window. |
| `episode_incidents_unique` | object | yes | Distinct episode keys per category: `policy`, `soft`, `hard`. |
| `feed_incidents_unique` | object | yes | Distinct feed URLs with feed-scoped rows per category: `policy`, `soft`, `hard`. |
| `episodes_documented_skips_unique` | integer | yes | Operator shortcut: same value as `episode_incidents_unique.policy`. |
| `episodes_other_incidents_unique` | integer | yes | Sum of `episode_incidents_unique.soft` and `episode_incidents_unique.hard`. |
| `semantics_note` | string | yes | Short prose on how to read `episodes_processed` vs incidents. |

### Reading `ok`, `episodes_processed`, and incidents

A feed can show **`ok: true`** and **`episodes_processed: 0`** when the pipeline exited without a
feed-level exception but no episode completed the full success path (for example every episode hit a
documented policy skip). **`batch_incidents`** and per-feed **`episode_incidents_unique`** summarize
`corpus_incidents.jsonl` for this batch only so “all feeds ok” is not confused with “every episode
produced a transcript.”

## Service API mirror

When using `service.run` / `run_from_config_file` with multi-feed config, the same payload as
`corpus_run_summary.json` is also returned on **`ServiceResult.multi_feed_summary`** (see
[SERVICE.md](SERVICE.md)). Single-feed runs leave this field `null`.

## Partial batches and unified index

If one feed fails, the batch still **writes** manifest and summary, sets `overall_ok` to false, and
(if `vector_search` is true and `vector_backend` is `faiss`) still runs **`index_corpus` on the
corpus parent** so completed feeds contribute to `<corpus_parent>/search`. Failed feeds contribute
no new metadata until a later successful run.

## Hybrid `metadata/` layout (indexing)

If both **`feeds/`** and a top-level **`metadata/`** directory exist under the corpus parent,
episode metadata discovery for the unified index includes **both** trees (legacy or auxiliary
files at the parent plus per-feed metadata under `feeds/<id>/…/metadata/`).
