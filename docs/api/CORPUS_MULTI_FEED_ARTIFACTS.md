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
| `schema_version` | string | yes | Contract version; currently `1.0.0`. |
| `corpus_parent` | string | yes | Absolute, normalized corpus parent path. |
| `finished_at` | string | yes | UTC ISO 8601 with `Z` when the summary document was written (batch finalize). |
| `overall_ok` | boolean | yes | True only if every feed row has `ok: true`. |
| `feeds` | array | yes | Per-feed outcomes (see below). |

**`feeds[]` object:**

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `feed_url` | string | yes | RSS URL. |
| `ok` | boolean | yes | Feed pipeline success. |
| `error` | string or null | yes | Error message when `ok` is false; else null. |
| `episodes_processed` | integer | yes | Count for that feed. |
| `finished_at` | string | optional | Present when the runner recorded a per-feed completion time (ISO 8601 `Z`). |

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
