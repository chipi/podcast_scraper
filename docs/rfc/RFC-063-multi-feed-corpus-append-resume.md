# RFC-063: Multi-Feed Corpus, Append/Resume, and Unified Discovery

- **Status**: Draft
- **Authors**: Marko Dragoljevic
- **Stakeholders**: Maintainer
- **Related PRDs**:
  - None — GitHub [#440](https://github.com/chipi/podcast_scraper/issues/440) and [#444](https://github.com/chipi/podcast_scraper/issues/444) track scope.
- **Related ADRs**:
  - [ADR-051](../adr/ADR-051-per-episode-json-artifacts-with-logical-union.md) — per-episode GI/KG artifacts; union at query time
  - [ADR-060](../adr/ADR-060-vectorstore-protocol-with-backend-abstraction.md) — vector index abstraction (RFC-061)
  - [ADR-061](../adr/ADR-061-faiss-phase-1-with-post-filter-metadata.md) — FAISS Phase 1 indexing
- **Related RFCs**:
  - [RFC-001](RFC-001-workflow-orchestration.md) — pipeline orchestration
  - [RFC-004](RFC-004-filesystem-layout.md) — output directories and run scoping (**extended** by this RFC)
  - [RFC-007](RFC-007-cli-interface.md) — CLI and validation
  - [RFC-008](RFC-008-config-model.md) — configuration model
  - [RFC-049](RFC-049-grounded-insight-layer-core.md) — GIL artifacts
  - [RFC-055](RFC-055-knowledge-graph-layer-core.md) — KG artifacts
  - [RFC-061](RFC-061-semantic-corpus-search.md) — semantic index; **recursive corpus discovery** and composite keys align here
  - [RFC-062](RFC-062-gi-kg-viewer-v2.md) — viewer and `serve`; corpus root path semantics
- **Related UX specs**:
  - [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) — GI/KG viewer (corpus root folder)
- **Related Documents**:
  - [GitHub #440](https://github.com/chipi/podcast_scraper/issues/440) — multiple RSS feeds
  - [GitHub #444](https://github.com/chipi/podcast_scraper/issues/444) — append / incremental resume
  - [GitHub #505](https://github.com/chipi/podcast_scraper/issues/505) — unified corpus parent indexing (recursive metadata, composite keys, search/explore)
  - [GitHub #506](https://github.com/chipi/podcast_scraper/issues/506) — corpus manifest, status CLI, machine-readable multi-feed summary
  - [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md) — normative `corpus_manifest.json` / `corpus_run_summary.json` (#506)
  - `docs/architecture/ARCHITECTURE.md` — multi-feed outer loop documented (GitHub #440)
  - `docs/architecture/TESTING_STRATEGY.md` — new test tiers as needed
  - `config/examples/acceptance_multi_feed_planet_money_journal_openai.yaml` — tracked full-pipeline acceptance preset (Planet Money + The Journal, OpenAI); `USE_FIXTURES=1` rewrites `feeds:` to local E2E fixture URLs (may be copied to `config/acceptance/full/` locally)
  - `config/manual/manual_multi_feed_planet_money_journal_openai.yaml` — manual validation preset (same feeds, `max_episodes: 1`)

## Abstract

This RFC specifies **multi-feed ingestion** (N RSS feeds in one config / CLI run), **opt-in append/resume**
(filesystem + `index.json`, no SQLite in this phase), a **unified corpus layout** (parent directory with
`feeds/<stable_feed_id>/` subtrees), and **corpus-wide discovery** for semantic index, `search`, `gi explore`,
and `serve` so KG, GI, and search work across **all** feeds without duplicating artifacts.

**Architecture Alignment:** Additive to the modular pipeline: outer orchestration over feeds, inner
`run_pipeline` behavior and episode parallelism unchanged unless explicitly gated (e.g. vector auto-index).
Extends RFC-004 run/output semantics where append requires **stable** per-feed workspaces.

## Problem Statement

Today the main CLI is built around **one** optional RSS URL; `Config` exposes a single `rss_url`. Large
overnight jobs need **many feeds**, **isolated failures** (one feed or episode must not block others), and
**resumability** after crashes or code fixes. The current `run_*` timestamped directories and the fact that
`_finalize_pipeline` (including `index.json` and `maybe_index_corpus`) is skipped when processing **raises**
make naive “rerun” and index-only resume fragile.

Downstream, **RFC-061** indexing and **RFC-062** viewer assume a corpus root with `metadata/` at one level;
nested `feeds/<id>/…/metadata/` requires **recursive metadata discovery** and path resolution relative to
each episode’s feed root (`metadata_path.parent.parent`).

**Use Cases:**

1. **Overnight multi-feed run**: Ten feeds, one config; failures logged per feed; next morning rerun with
   `--append` completes only missing or failed episodes.
2. **Unified search**: One FAISS index under `<corpus>/search`; filter by `feed_id` from existing metadata.
3. **Viewer**: Same “Corpus root folder” as `serve --output-dir` pointing at parent; list/load GI/KG across
   feeds (artifacts API already `rglob`; search tab needs parent index).
4. **Fix and resume**: After a bugfix, append run skips completed work keyed by `episode_id` + artifact
   validation, not only path globs.

## Goals

1. **Multi-feed CLI and config**: Backward-compatible single `rss`; add repeatable `--rss` and/or `@file`;
   config allows a list of feeds without duplicating the full config object.
2. **Layout A**: Corpus parent with `feeds/<stable_feed_id>/`; each subtree keeps `transcripts/`, `metadata/`,
   manifests, and per-run artifacts consistent with existing pipeline outputs.
3. **Opt-in append**: Stable `effective_output_dir` per feed under append (no unbounded new `run_<timestamp>`
   churn); skip/retry driven by **artifact validation + `episode_id`**, with `index.json` as accelerator.
4. **Failure isolation**: One failing feed does not abort others; episode failures respect `fail_fast` /
   `max_failures` within a feed.
5. **Unified discovery**: `index_corpus`, `search` CLI helpers, `gi explore` metadata maps, and documented
   `serve` behavior work with **corpus parent** + recursive metadata + composite `(feed_id, episode_id)` keys
   in the vector index fingerprint map.
6. **Exit semantics**: Non-zero if any feed failed; structured per-feed summary in logs (and optional
   machine-readable summary — see §7).

## Constraints & Assumptions

**Constraints:**

- **Backward compatibility**: Single feed, no new flags → behavior matches current releases (including
  `derive_output_dir(rss)` when `output_dir` omitted).
- **Explicit parent for N > 1**: When two or more feeds are configured, **require** explicit corpus parent
  `output_dir` (do not infer only from “first RSS”).
- Must not break existing acceptance configs; add new configs for multi-feed / append.

**Assumptions:**

- Maintainer documents **single-writer** per corpus parent for v1 (optional lockfile later).
- Rate limits and API cost scale with N feeds; callers may throttle externally.

## Non-Goals (this phase)

- SQLite or other DB as primary resume store (may follow later; [RFC-051](RFC-051-database-projection-gil-kg.md)
  remains complementary).
- Cross-feed parallelism beyond “sequential feeds, parallel episodes per feed” unless trivial.
- Stage-level checkpoints (transcribe vs metadata) — episode-level v1 only.

## Design & Implementation

### 1. Corpus layout (Layout A)

```text
<corpus_parent>/
  feeds/
    <stable_feed_id_a>/
      transcripts/
      metadata/
      index.json          # per-feed run index (optional location TBD in implementation)
      run_manifest.json   # when written
    <stable_feed_id_b>/...
  search/                 # unified vector index (RFC-061), when built at parent
  corpus_manifest.json    # optional — see §7
```

**Stable feed id:** Derived deterministically from feed URL (hash + short slug); exact algorithm left as
implementation detail with collision tests.

**Single-feed compatibility:** Either keep today’s default `output/rss_<host>_<hash>/` without a `feeds/`
segment when only one feed and no multi-feed mode, **or** always use `feeds/<id>/` under an explicit parent
— **open choice** (see Open Questions).

### 2. Multi-feed orchestration

- **One `Config` template** (or clone per feed) so `workers`, providers, and feature flags match across feeds.
- **Outer loop**: for each feed, set `rss_url`, set per-feed `output_dir` to
  `join(corpus_parent, "feeds", stable_feed_id)`, wrap in **try/except** so one feed’s exception does not skip
  remaining feeds.
- **Inner**: existing `run_pipeline(cfg)` (or extracted core) unchanged except append/stable-dir and optional
  `skip_auto_vector_index` (§5).
- **Programmatic API**: Prefer a dedicated multi-feed entry point returning **per-feed** results; keep
  `service.run` single-feed until extended (avoid blocking on `ServiceResult` redesign).

### 3. Append / resume

- **Flag**: e.g. `--append` (opt-in). Without it, behavior matches current defaults (including interaction
  with `--skip-existing`).
- **Stable directory**: Under append, reuse the same per-feed workspace; **do not** create a new timestamped
  `run_*` subtree each invocation — requires coordinated changes with
  `podcast_scraper.utils.filesystem.setup_output_directory` and callers.
- **Truth source**: **Filesystem + validation first** (`episode_id`, parseable metadata/transcript where
  configured); **`index.json` second** for speed and last-error strings. Reconcile after crashes because
  finalize may not run on exception.
- **Schema**: Extend or version `index.json` as needed; document migration for old files.

### 4. Failure isolation

| Scope    | Behavior |
| -------- | -------- |
| Feed     | Log failure; continue other feeds; record status for append. |
| Episode  | Existing `fail_fast` / `max_failures` per feed. |
| Viewer   | Incomplete subtree must not break listing other feeds’ artifacts. |

### 5. Unified semantic index (RFC-061 integration)

- **Discovery**: Under corpus parent, glob `**/metadata/*.metadata.json` (and yaml variants), respecting
  existing `run_*` nesting if still present. If **`feeds/`** exists **and** the parent has a top-level
  **`metadata/`** directory, **both** are included (hybrid layout; GitHub #505 follow-up).
- **Path resolution**: For each metadata file, `episode_root = metadata_path.parent.parent`; join
  `grounded_insights.artifact_path`, `knowledge_graph.artifact_path`, and transcript rel paths against
  `episode_root`, not the corpus parent alone.
- **Keys**: Fingerprint / vector row identity uses **composite** `(feed_id, episode_id)` (or equivalent
  string) to avoid GUID collisions across feeds.
- **`maybe_index_corpus`**: For multi-feed batch toward one parent index, **disable** per-feed auto-index
  during inner runs and invoke **`index_corpus(corpus_parent)` once** after the batch (or document
  incremental per-feed alternative and cost).

**KG CLI**: Existing `rglob` for `.kg.json` already suits parent corpus; verify any flat-metadata-only paths.

### 6. CLI and config surface

- Positional `rss` preserved; add `--rss URL` (repeatable) and/or file list.
- Config file: `rss` as string **or** list, or dedicated `feeds:` key — merge rules documented.
- Aggregated exit code: non-zero if any feed failed.

### 7. Supplementary artifacts (holistic operations)

These are **small, optional** additions that pair well with multi-feed corpora. **Normative field
tables** and operational notes live in
[`docs/api/CORPUS_MULTI_FEED_ARTIFACTS.md`](../api/CORPUS_MULTI_FEED_ARTIFACTS.md) (published under
**API → Multi-feed corpus artifacts**).

1. **`corpus_manifest.json`** (at `<corpus_parent>/`): `schema_version`, `tool_version`,
   `corpus_parent`, `updated_at`, and `feeds[]` with `feed_url`, `stable_feed_dir`,
   `last_run_finished_at` (per-feed completion time from the runner), `ok`, `error`,
   `episodes_processed`.
2. **Corpus status command** (`corpus-status` / `podcast corpus-status`): print per-feed metadata counts,
   sample `index.json` errors, and whether `<corpus_parent>/search` exists — GitHub #506.
3. **`corpus_run_summary.json`**: batch summary with `finished_at`, `overall_ok`, and `feeds[]`
   (`feed_url`, `ok`, `error`, `episodes_processed`, optional `finished_at` per row). A structured log
   line `corpus_multi_feed_summary` echoes the same payload. **`service.run`** returns this document on
   **`ServiceResult.multi_feed_summary`** for multi-feed runs.

**Partial batches:** Manifest and summary are written even when some feeds fail (`overall_ok: false`).
With `vector_search` + FAISS, **parent `index_corpus` still runs** after finalize so successful feeds are
searchable; failed feeds add no metadata until a later successful run.

## Key Decisions

1. **Persistence**: Filesystem + `index.json` only for this RFC; DB later if needed.
2. **Append default**: Opt-in `--append`; non-append preserves legacy semantics for single-feed workflows.
3. **Resume identity**: Prefer `episode_id` from RSS over filename-only heuristics; document interaction with
   `skip_existing`.
4. **Unified index location**: Default `<corpus_parent>/search` when indexing at parent.
5. **`run_id`**: Audit and manifest, not a forced new output tree on every append invocation.

## Risks and Code-Informed Gaps

1. **Run directory churn**: Timestamped `run_*` under `setup_output_directory` fragments output unless append
   explicitly stabilizes per-feed roots.
2. **Finalize skipped on exception**: If `_process_episodes_with_threading` raises, `_finalize_pipeline` does
   not run — `index.json` may lag; reconcile from disk.
3. **`episode.idx` in filenames vs GUID `episode_id`**: Feed reordering changes paths; resume must not rely on
   paths alone.
4. **Double process / no lock**: Advisory exclusive lock **`.podcast_scraper.lock`** at the corpus
   parent during multi-feed **CLI** and **service** batches (`filelock`). Override with
   **`PODCAST_SCRAPER_CORPUS_LOCK=0`** when a second process must read-only the tree or for tests.
5. **Resource use**: N sequential `run_pipeline` calls may reload ML models N times; optional shared provider
   session later.
6. **Docs**: Update README, viewer README, semantic search guide, and `ARCHITECTURE.md` when behavior ships.

## Alternatives Considered

1. **Per-feed only (no parent index)**  
   **Pros**: No indexer changes. **Cons**: Conflicts with unified search/viewer goal; N manual index passes.

2. **Flat global `metadata/` (no `feeds/`)**  
   **Pros**: Simple glob. **Cons**: Collides with stable feed scoping, migration pain, harder per-feed deletes.

3. **SQLite-first resume**  
   **Pros**: Strong consistency. **Cons**: Out of scope for v1; user chose filesystem + `index.json` first.

## Testing Strategy

- **Unit**: `stable_feed_id`, path resolution (`episode_root`), composite index key formatting.
- **Integration**: Two feeds, one failing RSS URL; assert other completes; append second run skips completed
   episodes.
- **Regression**: Single-feed acceptance configs unchanged.
- **Optional slow**: Acceptance config for multi-feed overnight profile (tiered per `TESTING_STRATEGY.md`).

## Rollout & Phasing

1. **Phase 1**: Multi-feed orchestration + layout A + explicit parent `output_dir` + per-feed isolation.
2. **Phase 2**: Append/stable dir + `index.json` extensions + filesystem reconciliation.
3. **Phase 3**: Unified `index_corpus` / `search` / `gi explore` / docs + composite keys + batch vector index
   policy.
4. **Phase 4**: `corpus_manifest.json`, **`corpus-status`** CLI, machine-readable batch summary
   (GitHub #506 — manifest + `corpus_run_summary.json` + structured log line; **`ServiceResult.multi_feed_summary`**
   mirrors the summary JSON; per-feed `finished_at` / `last_run_finished_at`; normative contract
   **`docs/api/CORPUS_MULTI_FEED_ARTIFACTS.md`**; hybrid parent `metadata/` + `feeds/` discovery in #505).

## Relationship to Other RFCs

| RFC | Relationship |
| --- | ------------ |
| RFC-004 | This RFC **extends** filesystem/run layout rules for multi-feed and append-stable behavior. |
| RFC-061 | This RFC **requires** corpus-parent indexing and recursive metadata; composite vector keys. |
| RFC-062 | Viewer corpus root = parent; artifacts API already recursive; search needs parent index. |
| RFC-001 | Orchestration gains an outer multi-feed loop; inner pipeline unchanged. |

## Migration Path

1. Ship multi-feed behind explicit parent `output_dir` and new flags — no change for existing single-feed
   invocations.
2. Document “point `serve` / `search index` at corpus parent” once Phase 3 lands.
3. Optional ADRs may spin out: stable append layout, composite index keys, resume truth ordering.

## Open Questions

1. Exact **`stable_feed_id`** algorithm (length, collision handling).
2. **Non-append multi-feed**: same flat `feeds/<id>/` tree as append, or retain nested `run_*` for parity with
   historical single-feed runs.
3. **Incremental** `index_corpus(parent)` after each feed vs **batch-only** after all feeds (latency vs cost).

## References

- `src/podcast_scraper/workflow/orchestration.py` — `run_pipeline`, finalize path
- `src/podcast_scraper/utils/filesystem.py` — `setup_output_directory`, `derive_output_dir`
- `src/podcast_scraper/cli.py` — `_build_config`
- `src/podcast_scraper/search/indexer.py` — `index_corpus`, `maybe_index_corpus`
- `src/podcast_scraper/server/routes/artifacts.py` — recursive GI/KG listing
- `web/gi-kg-viewer/README.md` — corpus root instructions

## Lineage

Informal WIP notes for #440 / #444 were retired (2026-04) in favor of this RFC and
[CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md) for operational JSON contracts.
