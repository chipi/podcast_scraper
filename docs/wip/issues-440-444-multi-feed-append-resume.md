# WIP: Multi-feed CLI (#440) and append / resume (#444)

**Status:** Draft — scratch / iteration; **canonical design:** [RFC-063: Multi-Feed Corpus, Append/Resume, and Unified Discovery](../rfc/RFC-063-multi-feed-corpus-append-resume.md).  
**GitHub:** [#440 Support multiple rss feeds](https://github.com/chipi/podcast_scraper/issues/440), [#444 redo append mode / incremental](https://github.com/chipi/podcast_scraper/issues/444).  
**Last updated:** 2026-04-06

## Unified corpus goal (KG / GI / semantic search)

**Intent:** One **logical corpus** — **unified KG exploration, GI inspection, and semantic search across all feeds and all episodes**, not ten isolated per-feed indexes.

Implications:

- Users point **`--output-dir` (and `serve` default)** at the **corpus parent** (e.g. `<corpus>/` that contains `feeds/<feed_id>/…`), not at each feed subtree.
- **One** vector index (under `<corpus>/search` or `vector_index_path` relative to that parent) covers every episode; filters (e.g. by `feed_id`) use metadata already emitted in documents (`feed.feed_id` exists today via `generate_feed_id`).
- **`kg` corpus commands** already **`rglob` `.kg.json`**; entity rollup / export over the parent dir align naturally with unified KG once paths are consistent.

This does **not** require moving GI/KG files to a flat global folder: **on-disk layout stays layout A** (per-feed trees). Only **discovery and path resolution** in index/search/explore/serve must become **corpus-aware**.

### GI/KG viewer (v2): which folder to use?

The viewer stays **the same UX**: **Corpus root folder** in the UI must match **`serve --output-dir`** (or `SERVE_OUTPUT_DIR` / `PODCAST_SERVE_OUTPUT_DIR` default), exactly as `web/gi-kg-viewer/README.md` describes today.

| Layout | Path to enter (corpus root) |
| ------ | --------------------------- |
| **Single-feed (today)** | The pipeline output root for that run (e.g. `output/rss_<host>_<hash>/` or `…/run_<suffix>/` — wherever `metadata/` and `transcripts/` live for that feed). |
| **Multi-feed (layout A)** | The **parent** directory that contains **`feeds/`** — e.g. `/path/to/mycorpus` if artifacts live under `mycorpus/feeds/<feed_id>/…`. Do **not** point at `mycorpus/feeds/one_podcast` unless you intentionally want only that feed. |

**What already works against the parent:** `GET /api/artifacts` lists `**/*.gi.json` and `**/*.kg.json` **recursively** under that root (`src/podcast_scraper/server/routes/artifacts.py`), so **List files** and **Load selected into graph** can see every feed under `feeds/` **without** changing the viewer.

**What needs the unified indexer first:** **Semantic search** (and index stats) expect a FAISS store under `<corpus>/search` built with **`search index`** using the **same** corpus root. Until `index_corpus` discovers nested metadata and resolves paths per feed root, build the index from the parent path may be incomplete — same gap as CLI `search index`.

## Downstream impact: GI, KG, semantic index, `serve`

### What stays the same (pipeline / on-disk)

- **Pipeline** still writes each feed under its own **`effective_output_dir`** (`feeds/<id>/…`). GI/KG `artifact_path` fields stay **relative to that feed root** (`os.path.relpath(..., output_dir)` in metadata generation) — **no format change required** for multi-feed.
- **GI transcript layout** (`metadata/` → sibling `transcripts/`) remains valid **per feed root**.

### What must evolve for unified corpus (search / explore / serve)

Today these code paths assume **one** `output_dir` and **`output_dir/metadata/*.metadata.json`** (non-recursive):

- `index_corpus` / **`search index`** (`search/indexer.py`)
- **`search` CLI** (`_episode_to_gi_path` and friends)
- **`gi explore`** (`_episode_id_to_gi_path_from_metadata`)
- **`serve`** (default corpus dir)

For a parent layout:

```text
<corpus>/
  feeds/
    feed_a/…/metadata/*.metadata.json
    feed_b/…/metadata/*.metadata.json
```

they need:

1. **Recursive metadata discovery** (e.g. `**/metadata/*.metadata.json` under `<corpus>`, respecting existing `run_*` nesting if present).
2. **Per-episode root for path joins:** `grounded_insights.artifact_path` / `knowledge_graph.artifact_path` / `content.transcript_file_path` are relative to the **feed’s** output directory, not `<corpus>`. Resolution should use **`episode_root = metadata_path.parent.parent`** (the directory that contains `metadata/` and `transcripts/`) when joining relative paths — same value as `out` for today’s flat single-feed tree, but critical when `out` is the parent corpus.
3. **Unified index location:** default `<corpus>/search` (or configured `vector_index_path`) so **one** FAISS store spans all feeds.

### Episode identity in a merged index

- **Fingerprint / vector upsert keys** today use **`episode_id` alone**. Metadata already includes **`feed_id`** (`generate_feed_id`); for a unified index, treat **vector / fingerprint keys as `(feed_id, episode_id)`** (or a single composite string) so rare **GUID collisions across unrelated feeds** cannot clobber rows.
- **Backward compatibility:** single-feed corpora keep `feed_id` in metadata; composite keys remain stable when only one feed exists.

### KG CLI

- **Scan / export** already recurse with `rglob`; **entity rollup** over `<corpus>` should see all `.kg.json` files once they live under feed subtrees. Verify any code path that assumes a flat `metadata/` only (if present) and align with recursive discovery where needed.

### Append / resume and unified index

- After append completes new or repaired episodes, **`search index`** (incremental) runs against **`<corpus>`**; per-episode fingerprints use composite keys so only changed episodes re-embed.

### Summary

| Layer | Change for unified corpus |
| ----- | --------------------------- |
| Pipeline output layout | None (still per-feed under `feeds/<id>/`) |
| Metadata JSON shape | None required (`feed_id` + relative artifact paths already there) |
| `kg` rglob-based tools | Minimal / verification only |
| `index_corpus`, `search`, `gi explore`, `serve` | **Yes** — recursive metadata + `episode_root` path resolution + composite index keys |

**Phasing (suggested):** (1) Multi-feed pipeline + append (#440 / #444) with stable per-feed dirs. (2) **Unified discovery + indexing** pass so parent `--output-dir` satisfies that goal without duplicating artifacts.

## Decisions (locked for this iteration)

| Topic | Choice |
| ----- | ------ |
| Output layout | **A** — One corpus root; under it, **subdirectories per feed** (same internal layout as today: `transcripts/`, `metadata/`, manifests, index). |
| Append / resume default | **Opt-in** via explicit flag (e.g. `--append`); non-append behavior stays aligned with **current** defaults. |
| Persistence | **Filesystem + `index.json` (run index)** only; **no SQLite** in this phase; DB can follow later. |

## Pipeline goal

Run the CLI with **many feeds** (e.g. 10) in **one config**, reuse **existing intra-pipeline parallelism** (`workers`, etc.) unchanged. Let a long run finish overnight; if something fails, fix code or environment, **rerun with append**, and **only process what was not already completed** (per feed, per episode).

**Downstream:** Operate **one unified corpus** for **KG, GI, and semantic search** across every feed and episode (see **Unified corpus goal** above and [RFC-063 § Design](../rfc/RFC-063-multi-feed-corpus-append-resume.md#5-unified-semantic-index-rfc-061-integration)).

## Failure isolation (default behavior)

**Requirement:** Failures are **scoped**; one broken unit must not prevent progress on the rest.

| Scope | Expectation |
| ----- | ----------- |
| **Feed** | If feed *A* fails (RSS unreachable, parse error, auth, etc.), feeds *B…N* still run to completion for that invocation. Record feed *A* as failed in logs + summary; do not abort the whole multi-feed run by default. |
| **Episode** | If one episode fails (download, transcribe, metadata, GI, etc.), other episodes in **that** feed continue unless the user explicitly enables **stop-early** behavior (today: `fail_fast`, `max_failures` on `Config`). |

**Multi-feed orchestration** should wrap each feed in **isolated** error handling (catch, log, persist failure state for append, continue). Optional **opt-in** flags may later add “abort all feeds on first feed failure” for debugging; that must **not** be the default.

**Unified viewer / search:** One feed’s bad or missing artifacts must not break listing or loading others; APIs already scope by path — avoid whole-corpus failures when one subtree is incomplete.

## Backward compatibility

**Is “old configs and acceptance tests behave like the old version” reasonable?**

**Yes**, if we scope it precisely:

- **Single feed, no new flags:** Same effective behavior as today — same validation, same default output derivation relative to that one feed (see layout note below), same non-append semantics.
- **New capabilities are additive:** Multiple feeds require **new** CLI surface (e.g. repeatable `--rss`) and/or **new** config shape (`rss` as list or `feeds:`). Existing YAML/JSON with a **single string `rss`** keeps working without edits.
- **Tests:** Existing acceptance configs should remain valid; add new cases for multi-feed + append rather than rewriting old paths unless we intentionally deprecate something.

**Layout note for single-feed + layout A:** Today, default output is often `output/rss_<host>_<hash>/`. With layout A for **multi-feed**, the natural shape is `output_dir/feeds/<feed_slug_or_hash>/…`. For **strict** “one feed, zero new flags, identical paths,” we can define: **one feed** continues to use the **current** default root (no extra `feeds/` segment); **two or more feeds** use `feeds/<id>/` under a user-provided or derived parent. Alternatively, always use `feeds/` with a single child for one feed — that is a **small path change** for single-feed defaults and must be called out if we choose it. **Open:** pick one rule and document it in the implementation section when coding.

## Non-goals (this phase)

- Replacing `index.json` with a database.
- New cross-feed parallelism beyond what the pipeline already does for episodes (unless trivially “sequential feeds, parallel episodes per feed”).

## Requirements

### 1. One `Config`, reuse internal parallelism

- Prefer **one** `Config` instance (or clone-per-feed with identical knobs) so **workers**, transcription concurrency, and other settings apply **per feed** the same way they do today inside `run_pipeline`.
- Orchestration: **outer loop over feeds** (sequential feeds is fine v1); **inner** behavior unchanged.

### 2. Default without append / resume

- **Without** `--append` (or equivalent), behavior matches **today**: no obligation to reuse prior run state beyond what current flags already do (`--skip-existing`, etc.).
- **Retiring old options:** Acceptable **if** redundant after append + index-backed resume (e.g. overlap between `--skip-existing` and append semantics). Any removal needs a **changelog** and a **migration line** in this doc (“use X instead of Y”).

### 3. Append / resume (filesystem + index.json)

- **Opt-in** append mode:
  - **Stable** effective output root for that corpus (no accidental new `run_*` directory each restart — address timestamped `run_suffix` behavior documented in codebase).
  - Persist / read **per-episode** state in **`index.json`** (extend schema if needed): at least success vs failed vs incomplete, last stage, error summary, stable **episode id** (GUID-based where possible; align with `get_episode_id_from_episode`).
- On rerun with append:
  - **Skip** episodes marked **successfully complete** for the full pipeline (or for stages the user configured — define minimal v1: “transcript + metadata consistent with current flags”).
  - **Retry** failed or incomplete episodes.
- **Per-feed isolation:** Same as [Failure isolation](#failure-isolation-default-behavior): feed *i* failing does not prevent feed *j*; index / run summary records per-feed status for append retries.

### 4. Multi-feed (#440)

- CLI: backward-compatible single positional `rss`; add **repeatable** `--rss URL` and/or `@file` list.
- Config: allow multiple feeds in one file without duplicating the whole config.
- Exit code: define aggregation (e.g. non-zero if **any** feed had failures); log a **summary table** per feed. **Success** for a feed does not depend on other feeds having succeeded.

## Implementation checklist (for later PRs)

- [ ] Document single-feed default path rule vs `feeds/<id>/` (compatibility).
- [ ] Stable run directory in append mode (config + code).
- [ ] Extend or companion file for run index schema (version bump, migration for old `index.json`).
- [ ] Orchestrator: `for feed in feeds: run_pipeline(clone_cfg)` (or equivalent) with **per-feed try/except** so one failing feed does not skip remaining feeds; preserve episode-level `fail_fast` / `max_failures` semantics inside each feed.
- [ ] Tests: single-feed regression, multi-feed smoke, append retry after simulated failure.
- [ ] Deprecation list if `--skip-existing` overlaps append (optional consolidation).
- [ ] **Unified corpus:** recursive metadata discovery from corpus parent; resolve GI/KG/transcript paths using **`metadata_path.parent.parent`**; composite `(feed_id, episode_id)` keys for fingerprints / vector rows; align `serve` / `gi explore` / `search` CLI with same rules.
- [ ] **Hardening:** reconcile index + filesystem for append; decide finalize-on-crash / checkpointing; corpus lockfile or documented single-writer; multi-feed `ServiceResult` or contract; `maybe_index_corpus` strategy for parent dir.

## Critical gaps and risks (code-informed)

This section is intentionally skeptical: where a “simple” multi-feed loop or `--append` flag can fail after days of work if we do not nail semantics up front.

### 1. Output layout today is not “flat `output_dir/metadata`”

`setup_output_directory` always builds a **nested** `run_<suffix>` under `cfg.output_dir`, with `run_suffix` including a **new timestamp every run** (and often a config hash when ML features are on). If the target directory already exists, a **numeric suffix** (`_1`, `_2`, …) picks another new tree.

**Risk:** Append/resume that assumes “same folder as last time” without **explicit rules** will keep creating **new** `run_*` trees, fragmenting transcripts/metadata, breaking user mental models and unified indexing. **Mitigation:** Define append semantics as “reuse **stable** `effective_output_dir` per feed” (or stable symlink), and add tests that two invocations write to the **same** leaf `metadata/` when append is on.

### 2. Crash and exception paths skip `index.json` / metrics finalization

`run_pipeline` only reaches `_finalize_pipeline` (which writes `index.json`, `metrics.json`, `maybe_index_corpus`, run summary) if control flows past episode processing. If `_process_episodes_with_threading` **raises**, the `finally` block cleans up providers, then the **exception propagates** — **`_finalize_pipeline` is not called**. Failures earlier (e.g. RSS fetch in `_fetch_and_prepare_episodes`, provider setup) also skip finalization.

**Risk:** Treating `index.json` as the **sole** source of truth for resume overstates its reliability; after OOM, kill -9, or an unhandled exception, on-disk state and index can **diverge**. **Mitigation:** Append logic must **reconcile** index + filesystem (and define “complete episode” from artifacts). Optionally **flush** incremental index / checkpoint per episode or per stage; consider `finally` that always writes a **partial** run index (spec complexity vs benefit).

### 3. `episode_id` vs filenames use different stability rules

Stable identity for metrics uses `generate_episode_id` (GUID or hash including feed URL). **Filenames** still use **`episode.idx`** (RSS order) and `title_safe` plus `run_suffix`.

**Risk:** Feed reordering or title changes can produce **duplicate logical episodes** (same GUID, new filename) or confusing skips if resume keys only on paths. **Mitigation:** Resume/skip decisions should prefer **episode_id** from RSS; document interaction with `skip_existing` (path-based).

### 4. `service.run` and programmatic API

`service.run` wraps `run_pipeline` in a broad `try/except` and returns `episodes_processed=0` on failure. Multi-feed orchestration must define **per-feed** results (partial success) — **not** only a single `ServiceResult`.

**Risk:** Daemons and integrations assume one boolean success. **Mitigation:** New API shape (e.g. `MultiRunResult`) or documented “call `run_pipeline` per feed” pattern.

### 5. CLI config coupling to a single RSS URL

`_build_config` sets `output_dir` via `filesystem.derive_output_dir(args.rss, …)` from **one** feed.

**Risk:** Multi-feed needs an explicit **parent** output root and per-feed children; backward compatibility must be spelled out (see layout note elsewhere in this doc).

### 6. Unified semantic index vs `maybe_index_corpus`

Pipeline finalization calls `maybe_index_corpus(effective_output_dir, cfg)` for **that run’s** directory only. With many feeds and a **parent** corpus index, either **disable** per-run auto-index and run **one** `index_corpus(parent)` after all feeds, or **merge** incrementally with correct path resolution — otherwise embeddings and fingerprints drift or duplicate work.

**Risk:** N feeds × full index passes = cost; wrong `output_dir` = empty or partial index. **Mitigation:** Config flag or orchestrator step for “index once at corpus root after multi-feed batch”.

### 7. Cost, rate limits, and quotas

N feeds × many episodes × summarization / GI / KG / embeddings multiplies **API cost** and **429** risk. Isolation says one feed’s failure should not stop others, but **global** rate limits can still stall or fail unrelated episodes unless backoff is per-key and well tested.

### 8. Resource reuse vs isolation

Each `run_pipeline` call preloads ML models and builds providers. Sequential feeds may **reload** large models N times (latency, memory spikes), or you share providers across feeds (coupling, thread-safety, harder failure boundaries).

**Risk:** “Simple loop” is slow or OOM on GPU. **Mitigation:** Optional **shared provider session** for a multi-feed run (design doc + tests).

### 9. Concurrency: two processes, one corpus

No file locking on corpus dirs. Two overlapping runs (or cron overlap) can corrupt partial writes or race on `index.json` / FAISS files.

**Mitigation:** Document single-writer expectation; optional lockfile under corpus root.

### 10. Partial and corrupt artifacts

Crash mid-write can leave truncated transcript/metadata. Path-existence checks can mis-classify **bad** files as “done.”

**Mitigation:** Define completeness (schema validation, min size, or stage flags in index); append retries **validation failures**.

### 11. `generate_gi` / `generate_kg` vs run suffix

`_build_provider_model_suffix` does **not** include GI/KG toggles. Timestamp still changes every run, so run dir churn is dominated by time — but **config hash** stability can mask some changes. Any assumption “same hash = same run folder” is fragile.

### 12. Tests and CI surface area

Multi-feed × append × unified index × crash recovery explodes combinations. **Risk:** Under-tested paths regress. **Mitigation:** Tiered tests — unit for path resolution and id keys; integration with 2 feeds + kill mid-run; acceptance config for overnight-style run (may be slow; mark tier).

### 13. Docs drift

README, `DEVELOPMENT_GUIDE`, RFC-061/062, viewer README, and `ARCHITECTURE.md` describe **single** `rss` and flat corpus assumptions. **Risk:** Users follow old instructions and point the viewer or `search index` at the wrong directory. **Mitigation:** Single “corpus layout” page linked from README when behavior ships.

## Strong recommendations (decide before coding)

Opinionated defaults so implementation does not thrash. Change them whenever you want.

### 1. Parent `output_dir` when N > 1

**Recommend:** **Require** an explicit corpus parent (`--output-dir` or config `output_dir`) when **two or more feeds** are configured. **Do not** infer the parent from “first RSS URL only” — that hides mistakes and fights `derive_output_dir`.

**Keep:** Single feed with no `output_dir` → continue **today’s** `derive_output_dir(rss)` behavior.

### 2. Canonical tree per feed + append vs `run_<timestamp>`

**Recommend:** Treat **multi-feed + append** as the driver for **stability**:

- Per-feed workspace: `<parent>/feeds/<stable_feed_id>/` (stable id = hash/slug of feed URL; document algorithm).
- **With `--append`:** reuse that directory across invocations and **do not** create a **new** `run_<timestamp>…` subtree every time (today’s logic would fragment output and break resume). Concretely: append mode should resolve to **one durable `effective_output_dir` per feed** (either flat `transcripts/` / `metadata/` under `feeds/<id>/`, or **one** explicitly stable `run_<fixed-id>` — pick one and test).
- **Without append** on multi-feed: first run may create the same tree; optional separate policy for “experimental” runs is **out of scope** until v1 ships.

**Single-feed non-append:** preserve current behavior (including existing `run_*` semantics) for acceptance / backward compatibility.

### 3. Truth source for resume

**Recommend:** **Artifact + validation first**, **index second**.

- **Skip** an episode only if configured stages are **satisfied** (e.g. transcript + metadata present and parseable / min sanity checks), keyed by **`episode_id`**, not only by filename glob.
- **`index.json`:** update on successful finalization; use to **accelerate** scans and store last error. After crash, **reconcile** by walking metadata/transcripts — do not assume index is complete.

### 4. Vector index (`maybe_index_corpus` / unified search)

**Recommend:** For **multi-feed batch** runs with a **parent** corpus and unified search:

- **Turn off** per-feed automatic indexing during each `run_pipeline` (e.g. orchestrator passes `vector_search=False` for inner runs, or a dedicated `skip_auto_vector_index` flag), then run **`index_corpus(parent)` once** after all feeds finish (or after each feed if you need incremental UI — then document the cost).

**Single-feed:** keep current `maybe_index_corpus(effective_output_dir)` behavior.

### 5. Service / programmatic API

**Recommend:** **Do not** block multi-feed on redesigning `service.run` initially.

- Add a **clear** multi-feed entry point (e.g. `run_multi_feed(...)` returning **per-feed** results: ok/fail, counts, paths) used by CLI.
- **`service.run`:** remains single-feed until you explicitly extend it with a structured multi-feed result type.

### 6. Checkpoint granularity

**Recommend:** **Episode-level only for v1.** Stage-level checkpoints (transcribe done, metadata not) are powerful but multiply edge cases; defer unless you hit a concrete pain point.

### 7. Concurrency / locking

**Recommend:** **v1 — document “one writer per corpus parent.”** Optional lockfile under `<parent>/.podcast_scraper.lock` as a fast follow if cron overlap is real.

### 8. Dry-run

**Recommend:** **v2 nice-to-have** — “would skip / would retry” from append state. v1 dry-run can stay approximate.

### 9. `run_manifest` / `run_id` with append

**Recommend:** **`run_id`** remains useful for **audit** (who ran what), not for forcing a new output tree on every append. Append runs may log the same stable corpus layout; manifest records **invocation** timestamp separately from **output path**.

## Open questions (still for you to pin down)

1. Exact **algorithm for `stable_feed_id`** (hash length, collision handling).
2. Whether **non-append multi-feed** uses the same flat `feeds/<id>/` layout as append, or keeps nested `run_*` for parity with historical single-feed runs (compatibility vs simplicity).
3. **Incremental** parent `index_corpus` after each feed vs **batch-only** (latency vs cost).

## Changelog of this note

| Date | Change |
| ---- | ------ |
| 2026-04-06 | Initial draft from issue spec + user decisions (A, opt-in append, index.json only) and compatibility / overnight-run goals. |
| 2026-04-06 | Added downstream impact (GI, KG, semantic index, serve): per-feed output dir = zero change; parent-only corpus needs recursive scan or per-feed index. |
| 2026-04-06 | Reframed goal: **unified** KG/GI/semantic search over corpus parent; doc path resolution + composite index keys; phased with #440/#444. |
| 2026-04-06 | GI/KG viewer: corpus root = parent containing `feeds/`; `/api/artifacts` already recursive; search tab waits on unified index. |
| 2026-04-06 | **Failure isolation:** default — one failing feed or episode must not block others; orchestrator per-feed isolation + checklist. |
| 2026-04-06 | **Critical gaps and risks:** run-dir churn, finalize skipped on exception, index vs FS truth, id vs filename, service API, indexing strategy, locking, docs. |
| 2026-04-06 | **Strong recommendations (pre-coding):** explicit parent for N>1, stable feed dirs + no run_ts churn under append, FS-first resume, batch vector index, episode-level checkpoints, defer `service.run` multi-feed. |
| 2026-04-06 | Wording: removed “product call / product judgment”; **Pipeline goal**; open questions = your decisions. |
