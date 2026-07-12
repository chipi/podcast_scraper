# API Migration Guide

Documentation for migrating between major and minor versions of the `podcast_scraper` **API surface** —
endpoint additions, response-shape moves, config renames, upgrade recipes for callers of the HTTP API
and the Python library.

**On-disk corpus migrations** (moving a deployed corpus across a release boundary) live in a separate
framework — see [Corpus upgrade framework (#862)](../guides/CORPUS_UPGRADE.md). The two guides span the
same release cadence; read both when moving both callers and a deployed corpus across a version.

**Read-time schema shims** for legacy artifacts live in
`src/podcast_scraper/migrations/gil_kg_identity_migrations.py` — used by the server and graph build to
accept older artifact shapes without an in-place upgrade. See the [Corpus upgrade framework](../guides/CORPUS_UPGRADE.md#other-migration-surfaces-in-the-repo)
guide's "Other migration surfaces" section.

> **For agents updating this guide:** every version bump on `pyproject.toml` needs a matching
> `vN.N.0 to vN.N+1.0` section here documenting API-visible changes. If the release also ships a
> framework migration (`upgrade/migrations/mNNNN_*`), the guide's version section MUST cite it and
> link to [CORPUS_UPGRADE.md](../guides/CORPUS_UPGRADE.md). The trigger conditions for adding a
> framework migration are documented in
> [CORPUS_UPGRADE.md → When is a migration required?](../guides/CORPUS_UPGRADE.md#when-is-a-migration-required)
> and in `AGENTS.md` → "Migrations — WHEN to add one".

---

## v2.6.0 to v2.7.0 (Current) {#v270-lancedb-hybrid-and-v3-schema}

v2.7.0 completes the RFC-090 hybrid-retrieval move and lands the RFC-097 v3 GI schema.
On-disk artifacts change shape; API surface is largely additive but the underlying
storage and schema semantics shift. **Existing corpora need one command to migrate:**

```bash
CORPUS_DIR=/path/to/corpus_root
make upgrade-check   CORPUS_DIR=$CORPUS_DIR  # exits 2 if migrations pending
make upgrade-corpus  CORPUS_DIR=$CORPUS_DIR  # apply (idempotent, --yes)
make upgrade-verify  CORPUS_DIR=$CORPUS_DIR
```

See [Corpus upgrade framework (#862)](../guides/CORPUS_UPGRADE.md) for the full
registered-migrations list and the ledger model. Fresh corpora produced by v2.7.0's
pipeline land at the current version and need no upgrade.

### Storage: FAISS retired → two-tier LanceDB (ADR-099 / #995)

FAISS is removed from the hot path. Search now runs on a two-tier LanceDB index
(transcript segments + GIL insights) with BM25 + dense vector retrieval combined via
RRF (RFC-090). The single search path is LanceDB when a two-tier index exists.

**API impact:** none for callers of the search endpoints — the response shape is
preserved (compound results). Callers that reached into `search/` sidecar files
directly must re-read; the on-disk layout is different.

**Corpus impact:** the `m0001_faiss_to_lance` migration is now a recorded no-op
(FAISS was already retired before the ledger existed); `m0002_two_tier_native_reindex`
builds the two-tier LanceDB index natively from artifacts for pre-2.7 corpora. Both
run automatically via `make upgrade-corpus`.

### GI schema: v3.0 typed mentions (RFC-097 / #1036)

Grounded Insight artifacts (`*.gi.json`) move to `schema_version: 3.0`:

- Legacy `MENTIONS` edges (Insight → Person/Org) are rewritten to typed
  `MENTIONS_PERSON` / `MENTIONS_ORG` based on the target id's prefix.
- Legacy `Insight.insight_type` vocab (`fact` / `opinion`) normalises to the v3 vocab
  (`claim` / `observation`); out-of-vocab labels become `unknown`.
- KG-side `MENTIONS` edges (Topic → Episode discovery) stay untouched by design.

**Migration:** `m0003_gi_v3_typed_mentions` handles this per-file when
`make upgrade-corpus` runs. Idempotent — a `.gi.json` already at 3.0 with typed edges
passes through unchanged.

**API impact:** consumers reading `gi.json` directly must accept both edge-type sets
until every corpus in flight has been upgraded. The read-time helpers in
`podcast_scraper.migrations.gil_kg_identity_migrations.migrate_gi_document_v3`
provide a runtime shim if you cannot upgrade a corpus in place.

### Additive: enrichment layer + read-time cross-episode surfaces

The RFC-088 enrichment layer (`podcast enrich` / `make enrich`) adds a versioned set
of deterministic + ML enrichers producing sidecar artifacts under
`<corpus>/enrichments/` and `<corpus>/metadata/enrichments/`. Read-time server APIs
join these into episode / person / topic surfaces without a schema bump. Fresh
corpora produced by v2.7.0's pipeline emit these by default; older corpora can be
enriched retroactively via `make enrich CORPUS=<dir>`.

### Additive: instance-to-instance corpus portability (#1175)

`make export-corpus` + `make import-corpus` produce a portable snapshot (identical
format to the CI backup path) and restore it locally — no `gh` dependency. Useful
for laptop ↔ VPS moves, prod ↔ codespace transplants, and airgapped restores. See
[Corpus airgap runbook](../guides/CORPUS_AIRGAP_RUNBOOK.md).

### Deployment: `upgrade-check` gates on stale corpora (#1176)

`scripts/ops/restore_corpus_from_tarball_host.sh` — invoked by
`prod-restore-corpus.yml` / `drill-restore-corpus.yml` — now runs
`podcast upgrade run --yes` between corpus extract and container recycle. A
restore that lands an older snapshot onto a newer code deploy migrates the corpus
before the api boots. Local restores (`make restore-corpus` /
`make import-corpus`) should be followed by the same `make upgrade-corpus` step;
see [Corpus airgap runbook](../guides/CORPUS_AIRGAP_RUNBOOK.md#post-import-apply-pending-upgrade-migrations).

### Related

- [Corpus upgrade framework (#862)](../guides/CORPUS_UPGRADE.md) — registered-migrations
  list, ledger model, adding a migration.
- [Corpus snapshot manifest and restore](../guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md) —
  SSOT for backup / restore surfaces.
- [RFC-090](../rfc/RFC-090-hybrid-retrieval.md), [ADR-099](../adr/ADR-099-lancedb-first-single-index-search.md),
  [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md), [ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md).

---

## v2.5.0 to v2.6.0 {#v260-viewer-and-http}

v2.6.0 adds **viewer and HTTP** capabilities; the core library API is unchanged.

### Additive: Corpus Library and index operations

- **New FastAPI routes** under `/api/corpus/` (feeds, episodes, detail, similar episodes). See [Server Guide](../guides/SERVER_GUIDE.md) and [RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md).
- **`POST /api/index/rebuild`** — background vector index rebuild; poll **`GET /api/index/stats`** for `rebuild_in_progress` and errors.
- **Vue viewer** — Library tab and dashboard hooks aligned with the above.

**Migration:** `pip install -e '.[dev]'` if you call the HTTP API or run `podcast serve`. No changes required for `run_pipeline` / `service.run` callers.

### Feeds API and jobs: structured `feeds.spec.yaml` (RFC-077 / #626)

- **Canonical corpus feeds file** is **`feeds.spec.yaml`** at the corpus root (root object with **`feeds`** array). The viewer **`GET`/`PUT /api/feeds`** contract uses JSON **`{ "feeds": [...] }`** (not **`urls`**).
- **Pipeline jobs** subprocess passes **`--config`** and **`--feeds-spec`** (when the file exists) to `python -m podcast_scraper.cli`, matching the main CLI flags. Manual runs often add **`--profile <name>`** alongside **`--config`** for the same merge semantics; see [CLI.md — Quick Start](CLI.md#quick-start).
- **Migration from `rss_urls.list.txt`:** convert one URL per line to `feeds: ["https://...", ...]` in YAML or JSON, save as **`feeds.spec.yaml`**, or use **`config/examples/feeds.spec.example.*`** as a template. **`--rss-file`** remains supported on the CLI for line lists but is not what the Feeds API or job runner use.

### Additive: Corpus Digest and health discovery (RFC-068)

- **`GET /api/corpus/digest`** — rolling-window digest of recent episodes (feed-diverse) plus optional semantic topic bands when a vector index exists. See [Server Guide](../guides/SERVER_GUIDE.md) and [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md).
- **`GET /api/health`** — response may include **`corpus_digest_api`** (`bool`). **`false`** disables Digest / glance in the viewer. If the field is **omitted** but **`corpus_library_api`** is true, the GI/KG viewer **infers** digest is available (legacy health JSON); if the running process is too old to mount **`GET /api/corpus/digest`**, the digest request fails — upgrade/restart the API from a current `[dev]` install.

---

## v2.3.2 to v2.4.0

v2.4.0 introduces a multi-provider ecosystem and changes several defaults.

### Breaking Behavior Changes

These are not code-breaking but change the default behavior of the pipeline:

1. **Automatic Transcription**: `transcribe_missing` now defaults to `true`.
   - **Migration**: If you want to only download existing transcripts, explicitly set `transcribe_missing: false` in your config.
2. **Whisper Model**: The default `whisper_model` changed from `base` to `base.en`.
   - **Migration**: For non-English podcasts, you must now explicitly set `whisper_model: base` (or another multilingual model).
3. **Output Structure**: Transcripts and metadata are now placed in subdirectories.
   - **Migration**: Update any scripts that assume all files are in the root run directory.

### Multi-Provider Configuration

v2.4.0 replaces specific provider flags with a unified provider system:

- New fields: `transcription_provider`, `speaker_detector_provider`, `summary_provider`.
- Supported providers: `whisper`, `spacy`, `transformers` (local), and cloud providers like `openai`, `anthropic`, `mistral`, etc.

---

## v1.0 to v2.0

Version 2.0 refactored the monolithic v1.0 into a clean modular architecture.

### Modular Architecture

**Before (v1.0):** Monolithic `podcast_scraper.py` file with no formal public API.

**After (v2.0):** focused modules with 4 primary public exports:

```python
from podcast_scraper import Config, load_config_file, run_pipeline, cli
```

### New Usage Pattern

```python
import podcast_scraper

# Configuration
config = podcast_scraper.Config(
    rss_url="https://example.com/feed.xml",
    output_dir="./transcripts",
    max_episodes=10,
)

# Run pipeline
count, summary = podcast_scraper.run_pipeline(config)
```

---

## Version History

| Version | Date | Highlights |
| ------- | ---- | ---------- |
| **v2.7.0** | 2026-07 | Two-tier LanceDB hybrid retrieval (RFC-090 / ADR-099, FAISS retired), v3 GI schema with typed mentions (RFC-097), RFC-088 enrichment layer, `make export-corpus` / `import-corpus` (#1175), upgrade-check gates in restore paths (#1176). |
| **v2.6.0** | 2026-04 | Corpus Library `/api/corpus/*`, index rebuild API, viewer Library UX, RFC-064 profile tooling. |
| **v2.5.0** | 2026-02 | LLM provider expansion, production hardening, MPS exclusive mode, LLM metrics. |
| **v2.4.0** | 2026-01 | Multi-provider ecosystem, production defaults, cache CLI. |
| **v2.3.0** | 2025-11 | Added service API and episode summarization. |
| **v2.2.0** | 2025-11 | Metadata generation (JSON/YAML). |
| **v2.1.0** | 2025-11 | Automatic speaker detection (NER). |
| **v2.0.0** | 2025-11 | Modular architecture foundation. |
| **v1.0.0** | 2025-11 | Initial monolithic release. |

## Checking API Version

```python
import podcast_scraper

# Both will return the same string, e.g., "2.7.0"
print(podcast_scraper.__version__)
print(podcast_scraper.__api_version__)
```
