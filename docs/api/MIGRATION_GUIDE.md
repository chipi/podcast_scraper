# API Migration Guide

Documentation for migrating between major and minor versions of the `podcast_scraper` API.

---

## v2.5.0 to v2.6.0 (Current) {#v260-viewer-and-http}

v2.6.0 adds **viewer and HTTP** capabilities; the core library API is unchanged.

### Additive: Corpus Library and index operations

- **New FastAPI routes** under `/api/corpus/` (feeds, episodes, detail, similar episodes). See [Server Guide](../guides/SERVER_GUIDE.md) and [RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md).
- **`POST /api/index/rebuild`** — background vector index rebuild; poll **`GET /api/index/stats`** for `rebuild_in_progress` and errors.
- **Vue viewer** — Library tab and dashboard hooks aligned with the above.

**Migration:** `pip install -e '.[server]'` if you call the HTTP API or run `podcast serve`. No changes required for `run_pipeline` / `service.run` callers.

### Additive: Corpus Digest and health discovery (RFC-068)

- **`GET /api/corpus/digest`** — rolling-window digest of recent episodes (feed-diverse) plus optional semantic topic bands when a vector index exists. See [Server Guide](../guides/SERVER_GUIDE.md) and [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md).
- **`GET /api/health`** — response may include **`corpus_digest_api`** (`bool`). **`false`** disables Digest / glance in the viewer. If the field is **omitted** but **`corpus_library_api`** is true, the GI/KG viewer **infers** digest is available (legacy health JSON); if the running process is too old to mount **`GET /api/corpus/digest`**, the digest request fails — upgrade/restart the API from a current `[server]` install.

---

## v2.3.2 to v2.4.0

v2.4.0 introduces a multi-provider ecosystem and changes several defaults.

### ⚠️ Breaking Behavior Changes

These are not code-breaking but change the default behavior of the pipeline:

1. **Automatic Transcription**: `transcribe_missing` now defaults to `true`.
   - **Migration**: If you want to only download existing transcripts, explicitly set `transcribe_missing: false` in your config.
2. **Whisper Model**: The default `whisper_model` changed from `base` to `base.en`.
   - **Migration**: For non-English podcasts, you must now explicitly set `whisper_model: base` (or another multilingual model).
3. **Output Structure**: Transcripts and metadata are now placed in subdirectories.
   - **Migration**: Update any scripts that assume all files are in the root run directory.

### 🚀 Multi-Provider Configuration

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

# Both will return the same string, e.g., "2.6.0"
print(podcast_scraper.__version__)
print(podcast_scraper.__api_version__)
```
