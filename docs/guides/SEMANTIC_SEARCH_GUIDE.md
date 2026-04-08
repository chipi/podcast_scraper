# Semantic corpus search (RFC-061)

Meaning-based retrieval over **Grounded Insights** (insights and quotes), **summary
bullets**, **transcript chunks**, and **Knowledge Graph** **Topic** / **Entity** nodes
(when `kg.json` is present) in a pipeline output directory. Phase 1 uses a
local **FAISS** index (`faiss-cpu`) and the same embedding stack as GIL evidence
(`embedding_loader`, sentence-transformers).

## When to use it

- Cross-episode questions: “What do my podcasts say about X?” without exact keywords.
- **`gi explore --topic`**: when `<output_dir>/search/vectors.faiss` exists, topic
  matching uses the vector index first (metadata is scanned to find `gi.json` paths;
  only matching episodes load full artifacts), then falls back to substring/topic-label
  matching if the index is missing, fails, or returns no hits.
- Ad hoc CLI search: **`podcast search`** with filters (`--type`, `--feed`, `--since`,
  `--speaker`, `--grounded-only`, `--top-k`, `--format`).

## Enable indexing

### Config (YAML or CLI)

In your config file (see `config/examples/config.example.yaml`):

```yaml
vector_search: true
# Optional overrides:
# vector_index_path: search          # default relative dir under output_dir
# vector_embedding_model: minilm-l6
# vector_chunk_size_tokens: 300
# vector_chunk_overlap_tokens: 50
```

`vector_search: true` runs the embed-and-index step after pipeline finalize (when
metadata and content are available). Default index directory is **`output_dir/search`**.

### Multi-feed corpus parent (GitHub #505)

When **`output_dir`** is a corpus parent with a **`feeds/`** directory (two or more feeds in
one run), the indexer discovers episode metadata under each
**`feeds/<stable_id>/…/metadata/`** tree (including nested `run_*` folders). Artifact paths
in metadata are resolved relative to each episode’s workspace (the directory that contains
that feed’s `metadata/`). Vector row ids and fingerprint keys use a composite
**`(feed_id, episode_id)`** scope when `feed_id` is present in metadata, so GUID collisions
across feeds cannot clobber FAISS rows.

Per-feed pipeline runs set **`skip_auto_vector_index: true`** internally when
`vector_search` and FAISS are enabled; after all feeds complete, the CLI or service builds
**one** index at **`<corpus_parent>/search`**.

### Manual index / rebuild

```bash
python -m podcast_scraper.cli index --output-dir /path/to/run
python -m podcast_scraper.cli index --output-dir /path/to/run --rebuild
python -m podcast_scraper.cli index --output-dir /path/to/run --stats
```

For a multi-feed tree, **`--output-dir`** should be the **corpus parent** (the directory that
contains **`feeds/`**), not an individual feed subdirectory.

### Index upgrade / stale rows (multi-feed)

If you **change embedding model**, **chunking settings**, or **composite fingerprint scope** (feed +
episode keys — GitHub #505), existing FAISS rows may no longer match metadata on disk. Rebuild the
parent index with:

```bash
python -m podcast_scraper.cli index --output-dir /path/to/corpus_parent --rebuild
```

Until you rebuild, search may return hits that no longer map cleanly to current files, or miss new
episodes. See [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md) for manifest /
summary contracts.

## Query

```bash
python -m podcast_scraper.cli search "your question" --output-dir /path/to/run
python -m podcast_scraper.cli search "your question" --output-dir /path/to/run --format json --top-k 20
```

Use **`--index-path`** if the index is not under `<output_dir>/search`.

## Web UI (GI / KG Viewer v2)

The Vue viewer can run the **same** corpus search as the CLI via **`GET /api/search`** when
the FastAPI server is up (`pip install -e ".[server]"` + `[ml]` for FAISS/embeddings) and
the index exists under `<corpus>/search/`. Set **Corpus root** in the sidebar to your
pipeline output directory. See
[web/gi-kg-viewer/README.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md)
(*Semantic search*) and [README.md](https://github.com/chipi/podcast_scraper/blob/main/README.md)
(*GI / KG Viewer*).

## `gi explore` and `gi query`

- **`gi explore --topic "…"`** — If a FAISS index is present at **`output_dir/search`**
  (or the path implied by your pipeline layout), insights are ranked by embedding
  similarity to the topic string. Otherwise behavior is unchanged (substring + Topic
  labels).
- **`gi query`** — Questions that map to topic/speaker filters use the same
  **`run_uc5_insight_explorer`** path, so they benefit from the index when present.

## Requirements

- **`pip install -e ".[ml]"`** (or equivalent) for sentence-transformers and FAISS.
- Embedding models must be available locally (or cached); search uses
  **`allow_download=False`** — run the pipeline or `index` once with network if needed.

## Related docs

- **RFC-061:** `docs/rfc/RFC-061-semantic-corpus-search.md`
- **PRD-021:** `docs/prd/PRD-021-semantic-corpus-search.md`
- **GIL CLI:** [Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md)
