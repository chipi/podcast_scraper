# Semantic corpus search (RFC-061)

Meaning-based retrieval over **Grounded Insights** (insights and quotes), **summary
bullets**, **transcript chunks**, and **Knowledge Graph** **Topic** / **Entity** nodes
(when `kg.json` is present) in a pipeline output directory. The **shipped** implementation
uses a local **FAISS** index (`faiss-cpu`) and the same embedding stack as GIL evidence
(`embedding_loader`, sentence-transformers). **Qdrant**, remote services, and other
platform-scale options are **not implemented** — see **Draft** [RFC-070](../rfc/RFC-070-semantic-corpus-search-platform-future.md).

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

Response hits are **JSON objects** with `doc_id`, `score`, `metadata`, `text`, optional
`supporting_quotes` (insight rows), and optional **`lifted`** on **transcript** rows when
RFC-072 lift applies (see below).

## Chunk-to-Insight lift and offset verification (RFC-072 / #528) {#chunk-to-insight-lift-and-offset-verification-rfc-072--528}

**Lift:** For hits whose metadata has **`doc_type`: `transcript`**, the server may attach a
**`lifted`** dictionary: **insight** (id, text, grounded, optional `insight_type` /
`position_hint`), **speaker** and **topic** (canonical id + **`display_name`** from
**`bridge.json`** when present), and **quote** timestamps from the matched **Quote** node.
Matching uses **half-open overlap** between the chunk’s **`char_start` / `char_end`** and a
**Quote** span, then walks **SUPPORTED_BY**, **ABOUT**, and **SPOKEN_BY** edges in `gi.json`.
No FAISS rebuild is required; this is **query-time** enrichment.

**Response counters:** On successful search, the JSON may include **`lift_stats`** with
**`transcript_hits_returned`** (rows in that response with `doc_type: transcript`) and
**`lift_applied`** (those rows carrying a non-null **`lifted`** object). Counts apply to
the paginated **`top_k`** slice after server-side KG surface dedupe.

**Optional corpus overrides:** A file **`cil_lift_overrides.json`** at the corpus root
(same directory you pass as **`path`** / `--output-dir`) can tune lift without reindexing:

- **`transcript_char_shift`** (integer) — added to each transcript hit’s **`char_start`** /
  **`char_end`** before overlap with **Quote** spans (fixed offset when index and GI use
  different normalised text bases).
- **`entity_id_aliases`** / **`topic_id_aliases`** — string-to-string maps; **speaker** /
  **topic** ids from the graph are resolved through the map before **`bridge.json`**
  **`display_name`** lookup and before emitting **`lifted.speaker.id`** / **`lifted.topic.id`**.

**Prerequisite:** **Quote** offsets and **index** transcript chunk offsets must refer to the
**same normalised transcript text**. Validate on a real indexed corpus with:

```bash
python -m podcast_scraper.cli verify-gil-chunk-offsets --output-dir /path/to/corpus --strict --min-overlap-rate 0.95
```

or **`make verify-gil-offsets-strict`** (override **`GIL_OFFSET_VERIFY_DIR`**). The verifier
walks discovered **`*.metadata.json`** paths (including **feed-nested** `feeds/.../metadata/`)
so multi-feed outputs are included.

**Verifier JSON:** The tool prints `verdict`, `overlap_rate`, per-episode breakdown, and
`warnings`. Useful labels:

- **`aligned`** — Overlap rate at least **0.99** and every scanned episode had at least one
  transcript chunk row in the index (no “GI but no chunks” episodes).
- **`mostly_aligned`** — Overlap rate at least **0.85** but not meeting `aligned`.
- **`divergent`** — Overlap rate below **0.85**; treat as blocking for lift until offsets or
  indexing are fixed.
- **`no_quotes`** — No Quote nodes in the GI files that were scanned (empty or missing GIL).

If transcript text normalisation (BOM, newlines, Unicode) differs between indexing and GIL
generation, overlap drops; use the report to align pipelines (RFC-072 Known Limitations).

**Flags:** `--index-path DIR` (default `<output-dir>/search`), `--strict` (non-zero exit when
overlap is below `--min-overlap-rate`, verdict is `divergent`, or no quotes were found),
`--max-samples N` (cap sample quote ids listed per episode when there is no overlap). For
**`make verify-gil-offsets-strict`**, set **`GIL_OFFSET_VERIFY_DIR`** for the corpus root and
optionally **`GIL_OFFSET_MIN_RATE`** (default **0.95**).

**See also:** [GIL / KG / CIL cross-layer](GIL_KG_CIL_CROSS_LAYER.md),
[RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) (Phase 5 / Known
Limitations).

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

- **RFC-061 (FAISS, shipped):** [RFC-061](../rfc/RFC-061-semantic-corpus-search.md)
- **RFC-072 (CIL, bridge, lift):** [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md)
- **RFC-070 (platform / future, Draft):** [RFC-070](../rfc/RFC-070-semantic-corpus-search-platform-future.md)
- **PRD-021:** [PRD-021](../prd/PRD-021-semantic-corpus-search.md)
- **GIL CLI:** [Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md)
- **Cross-layer map:** [GIL / KG / CIL cross-layer](GIL_KG_CIL_CROSS_LAYER.md)
