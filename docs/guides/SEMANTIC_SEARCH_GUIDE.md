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

## Corpus topic clustering (RFC-075)

After **`kg_topic`** rows exist in the FAISS index, you can build **`search/topic_clusters.json`**
with **`topic-clusters`** (see [CLI reference](../api/CLI.md) — section **Topic clusters**).
Optional **`--merge-cil-overrides`** writes merged **`topic_id_aliases`** into
**`cil_lift_overrides.json`** (see **Optional corpus overrides** below). The GI/KG viewer loads
clusters via **`GET /api/corpus/topic-clusters`** when serving the same corpus root. Details:
[RFC-075](../rfc/RFC-075-corpus-topic-clustering.md).

New CLI runs emit **`schema_version`: `"2"`** with distinct ids: **`graph_compound_parent_id`**
(`tc:`…, viewer **TopicCluster** parents only) and **`cil_alias_target_topic_id`** (`topic:`…,
**`topic_id_aliases`** merge target). Older **`topic_clusters.json`** files may still use v1
keys (`cluster_id`, `canonical_topic_id`); readers accept both.

**Search responses:** Membership is **not** copied into FAISS metadata at index time. On each
**`GET /api/search`** (and CLI search), the server **joins** **`search/topic_clusters.json`**
by **`metadata.source_id`** for **`doc_type`: `kg_topic`** rows and attaches
**`metadata.topic_cluster`** with **`graph_compound_parent_id`**, **`canonical_label`**, and
optional **`cil_alias_target_topic_id`** when present. If the JSON file is missing or the topic
is not in any cluster, the field is omitted.

### Id spaces (mental model)

The stack uses **several parallel id systems**; they stay aligned by convention, not by a single
global ontology:

| Lens | Role |
| ---- | ---- |
| **Embedding / FAISS** | Similarity search over chunk vectors; **`kg_topic`** rows use **`topic:{slug}`** as **`metadata.source_id`** (episode-local KG identity). |
| **`topic:…` (KG)** | Canonical **Topic** node ids in **`*.kg.json`** and in the index; CIL **aliases** can merge variants for lift and timelines. |
| **`tc:…` (TopicCluster)** | **Viewer-only** compound parent ids from **`graph_compound_parent_id`** in **`topic_clusters.json`** — grouping in Cytoscape, **not** a FAISS key and **not** the same field as **`cil_alias_target_topic_id`**. |
| **Overrides** | **`cil_lift_overrides.json`** (and optional **`topic_id_aliases`** from **`topic-clusters --merge-cil-overrides`**) tune identity without rewriting KG files. |

Details and field renames (v1 vs v2): [RFC-075 § Artifact schema](../rfc/RFC-075-corpus-topic-clustering.md#3-artifact-schema-topic_clustersjson).

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
`supporting_quotes` (insight rows), optional **`lifted`** on **transcript** rows when
RFC-072 lift applies (see below), and optional **`metadata.topic_cluster`** on **kg_topic**
rows when **`topic_clusters.json`** exists (RFC-075 join; see **Corpus topic clustering** above).

**Quote speaker fields (GitHub [#541](https://github.com/chipi/podcast_scraper/issues/541)):** On
**insight** hits, each entry in **`supporting_quotes`** may include **`speaker_id`** /
**`speaker_name`** mirroring **`.gi.json`**; both are often absent when transcription segments
lack diarization labels. On **transcript** hits, **`lifted.speaker`** / **`lifted.topic`**
carry **`display_name`** from **`bridge.json`** when the graph id resolves; the matched
**`lifted.quote`** may still have audio timestamps while speaker display is missing. Canonical
rules: [Development Guide — GI quote `speaker_id`](DEVELOPMENT_GUIDE.md#gi-quote-speaker-id).

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
  After **RFC-075** clustering, **`topic-clusters --merge-cil-overrides`** can append
  **`topic_id_aliases`** derived from **`topic_clusters.json`**; keys already in the file
  keep their values (hand overrides win).

**Prerequisite:** **Quote** offsets and **index** transcript chunk offsets must refer to the
**same normalised transcript text**. Validate on a real indexed corpus with:

```bash
python -m podcast_scraper.cli verify-gil-chunk-offsets --output-dir /path/to/corpus --strict --min-overlap-rate 0.95
```

or **`make verify-gil-offsets-strict`** (override **`GIL_OFFSET_VERIFY_DIR`**). The verifier
walks discovered **`*.metadata.json`** paths (including **feed-nested** `feeds/.../metadata/`)
so multi-feed outputs are included.

**Verifier JSON:** The tool prints `verdict`, `overlap_rate`, `quotes_total` (all Quote nodes
in scanned GI), `quotes_verifiable_against_index` (quotes in episodes that have at least one
transcript chunk in the index), `quotes_skipped_no_transcript_index`, per-episode breakdown, and
`warnings`. **`overlap_rate`** is **only** over verifiable quotes (episodes with no transcript
vectors in the index are skipped for the ratio so partial indexing does not count as misalignment).

Useful labels:

- **`aligned`** — Overlap rate at least **0.99** on the verifiable subset (episodes without
  indexed transcript chunks may still appear in the report with
  `quotes_skipped_no_transcript_index` on the row).
- **`mostly_aligned`** — Overlap rate at least **0.85** but not meeting `aligned`.
- **`divergent`** — Overlap rate below **0.85**; treat as blocking for lift until offsets or
  indexing are fixed.
- **`no_quotes`** — No Quote nodes in the GI files that were scanned (empty or missing GIL).
- **`no_indexed_transcript_for_quotes`** — GI has Quotes but no transcript chunk rows in the
  index for those episodes (strict mode passes; nothing to compare).

If transcript text normalisation (BOM, newlines, Unicode) differs between indexing and GIL
generation, overlap drops; use the report to align pipelines (RFC-072 Known Limitations).

**Flags:** `--index-path DIR` (default `<output-dir>/search`), `--strict` (non-zero exit when
overlap is below `--min-overlap-rate` or verdict is `divergent`; `no_quotes` and
`no_indexed_transcript_for_quotes` exit zero),
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
