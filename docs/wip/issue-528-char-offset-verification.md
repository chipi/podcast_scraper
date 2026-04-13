# Issue #528 — Char offset verification (RFC-072 Phase 5, Step 1)

## Purpose

Before relying on **chunk-to-Insight lift** in semantic search, confirm that **GIL Quote**
`char_start` / `char_end` and **FAISS transcript chunk** metadata use the same transcript
coordinate space (half-open ranges `[start, end)` per `chunk_transcript` and GI schema).

## Command

From the repo root, with the `[server]` / FAISS stack available and a corpus that has
`metadata/*.metadata.json`, `*.gi.json`, and a built index under `search/`:

```bash
podcast verify-gil-chunk-offsets --output-dir /path/to/corpus
```

Optional:

- `--index-path DIR` — override default `<output-dir>/search`
- `--strict` — exit non-zero if overlap rate is below `--min-overlap-rate` (default `0.95`),
  if the verdict is `divergent`, or if no Quote nodes are found
- `--max-samples N` — cap sample quote ids listed per episode when there is no overlap

The tool prints a JSON report with `verdict`, `overlap_rate`, per-episode breakdown, and
`warnings`.

## Interpreting results

- **`aligned`** — overlap rate at least 0.99 and every scanned episode had at least one
  transcript chunk row in the index (no “GI but no chunks” episodes).
- **`mostly_aligned`** — overlap rate at least 0.85 but not fully meeting `aligned`.
- **`divergent`** — overlap rate below 0.85; treat as blocking for lift until offsets or
  indexing are fixed.
- **`no_quotes`** — no Quote nodes in the GI files that were scanned (empty or missing GIL).

If transcript text normalisation (BOM, newlines, Unicode) differs between indexing and GIL
generation, overlap drops and the report should be used to design a mapping layer (see RFC-072
Known Limitations).

## Step 2 (search lift)

When Step 1 is acceptable for your corpus, `GET /api/search` enriches **transcript** hits with an
optional `lifted` object (insight, speaker, topic, quote timestamps) using the same overlap rule
and `bridge.json` display names. No FAISS rebuild is required.
