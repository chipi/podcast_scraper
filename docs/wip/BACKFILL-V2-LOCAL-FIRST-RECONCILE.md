# Backfill v2 — local-first reconcile (#55, follow-up to #1796)

**Status:** implementing · **Branch:** `feat/backfill-local-first-reconcile` (off `main`) · **Date:** 2026-08-20
**Epic:** audio cold-storage archive (#1788) · **Supersedes behaviour merged in:** #1796 (#1787)

## Why this exists

The `archive backfill` merged in #1796 only knows one move: **re-download every
missing episode from the publisher**. That is wrong for the world we're actually in:

1. **Prod already holds hundreds of episodes' worth of local audio** — ~52.7 GB
   under `media/` and ~12.6 GB under `.podcast_scraper/audio-cache/`, left by past
   runs. Those are the **original bytes** that produced the transcripts.
2. Re-downloading throws those originals away and replaces them with
   **dynamic-ad re-encodes** (`byte_identical=False`) — and any episode that
   rolled off the publisher's window is lost entirely.
3. The per-run eviction (#1787) only deletes local audio **confirmed in cold**.
   Until the existing local audio is uploaded to cold, it is **stranded**:
   neither preserved nor reclaimable. So "upload what we already have" is the
   **prerequisite** for reclaiming the ~65 GB, not just a nicety.

## The model — backfill becomes a reconcile pass

Per selected episode, in order:

| Case | Condition | Action | Outcome | Provenance |
|---|---|---|---|---|
| **Already in cold** | `already_archived(guid)` | nothing to fetch; local copy is cleaned in the sweep step | `already_present` | — |
| **Harvest local original** | not in cold, a local original exists (audio-cache by GUID, or `media/` via `audio_relpath`) | upload the **original** bytes to cold — no download | `harvested` | `byte_identical=True`, origin `backfill_harvest_local` |
| **Download** | not in cold, no local copy | fetch from publisher, **with retry + backoff** | `stored` / `rolled_off` / `fetch_failed` | `byte_identical=False`, origin `backfill_refetch` |
| **No URL** | metadata has no `media_url` | skip | `no_media_url` | — |

Then a **cleanup sweep** (`offload.sweep_corpus`) evicts every local `media/`
file now confirmed in cold + size-matched — this is both the "move" completion
(reclaim the just-harvested copies) and the operator's refinement #1
("already in cold → also make sure there's no local garbage"). Size-guarded
(advisor H1): a file whose cold size ≠ local size is **kept**, never destroyed.

## Dry-run reports the split before anything moves

```
archive backfill (dry-run) — nothing has been fetched
  feed            in-corpus  in-cold  move-local  download
  <feed>              120         20          70        30
  ...
  totals: 700 episode(s) — 120 already in cold, 300 to move from local,
          280 to download, 0 without a media_url
  estimated download: >= NN.N GB (floor)
  cleanup: would evict 300 local file(s) (48.2 GB) already in cold
```

## Resiliency — the deliberate compromise

The operator asked for the download resilience "without overdoing it". Two options were weighed:

- **Reuse the pipeline's `http_download_to_file`** (httpx `RetryTransport`,
  exponential backoff for free). **Rejected:** `fetch_url` swallows the HTTP
  status code, which **collapses `rolled_off` (404/410, unrecoverable) into
  `fetch_failed` (retryable)**. The whole dry-run/result split depends on that
  distinction.
- **Keep the `urllib` `_download`** (it sees `exc.code`) **+ a small bounded
  retry wrapper** — 3 attempts, exponential backoff, **skip-retry on 404/410**.
  Plus the existing per-host `HostRateLimiter` (the "pauses between hitting the
  same feed"). **Chosen** — less work, keeps the rolled-off signal.

## What's new vs. reused

- **New:** `harvested` outcome, `build_local_lookup(corpus)` (guid → local
  original path, from run `media/` + audio-cache), `_download_with_retry`,
  `record_harvest_provenance`, 4-way dry-run/result formatting, `--max-retries`.
- **Reused as-is:** `already_archived`, `store_via`, `record_pipeline_provenance`,
  `HostRateLimiter`, and `offload.sweep_corpus` / `evict_run_dir` (the
  size-guarded cleanup — refinement #1 is literally the existing sweep).

## Safety invariants (unchanged from #1787)

- Eviction only ever touches files under `<run_dir>/media/`, resolved from
  `content.audio_relpath` and re-checked to sit under `media/`.
- A local file is deleted only when cold holds the audio **and** cold size ==
  local size. Unknowable cold size → keep.
- Harvest uploads only when cold is a **miss** for that GUID (so `store_via`
  genuinely uploads rather than dedup-collides) → `byte_identical=True` is honest.
- No transcription, no enrichment, no LLM calls — download/upload/evict only.

## Not covered / out of scope for this pass

- **audio-cache eviction.** The cleanup sweep reclaims `media/` (the 52.7 GB);
  it does **not** delete the `.podcast_scraper/audio-cache/` copies (12.6 GB).
  The cache is the reproducibility store and is pruned by the backup path, not here.
- **Selector-scoped cleanup.** The cleanup sweep runs corpus-wide regardless of
  `--feed`/`--since` (it is safe — confirmed-in-cold + size-guarded — but it is
  not limited to the selected feeds). Documented in the run output.
- The prod workflow (`backfill-audio-prod.yml`) input surface is unchanged;
  `--max-retries` defaults to 3 and is not exposed as a workflow input yet.
