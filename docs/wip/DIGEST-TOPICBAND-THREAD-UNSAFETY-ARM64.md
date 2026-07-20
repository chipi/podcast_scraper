# /api/corpus/digest topic-band thread-unsafety on arm64 — followup

Status: **fixed on main via warmup shape** (#1205 + siblings on
`origin/main`; commits `b6955854` → `8b5a1c07` → `62c049e5` →
`0fe0854b`). Superseded my earlier unconditional-single-thread
workaround from `feat/graph-v3`; the rebase onto main took main's
shape, my regression test (`tests/integration/server/test_corpus_digest_topics_warmup.py`)
was rewritten to lock the new invariant in. File:
`src/podcast_scraper/server/routes/corpus_digest.py`.

Left in `docs/wip/` because the underlying arm64 first-touch fragility
in sentence-transformers / sentencepiece / LanceDB is **not fixed
upstream** — the warmup shape works around it. Future PRs that want
concurrency-from-t=0 must either verify the upstream fix landed or
add their own warmup + regression test.

## Symptom

Running the ML stack-test (`make stack-test-ml-ci`) on macOS arm64 (Docker for
Mac, M1/M2), the `api` container exits **139** (SIGSEGV) partway through the
first `GET /api/corpus/digest` after a pipeline job completes. Downstream
smoke tests fail with `502 Bad Gateway` because the api container is dead.

Faulthandler traceback ends inside
`src/podcast_scraper/server/routes/corpus_digest.py:303` (the
`ThreadPoolExecutor.map(_band, topics_cfg)` call). The Python extension-module
list at the crash included `torch._C.*`, `sentencepiece._sentencepiece`,
`sklearn.*`, `scipy.*`, `sentence_transformers`-derived pyarrow, etc.

## Diagnosis

`_band` calls `_topic_band_for_query` → `run_corpus_search` → LanceDB search
(sentence-transformers query embedding + tokenization). The **outer**
`ThreadPoolExecutor.map` spawns up to 8 worker threads and asks each of them
to touch sentence-transformers / sentencepiece for the first time
concurrently. Native C extensions in that stack have known thread-safety
issues on **first import / first call** — the first thread that touches
sentencepiece's C tokenizer while other threads are simultaneously entering
the same code path SIGSEGVs the interpreter. The inner
`ThreadPoolExecutor(max_workers=1)` inside `_topic_band_for_query` is only
used as a timeout wrapper, so it doesn't hide the outer concurrent access.

The problem is **first-touch** concurrency, not steady-state — a warm process
after one search has succeeded is fine. But the api container in stack-test
is freshly started before each Playwright run, so the very first digest call
after the pipeline job is the crash trigger.

Not reproduced on x86_64 linux CI (untested this session; historically has
been passing). Suspect the specific torch / sentencepiece wheel builds on
arm64 are more susceptible.

## Current workaround

Sequential `map()` in place of `ThreadPoolExecutor.map()` — see the comment
block in `corpus_digest.py` above the change. Unconditional (no env-var
knob) so stack-test and prod run the same code path; a stack-only branch
would defeat the purpose of stack-test as a fidelity check.

Cost of the workaround: digest with N topic bands takes
approximately N × single-search-latency instead of `ceil(N/8) ×
single-search-latency`. Stack-test fixture has ~1 topic band so the cost is
zero; the fixture where operators feel it is `digest_topics_configured`
> 8 with real user queries. Not a problem for current corpus sizes; will
become one if operators wire many bands.

## Options for a real fix (pick one when we need concurrency back)

1. **Warm the embedding model on api startup.** Force sentence-transformers
   / sentencepiece to fully initialize before uvicorn accepts requests. If
   the crash is first-touch race, warmup eliminates it. Cost: adds a few
   hundred ms to api container boot; easy to implement in the lifespan hook
   in `src/podcast_scraper/server/app.py`. Verify by restoring the 8-worker
   pool and rerunning stack-test-ml-ci twice.

2. **Serialize the shared tokenizer with a lock.** Wrap `run_corpus_search`
   (or the sentence-transformer encoder call) in a `threading.Lock()` so
   only one thread ever enters the C extension at a time. Effectively
   single-threads the search but keeps the concurrent I/O around it.
   Cost: still slow.

3. **Move to process-level parallelism.** `ProcessPoolExecutor` in place of
   `ThreadPoolExecutor`. Cost: forks aren't cheap; sentence-transformers
   loading in each worker is slow on cold-start. Only worth it for very
   large topic-band counts.

4. **Upgrade / patch upstream.** Check sentencepiece + torch release notes
   for arm64 tokenizer thread-safety fixes. Cost: research + wheel
   availability.

Recommended path when this comes up: try #1 first (cheapest, most targeted
diagnosis), then measure.

## Repro

```bash
# On macOS arm64 (M1/M2) with Docker Desktop:
env -u NODE_OPTIONS make stack-test-ml-ci
```

Before the workaround: `docker ps -a` shows `compose-api-1  Exited (139)`
mid-run. After the workaround: 27 stack tests pass.

## History

- 2026-07-18 — workaround landed in `feat/graph-v3` HD hardening batch
  (commit `1db52bea`).
