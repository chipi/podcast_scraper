# Arc: LLM budgets, context limits, and registry plumbing

**Opened:** 2026-09-13. **Status:** in progress — 4 shipped, 3 issues open, 1 fix uncommitted.

One arc, because every item below is the same underlying mistake in a different place: **a number
that governs an LLM call lives somewhere that cannot see what it is governing.** A module constant
that cannot see the model. A registry value that never reaches the provider. An output budget that
cannot see the prompt. A transcript clip that cannot see the context window.

Read this before picking up any individual issue — several of them look independent and are not.

## The shipped work (on `main`, NOT deployed as of writing)

Prod runs `sha-1ac4902`. None of this is live, so none of it can be measured yet.

| commit | what | addresses |
| --- | --- | --- |
| `320f2db0` | `response_format={"type":"json_object"}` on the bundled quote call | the `stop` half of quote failures (78 of 293) |
| `320f2db0` | `Extra data` salvage via `raw_decode` in the bundle parser | a handful within that half |
| `320f2db0` | `gil_evidence_quote_bundle_chunk` promoted from module constant to declared field, 10 -> 8 | nothing measured; a fit bound |
| `08ce727a` | duration measured from media/transcript when RSS omits it | `duration_seconds=0` on feeds whose publisher sends no tag |
| `9fe7c703` | capture the failing reply on a bundled-quote parse failure | evidence for #1893 |
| `303971d4` | work-list of 21 quote-damaged episodes | repair after deploy |

**Uncommitted:** speaker-detection description clip (#2011), 11 tests passing.

## The measured baselines

Everything below is measured, not estimated. Re-measure against these rather than re-deriving.

| quantity | value | source |
| --- | --- | --- |
| bundled quote parse failures | **293 / 1,086 episodes (27%)** | prod job logs 2026-08-10..09-11 |
| — `finish_reason=length` (runaway) | **215 (73%)** | same |
| — `finish_reason=stop` (bad JSON) | **78 (27%)** | same |
| vLLM server context rejections | **565 in 30 days**, ~7/day ongoing | VictoriaLogs 2026-08-14..09-13 |
| speaker-detection 400s | **13** | same |
| pipeline throughput | **~5.2x realtime**, ~63 min mean episode | Batch B/C runs |
| storage | **~0.28 GB / 100 episodes** | audio offloads to cold storage |

**Two measurement traps**, both hit during this investigation:

1. `/api/jobs/{id}/log` does **not** carry the vLLM container's stderr. Counting server 400s there
   yields 34; VictoriaLogs yields 565 for the same window. **Count context rejections in
   VictoriaLogs.**
2. `/tokenize` on `tests/fixtures/transcripts/v2/*` gives 4.7-4.9 chars/token. That is synthetic
   clean prose. Prod runs `screenplay: true, diarize: true`, so real transcripts carry speaker
   labels and timestamps and tokenise denser — which is why the 3.5 estimate at
   `openai_provider.py:255` tracks reality nearly 1:1. **Do not re-derive the ratio from fixtures.**

## The open items

### #1893 — output budget vs served context (the `length` failures, 73%)

Every truncating call consumes its **entire** budget whatever the budget is: `5120/5120` at ten
insights, and `1280/1280` at a bisected batch of **two** already running at 640 tokens/insight. One
broke at char 15 of a 3,834-char document — a single unterminated string running to the ceiling.

**Batch size is therefore NOT the lever**, and the chunk 10 -> 8 change must not be sold as the fix.
Two candidate causes needing opposite treatments:

* **decoding loop** -> a penalty. Note `presence_penalty: 1.5` is ALREADY in the registry for this
  model and never reaches the wire — see #2051. That is the cheapest thing to try.
* **genuine over-generation** -> a budget computed against served context minus actual prompt.

Local reproduction is unreliable: it fired once in seven full GI builds and never in 36 isolated
quote calls. `gi_require_grounding` drops the insights that lost their quotes, so the artifact's
survivors are the wrong sample and the failing batch cannot be rebuilt from disk. **The capture in
`9fe7c703` exists to settle this from production instead** — its `repeated_12gram_count` is the
discriminator: 0-2 means over-generation, dozens means a loop.

### #2011 — speaker detection 400s

Fix written, uncommitted, 11 tests. Cause is per-episode, not per-show: Latent Space publishes
**episode descriptions up to 137,398 chars** (median 41,590) into a call that sends no transcript
and asks for 300 output tokens. Bound at 8,000 chars — ~2x the largest description on any feed that
has never overflowed, so nothing working changes.

### #2050 — context window is a MODEL property, not a global constant

`LLM_NARROWEST_CONTEXT_TOKENS = 32_768` is global; the 127-minute episode ceiling derives from it;
episodes above it are skipped entirely. The registry serves models with 128k-1M windows, all
clipped to the narrowest one.

**The design was revised after review** — the original proposal fixed one of six ceilings. On prod
(`bundled` quote mode) that constant governs only the episode gate. The stages that actually
overflow each carry their own literal:

| site | its limit |
| --- | --- |
| `summarize()` | **none at all** |
| GI extraction | 120,000 chars |
| KG extraction | 120,000 chars |
| bundled quotes | 50,000 chars |
| speaker description | 8,000 (#2011) |
| megabundle | 25,000 chars |
| declared capability | `max_context_tokens = 128000`, hardcoded on the OpenAI base class, **wrong for vLLM**, budgeted from by nothing |

Revised shape: unify the provider's context slot (`max_context_tokens` declared +
`_context_limits` learned), add ONE `transcript_budget_chars()`, route all six sites through it,
discover `max_model_len` in `_verify_served_model` (which already GETs `/v1/models`), declare the
served value on the vLLM `StageOption` (already a provider/model/**endpoint** triple, i.e. a
deployment), and only then raise the flag.

**Root cause lives in another repo:** `agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml`
sets `--max-model-len=32768` with the comment *"autoresearch summary inputs cap at ~30k tokens so we
don't need the full 128k window"* — an eval-harness assumption that became this pipeline's
episode-length policy when it started sharing that endpoint.

### #2051 — registry `extra_settings` silently dropped on every stage except GI

16 orphaned keys across 6 stages. **GI has zero** — it is the only stage with the
raise-on-unmapped-key guard (`model_registry.py:2964-2975`). Every stage without it leaks.

Impact measured: **6 of 10 have zero effect** (registry value == running value, kept in sync by
hand), 2 belong to unselected options, and **exactly one matters on the active prod path**:
`vendor_sampling` on `vllm_qwen3_30b_a3b_nvfp4`.

**The trap:** do not bulk-apply it. Prod sends `temperature=0.0` deliberately; `vendor_sampling`
carries `temperature: 0.7`. Plumbing it wholesale makes summarization non-reproducible — the exact
failure `gi_insight_temperature` exists to prevent. Take `presence_penalty`, decide
temperature/top_p/top_k deliberately, record the decision.

### #1985 — raise the DGX vLLM to 64k

Memory is not the constraint, measured on the box:

```
max_position_embeddings: 262144   rope_scaling: null      <- model does 256k natively
GPU KV cache size: 199,904 tokens   concurrency 6.10x @ 32k
48 layers x 4 KV heads x 128 dim x 2 x 1 byte (fp8) = 48 KB/token
--gpu-memory-utilization=0.25 on 121.7 GiB unified, 75 GiB free
```

64k costs 3 GiB/request -> ~3 concurrent, no memory change needed. Every observed overflowing
prompt is under 43,373 tokens, so **64k swallows all 565**.

**But raising the flag alone changes nothing** while `LLM_NARROWEST_CONTEXT_TOKENS` still clips at
32k. #2050 is its prerequisite, and making the window a discovered value is what removes the trap.

### #1984 — chunk / map-reduce for long episodes

May become unnecessary at 64k. Do not start it before #2050 and #1985 are answered.

## Ordering, and why

1. **Deploy what is on `main`.** Nothing else is measurable until then — every baseline above is
   pre-fix, and the capture instrumentation only produces evidence in production.
2. **#2051 `presence_penalty`** — cheapest possible test of #1893's leading hypothesis, because the
   value is already researched and already in the registry. Wire-level before/after, plus a
   determinism guard.
3. **#2050 provider unification** — `transcript_budget_chars()` routing all six sites. This is the
   real refactor; everything else in the context family is downstream of it.
4. **#1985 raise the flag** — last, once the config side can see it.
5. **#1893 proper fix** — decided by capture evidence, not by reasoning. May be closed by (2).
6. **#1984** — only if still needed.

## What must NOT be assumed

* That the chunk 10 -> 8 change fixed anything. It did not; the `insights=2` case at `1280/1280`
  disproves it.
* That the 21 damaged episodes lost content. The per-insight staged path is the designed fallback
  and it works — the reproduction produced 8 insights and 20 quotes despite the runaway. Repair
  them and measure the before/after insight count; the answer may be "nothing was lost".
* That local reproduction represents production. Four of five attempts produced healthy replies.
* That `temperature=0` makes this deterministic. vLLM continuous batching means identical inputs
  give different results; the same GI build reproduced a failure once in seven runs.
