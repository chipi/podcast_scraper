# Arc: LLM budgets, context limits, and registry plumbing

**Opened:** 2026-09-13. **Status:** planning. 6 commits on `main`, **none deployed**; 6 issues
open; 1 fix uncommitted. No measurement in this arc is possible until the deploy.

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
| `08ce727a` | duration measured from the media file when RSS omits it | **partial — see below** |
| `9fe7c703` | capture the failing reply on a bundled-quote parse failure | evidence for #2053 |
| `303971d4` | work-list of 21 quote-damaged episodes | repair after deploy |

**Uncommitted:** speaker-detection description clip (#2011), 11 tests passing. Nothing else — the
retry-gating / tail-degeneracy / capture-`in_tok` edits written on 2026-09-13 were reverted; this is
a planning arc, not an implementation one. The reverted patch is kept at
`scratchpad/halted-advisor-fixes.patch` for whoever picks up the runaway.

**`08ce727a` does NOT close the `duration_seconds=0` hole, and its commit message says it does.**
Verified 2026-09-13:

* `_duration_from_transcript()` is **dead code** — it reads `job.transcript_segments`, which is not
  a field on `TranscriptionJob` (`models/entities.py`). Only the ffprobe path can ever fire.
* The artifact's `duration_seconds` does not come from that function at all. It comes from
  `extract_episode_metadata(episode.item, feed.base_url)` at `workflow/stages/metadata.py:194` —
  **RSS only**. So Ottoman History still writes `duration_seconds=0` today.

What the commit actually fixed is the pipeline's internal `audio_sec` (throughput accounting), not
the published artifact. Closing the artifact hole is a separate, unstarted change: route
`episode_duration_seconds` through the measured value when the RSS tag is absent.

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

### #2053 — the bundled-quote runaway (the `length` failures, 73%)

**Scope correction, 2026-09-13.** This arc — and an earlier draft of this file — treated #1893 as
the issue for the bundled-quote runaway. **It is not, and never was.**

Read #1893's title and all nine of its comments: every one is
`BadRequestError: 400 ... maximum context length is 32768`. That is a **context overflow** — the
server refusing a request before generating anything. The failure described in this section is the
opposite shape: a **successful** call (HTTP 200) whose reply ran to `max_tokens` and came back as
unparsable JSON. No 400, no `BadRequestError`, a different GlitchTip fingerprint, a different
mechanism, and a different fix.

Consequences of the conflation:

* **#1893 is really the same population as #2050** — the 565 overflow rejections. Its fix is
  #2050 (budget derived from the served window) plus #1985 (a window big enough). It should not be
  waiting on capture evidence, and it will not be closed by #2051.
* **The 215 runaways — 73% of 293 bundled-quote failures, the largest single failure class in the
  arc — had no GitHub issue at all.** Filed as **#2053** on 2026-09-13. #2001 is the nearest
  neighbour and is not it: that is the *ollama* tier returning malformed JSON (the `stop` half), not
  the vLLM tier running to the ceiling.
* The capture in `9fe7c703` and the "decided by evidence, not reasoning" plan belong to **this**
  item — now #2053 — not to #1893.

The rest of this section describes #2053.

Every truncating call consumes its **entire** budget whatever the budget is: `5120/5120` at ten
insights, and `1280/1280` at a bisected batch of **two** already running at 640 tokens/insight. One
broke at char 15 of a 3,834-char document — a single unterminated string running to the ceiling.

**Batch size is therefore NOT the lever**, and the chunk 10 -> 8 change must not be sold as the fix.

The `2048` in #1893's original error is a second reason the two are distinct: it is
`GI_QUOTE_RESPONSE_TOKENS`, consumed at `openai_provider.py:2629` inside `extract_quotes()`
(`:2584`) — the **staged per-insight** path. The runaway happens in `extract_quotes_bundled()`
(`:2778`), whose budget is computed per batch and is never 2,048. The two paths share a name and
nothing else.

Two candidate causes needing opposite treatments:

* **decoding loop** -> a penalty. Note `presence_penalty: 1.5` is ALREADY in the registry for this
  model and never reaches the wire — see #2051. That is the cheapest thing to try.

  **A local A/B of this was run on 2026-09-13 and is INCONCLUSIVE, not supporting:**

  ```text
  CONTROL                 n=12  finish=length: 0/12  tokens med=930  max=1629
  presence_penalty=1.0    n=12  finish=length: 0/12  tokens med=1316 max=1617
  repetition_penalty=1.05 n=12  finish=length: 0/12  tokens med=1092 max=1769
  ```

  Zero truncations in the **control** arm, so the experiment never reproduced the failure it was
  meant to fix and the three arms are indistinguishable. Note also the penalty arm's median output
  ran ~40% LONGER than control. Do not cite this as evidence the penalty helps; it is evidence that
  the local harness cannot reproduce the runaway.

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

**The trap:** do not bulk-apply it. `vendor_sampling` carries `temperature: 0.7`, `top_p: 0.8`,
`top_k: 20`, `presence_penalty: 1.5` — four knobs, only one of which is wanted.

**Correction to an earlier reading of this file:** prod does *not* summarize at `temperature=0.0`.
`prod_dgx_full.yaml` sets `gi_insight_temperature: 0.0` and nothing else; it has no `vllm_temperature`
key, so `summary_temperature` falls back to its default of **0.3**
(`openai_provider.py:574`) and `speaker_temperature` likewise (`:565`). Summarization is therefore
already non-deterministic today. That does not make plumbing 0.7 harmless — it makes the true
change 0.3 -> 0.7, which is still a content change on every episode and still needs the gate in
"Validation and gating" below. It does mean the argument "this would break our determinism" is
wrong, and the honest statement of the risk is "this widens an existing spread".

### #1985 — raise the DGX vLLM to 64k

Memory is not the constraint, measured on the box:

```text
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

**Marko, 2026-09-13: "Deploy comes after we fix all of this."** One deploy at the end, not a deploy
first. An earlier draft of this file had deploy at step 1 on the grounds that nothing is measurable
until then. That reasoning was wrong for two reasons, both established below: the REF baseline
already exists on disk, and the validation venue is the DGX, not prod.

### The fact that makes deploy-last work

`prod_dgx_full.yaml:120` and `dev_dgx_full.yaml:135` set the **same** `vllm_api_base` —
`http://${DGX_TAILNET_HOST:-dgx-llm-1}:8003/v1` — and the same
`NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`. Dev and prod are one server and one model. So validating on
dev exercises prod's real serving path: the only differences are the code version and the corpus,
never the model, the tokenizer, or the context limit. Every code item below can be fully validated
before the deploy without touching prod.

### The order

0. **Freeze the REF first.** The current prod corpus **is** the baseline — those `.gi.json` artifacts
   were written by `sha-1ac4902` and are the only record of pre-fix behaviour. Export the chosen
   episode-ID set now. This costs minutes and is unrecoverable once anything overwrites it; the 21
   episodes in the repair worklist especially, since `--force-healthy` overwrites in place.
1. **#2051 `presence_penalty`** — cheapest test of the **runaway's** leading hypothesis; the value is already
   researched and already in the registry. Wire-level before/after on the DGX.
2. **#2050 provider unification** — `transcript_budget_chars()` routing all six sites. The real
   refactor; everything else in the context family is downstream of it.
3. **#2011** speaker clip — independent of the context family, already written, gate it and fold it
   into the same deploy.
4. **#1985 raise the DGX flag** — see the warning below; this one is **not** part of the deploy.
5. **Deploy once**, with all of the above.
6. **#2053** (the runaway) and **#1984** — second cycle, per the next section.

   **#1893 and #2011 are NOT second-cycle** — they are context-overflow 400s, fixed by #2050 +
   #1985 and by the clip respectively, so both close with this deploy.

### #1985 is a different lever with a different blast radius

Raising `--max-model-len` to 64k is a container restart in
`agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml`. It is **not** a podcast_scraper
deploy and does not wait for one. Three consequences:

* It takes effect on **prod** the moment the container restarts, because prod points at that server.
  There is no staging step and no code gate in front of it.
* It **cuts serving concurrency** — measured 6.10x at 32k, ~3 concurrent at 64k (3 GiB/request).
  Whatever ingestion is running gets slower. Restart it when the pipeline is idle, not mid-batch.
* Raising it alone still changes nothing useful while `LLM_NARROWEST_CONTEXT_TOKENS` clips at 32k.
  **#2050 is its prerequisite**, so the flag raise is worth doing only once the deployed code can
  discover and use the larger window.

### What deploy-last costs, stated plainly

**The bundled-quote runaway cannot be diagnosed before the deploy, and no ordering fixes that.**
Its plan is "decided by
capture evidence, not by reasoning" — and the capture (`9fe7c703`) only produces evidence in
production, because local reproduction fired once in seven full GI builds and never in 36 isolated
calls (and the 2026-09-13 penalty A/B above produced **zero** truncations in its control arm).
So it ships as *instrumentation*, not as a fix, and is closed in a second cycle once real captures
exist. #2051 may close it for free; if it does not, the evidence arrives after this deploy, not
before it. Treating it as blocking this deploy would block the deploy indefinitely.

This costs nothing that deploy-first would have bought: the capture only starts producing data
at the deploy either way.

## Validation and gating

Five of the seven items change **what gets written into an artifact**. They are not refactors and
must not be gated like refactors — a green unit suite says nothing about whether an insight got
worse. This section says, per item, what the gate is and what it is measuring.

### The tooling that exists (do not build new)

| tool | what it gives you |
| --- | --- |
| `make gil-quality-metrics DIR=<run> ARGS='--enforce --strict-schema'` | **hard pass/fail** on PRD-017 thresholds |
| `make compare-gil-runs REF=<run_a> CAND=<run_b>` | per-episode A/B between two run directories |
| `make corpus-gi-integrity-check CORPUS_DIR=...` | structural integrity of the written corpus |
| `make corpus-completeness-check` / `corpus-placeholder-check` / `corpus-summary-audit` | missing / placeholder / degenerate summaries |
| `make pipeline-validate PROVIDER=openai MODEL=<vllm model>` | summary -> GI -> KG -> bridge end to end |
| VictoriaLogs, window-scoped | the 400 count (**not** `/api/jobs/{id}/log` — see the measurement traps) |

The `--enforce` thresholds, from `gi/quality_metrics.py:146`:

```text
extraction_coverage    >= 0.80   (artifacts with >=1 insight AND >=1 quote)
grounded_insight_rate  >= 0.90
quote_validity_rate    >= 0.95
avg_insights_per_artifact  >= 5.0
avg_quotes_per_artifact    >= 10.0
```

### Two rules that apply to every item below

1. **The REF already exists — preserve it, do not regenerate it.** Every gate here is an A/B, and
   the "before" side is the prod corpus as written by `sha-1ac4902`. No special pre-deploy run is
   needed; what *is* needed is a snapshot of the chosen episode-ID set before anything overwrites
   it. `gi-repair --force-healthy` overwrites in place, so the 21 worklist episodes are the urgent
   ones. Do this at step 0 of the ordering above.
2. **`avg_insights_per_artifact` moving is ambiguous by construction.** `gi_require_grounding` drops
   insights that lost their quotes, so a rise can mean *better extraction* or *the same extraction
   with fewer drops* — opposite conclusions. Only `compare-gil-runs` over the **same episode IDs**
   separates them. A threshold pass alone is necessary and not sufficient.

### Per item

| item | changes artifacts? | gate |
| --- | --- | --- |
| **the deploy** (all items at once) | **yes** — `320f2db0` changes quote decoding | the deploy is the *last* step, so it carries every change below at once. Gate each item on the DGX first; then post-deploy, REF (prod corpus at `sha-1ac4902`) vs CAND on the fixed set: `--enforce` must pass and `compare-gil-runs` must show no episode losing insights or quotes |
| `9fe7c703` capture | no — writes diagnostic files only | disk-growth check on the capture dir; it is the one thing here that can ship on tests alone |
| **#2011** speaker clip | **yes** — changes speaker-detection input | roster diff on Latent Space: today those episodes have **zero** speakers, so the only acceptable outcome is more names than before, never fewer. Also assert no feed under 8,000 chars changed at all (that is what the 11 unit tests pin) |
| **#2051** `presence_penalty` | **yes** — changes sampling on every summary | wire-level capture proving the field reaches vLLM (the bug is that it silently does not); then REF/CAND `--enforce` + `compare-gil-runs`. **Ship `presence_penalty` alone.** Decide `temperature` 0.3 -> 0.7 as a separate change with its own A/B, and record the decision |
| **#2050** `transcript_budget_chars()` | **yes** — changes how much transcript each of six sites sees | this is the highest-risk item: one function now governs six call sites that today carry six different literals. Gate each site separately — a change that is right for GI extraction at 120k may be wrong for bundled quotes at 50k. Per-site REF/CAND, plus the 400 count in VictoriaLogs must not rise |
| **#1985** raise vLLM to 64k | **yes** — episodes previously *skipped* now get processed | two populations, gated differently: (a) previously-skipped episodes — measure they now produce artifacts that pass `--enforce`; (b) previously-working episodes — must be **byte-identical in shape**, because nothing about them should change. If (b) moves, the budget routing in #2050 is wrong |
| **#2053** the runaway | **yes** | second cycle. Decided by capture evidence, not reasoning. Gate: the `finish_reason=length` rate over a full job, against the measured 215/1,086 baseline |
| **#1893** overflow 400s | **yes** — via #2050/#1985 | the VictoriaLogs 400 count over a comparable window must go to ~0, against the measured 565/30d and ~7/day |
| **#1984** chunking | **probably never** | blocked on #2050 + #1985 by its own acceptance. After those land, re-measure the ~127-min skip; a zero residue closes it. Do not design a gate for it before that number exists |

### The repair worklist is the natural A/B population

`docs/wip/quote-repair-worklist-2026-09-11.txt` holds 21 episodes with known quote damage. They are
the best CAND set in the corpus: the failure already happened on them, so a fix that works is
visible immediately, and re-running them costs nothing new. Run `gi-repair --episode-ids
--force-healthy` on that list after the deploy and compare insight/quote counts per episode against
what is on disk today. **Snapshot their current artifacts first** — `--force-healthy` overwrites.

### What is NOT gated, and is a known hole

* **No gate catches a summary getting blander.** Every threshold above is structural — counts,
  coverage, validity. `quote_validity_rate` checks a quote is *findable in the transcript*, not that
  it is *the right quote*. A sampling change that keeps all counts and degrades judgement passes
  every gate in this document. `make silver-pairwise` (LLM judge, A/B) is the only tool that would
  see it, and it is not wired into any of the above. For #2051's temperature question specifically,
  that is the gap that matters.
* **CI runs none of this.** The eval gate and corpus audits are not in the CI pipeline; a green PR
  proves lint, types and unit tests only. Every gate here is something a person runs deliberately.
* **Determinism is not assertable on this stack.** vLLM continuous batching gives different results
  for identical inputs, so "same input, same output" cannot be a gate for any item. The same GI
  build reproduced a runaway once in seven runs. Gates must be statistical over a set, never
  a single-episode diff.

## What must NOT be assumed

* That the chunk 10 -> 8 change fixed anything. It did not; the `insights=2` case at `1280/1280`
  disproves it.
* That the 21 damaged episodes lost content. The per-insight staged path is the designed fallback
  and it works — the reproduction produced 8 insights and 20 quotes despite the runaway. Repair
  them and measure the before/after insight count; the answer may be "nothing was lost".
* That local reproduction represents production. Four of five attempts produced healthy replies.
* That `temperature=0` makes this deterministic. vLLM continuous batching means identical inputs
  give different results; the same GI build reproduced a failure once in seven runs.
