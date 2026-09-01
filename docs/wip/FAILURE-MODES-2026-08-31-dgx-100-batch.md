# Failure modes observed during the 100-episode DGX batch (2026-08-31)

Living document. Updated as the batch runs; census re-run periodically so counts move.
Fixes are built locally against these findings and **not pushed** — this is the material
for the next push, to be validated against the completed run.

**Census scope at time of writing:** 31 of 100 episodes, 4 of 10 feed jobs complete,
`sha-a624e77`, profile `prod_dgx_full`, Qwen3-30B-A3B on the DGX.

The census counts things that did **not** fail the run. That is the central lesson of
2026-08-30/31: this pipeline's defects surface as waste and silent degradation, not as
errors. A census that only counted `ERROR` would have reported a clean night while the
model was emitting 14x the requested output and every spend figure was inflated.

---

## Census (31 episodes)

```
     4  FALLBACK: primary->ollama          (3 extract_quotes_bundled, 1 summarize)
    21  TRUNCATION: finish_reason=length
    48  OVERGEN: model ignored count       median 1.5x  max 3.6x   >=5x: 0
    13  JSON: bundled parse failed
     4  JSON: malformed mid-document
    12  JSON: document ended early
    38  SALVAGE: truncated lines recovered
     4  VALUE GATE: self-grading
    30  VALUE GATE: unsupported            3054 graded, 638 unsupported (21%)
    31  SPEAKER: uncorroborated rejected
    13  SPEAKER: proposed unknown name
    22  ERROR lines
 12608  WARNING lines
```

---

## A. Fixed 2026-08-31 — validate against the finished run

| # | Fix | Evidence it is working |
|---|---|---|
| A1 | Quote-budget overflow no longer fails over to a weaker model | 13 JSON parse failures produced only **3** fallbacks. Under the old code every one would have gone to `llama3.1`. |
| A2 | `GI_INSIGHT_TOKENS_EACH` 150 -> 50 | Over-generation ratio **max 3.6x, zero >=5x**. Previous night: 24 events >=5x, worst **14x**. Capping generation earlier bounds the runaway. |
| A3 | GI cost billed once per call | Zero duplicate `(in,out)` token pairs in the verification run. |
| A4 | ffprobe duration fallback | Zero diarization fuse trips across 31 episodes (was the #1880 regression). |
| A5 | `skip_existing` honoured | `--no-skip-existing` present in argv; offset batch selected new episodes. |

**Validation to run at batch end:** re-run the census on all 10 jobs and confirm A1/A2
ratios hold at 100 episodes, not just 31.

---

## B. Known-open — cost bounded, behaviour unchanged

### B1. The model still ignores the insight count (#1891)
48 over-generation events across 31 episodes (~1.5/episode). Ratio is down (max 3.6x vs
14x) **only because the token budget truncates it sooner** — the model has not changed its
behaviour. Real fix is #1891: guided decoding, a stricter prompt, or bounded chunks.

### B2. Qwen3-30B JSON reliability is now the visible failure mode
All 3 quote fallbacks are malformed JSON with `finish_reason=stop` — not budget:

```
invalid JSON: Expecting ':' delimiter
invalid JSON: Expecting ',' delimiter
invalid JSON: Extra data: line 3 column 1
```

The token ceiling was masking this. Now that budget overflow is handled correctly, the
model's structural-output reliability is what remains. Candidate fix: vLLM **guided JSON
decoding** via `vllm_extra_body` — the config plumbing already exists
(`{ns}_extra_body` is merged into every call). Blocked on the fact that GI returns a
newline list, not JSON, so the output contract would have to change first.

### B3. Salvage is load-bearing
38 salvage recoveries across 31 episodes. `insight_salvage.salvage_truncated_lines` is
routinely rescuing truncated responses. It works — but a path that fires ~1.2x/episode is
carrying more weight than a fallback should. Worth an explicit metric rather than a log line.

---

## C. NEW findings from this batch — not previously known

### C1. "Summarization OVERRAN its 1200s deadline" is mislabelled — it is GI
**22 ERROR lines, all of this shape.** But measured `summary_sec` maxes at **634.7s**,
comfortably inside the deadline.

The deadline wraps `call_generate_metadata` — summarisation **plus GI plus KG** — while
logging as "Summarization" (`workflow/stages/processing.py:2342`, `summarization_timeout`
default 1200). GI alone measured **1327s** on a single episode, so it blows a 1200s budget
attributed to a stage that took half that.

**Impact:** anyone debugging this goes looking at the summariser, which is innocent. It
also inflates `summarization_deadline_overruns` into a meaningless counter.

**Fix direction:** rename the deadline and its metric to reflect the real scope
(`metadata_generation_*`), or split it into per-stage deadlines so the attribution is
honest. Cheap and worth doing in the next push.

### C2. `corpus_metadata_index` duplicates are 90%+ of log volume — and the underlying
### divergence is a KNOWN issue, now quantified

Of 12,608 WARNING lines, the overwhelming majority are:

```
1120  corpus_metadata_index: duplicate guid substack:p...
1120  corpus_metadata_index: duplicate episode_id subs...
 100  corpus_metadata_index: duplicate guid 6a68f858...
```

**The duplicates are real.** Verified on disk — the same episode exists in several run dirs
because every reprocess writes a fresh `run_<ts>/` without removing the old one:

```
guid substack:post:178618026  "Netflix's Engineering Culture"
  run_1ebba1af-..._20260814-055303/metadata/0011 - ....json   <- KEPT
  run_20260822-222441/metadata/0011 - ....json                <- discarded
  run_20260825-054545/metadata/0011 - ....json                <- discarded
```

**CORRECTION — I initially wrote this up as a new discovery. It is not.** The behaviour is
already documented in `gi/integrity.py:17-19`:

> `_scan_corpus_metadata_index` is first-writer-wins (keeps the OLDER entry) while search's
> `merged_episode_gi_paths` takes the NEWEST, so duplicates mean two subsystems disagree
> about which artifact is canonical

So the defect is sharper than "the index is stale": **the index and the search path resolve
the same episode to different files.** An integrity gate already exists to catch it.

What this batch adds is **scale**. One guid warned **1120 times**; 12,608 lines total. The
divergence is not an edge case, it is the single most common log line in the system, and
the warning volume grows with (runs x episodes).

Also worth noting: ordering is effectively arbitrary. Older dirs are `run_<uuid>_<ts>` and
newer ones are `run_<ts>`, so under a lexicographic sort which one wins depends on the first
hex character of a UUID — not on recency.

**FIXED 2026-08-31** (locally, not pushed) — and the cause was two stacked defects, the
second only visible once the first was fixed and a test still failed.

**Defect 1 — the index bypassed the single source of truth.**
`discover_metadata_files` / `dedupe_metadata_paths_newest_run_per_episode` is documented as
the CENTRAL corpus-membership rule that indexing, digest, topic-clusters, enrichment,
catalog and staleness all share *"so they can never diverge (the 94-vs-106 split-brain)"*.
`corpus_metadata_index` was the one caller that did not use it — it ran its own
`sorted(root.glob(...))` and kept the OLDEST. Now routed through the shared dedupe.

**Defect 2 — the shared rule could not read production's run dirs at all.**
`_RUN_TS_RE` was anchored `^run_(\d{8}-\d{6})`, which matches `run_<ts>_<hash>` only.
Production uses `run_<uuid>_<ts>_<hash>` — **all 397 run dirs on the box**. So the
run-folder-timestamp path this function exists to prefer had **never once fired in
production**, and supersession ordering rested entirely on mtime — which the function's own
docstring calls out as unsafe for a real corpus (file-copy / backup-restore / rsync churn).
Regex relaxed to find the timestamp after an optional prefix; `run_append_<hash>` still
correctly falls through to mtime.

**Real-world consequence beyond log noise:** `corpus_rollback` resolves an episode's run dir
through `by_id`. With oldest-wins, an episode-scoped rollback deleted the SUPERSEDED copy
and left the newest — the copy search serves. The delete reported success while the episode
stayed. The pre-existing warning text hinted at exactly this ("callers that act on ONE entry
(rollback episode delete) may leave a copy behind").

Guarded by 5 tests in
`tests/unit/podcast_scraper/workflow/test_corpus_index_agrees_with_search.py`, including the
case where the winner previously depended on the first hex character of a UUID. Full suite:
10,171 passed.

The 12k warnings disappear as a side effect — a reprocessed episode superseded by a newer
run is no longer a "duplicate" to complain about. Genuine same-run collisions still warn.

### C3. Feed jobs complete short of 10 episodes — NOT A DEFECT (investigated)
Completed jobs show **8, 8, 7, 7** of 10, so the batch lands near **75-80**, not 100.

**Cause: positional offset drifts on a live feed.** Investigated on job `e259ae92`:

```
Episodes to process: 10 of 298   (episode_offset=10 applied)
  [1], [2]  already on disk -> skipped by --skip-existing
  [3]..[10] processed
```

The two skipped episodes were ingested **2026-08-30 19:40 and 21:19** — last night's depth
run. `episode_offset=10` counts POSITIONS in the feed as it stands today, but positions
shift as feeds publish. Two new episodes appeared since last night, so "skip the newest 10"
landed two episodes shallower than intended and overlapped completed work.

`skip_existing` caught them correctly. **This is the safety net working, not failing.**

**Limitation worth naming (feature, not fix):** we asked for "10 new episodes" and expressed
it as "skip 10 positions". On a live feed those are different, and they diverge further the
longer between runs. The drift-immune form would be a selection mode meaning "the next N
episodes I have not ingested", resolved by episode_id/guid rather than position — the same
principle `corpus_metadata_index` applies elsewhere. That is a design decision, not a
mid-batch change.

**Practical:** expect each feed to fall short by roughly however many episodes it published
since the previous run.

### C4. Value gate rejects 21% of insights, and self-grades
```
3054 insights graded, 2416 grounded, 638 unsupported (21%)
```
Plus **4 SELF-GRADING warnings** — on `prod_dgx_full` the value-gate rater is the same
provider AND model as the summariser, so the model is grading its own output. The warning
(added 2026-08-30) notes this is lenient, ~10% of insights. A 21% rejection rate under a
lenient self-grader is worth a quality look — it is not obviously a defect, but it is not
obviously fine either.

---

## D. Observability defects found while measuring

### D1. `run_timing` assumes serial execution — my own bug
`model_stage_share_pct` reported **203%** on a multi-episode job. The pipeline runs
`processing=2`, so two episodes overlap and the per-stage sums exceed wall-clock.

The metric is correct for single-episode runs (98.3% share, the #1888 refutation) and
**meaningless for batch jobs**. Fix: normalise by observed concurrency, or emit per-episode
rather than per-run. Until then, batch aggregates from `run_timing` must not be quoted.

### D2. A monitor whose transport can die silently
The batch watcher polled prod over SSH and `continue`d on empty output. When the ssh-agent
identity dropped, it went quiet for **4.4 hours** while reporting a stale 10/100 — the run
was actually at 29/100 and healthy. Silence read as "nothing to report".

Rewritten to query VictoriaLogs over the tailnet (no SSH) and to log loudly on an empty
result. General rule: a monitor must fail noisily, and should not share a failure domain
with the thing it monitors.

---

## E. Non-findings — checked and healthy

- **DGX thermals/throttling:** 7h sustained, max 75C, avg 63.6C, `THROTTLE_REASONS` **0**.
- **Prod VPS:** no container starved. Heaviest PSI cpu-wait over 7h is `alloy` at 53s;
  pipeline containers do not register.
- **Prefix caching:** working, **72%** hit rate (10.07M/14.00M). Prompt size is not a lever.
- **Diarization:** zero fuse trips since the ffprobe fix.

---

## F. The headroom finding (feeds #1888)

```
KV cache pool     9.15 GiB = 199,904 tokens
in use            6.4%     =  12,753 tokens   == exactly ONE request
typical request   13,256 tokens               -> pool holds ~15 concurrently
num_requests_running 1.0   num_requests_waiting 0.0
```

GPU reads 90.9% "utilised" because `DCGM_FI_DEV_GPU_UTIL` measures time-with-kernel-resident,
not capacity. One decode stream on a 3B-active MoE keeps a kernel resident while using little
compute. **91% busy, 6% cache, 1 request** = occupied, not saturated.

The 32 `extract_quotes` calls per episode are mutually independent. Running them concurrently
is the untested hypothesis; expectation 4-7x aggregate throughput on a bandwidth-bound MoE,
**not measured**.

---

## G. Status roll-up (2026-08-31 23:50) — what is closed, what is not

Batch state: still running. Episodes reaching the last stage (`kg`), per feed job:
`8, 8, 8, 8, 7, 7, 7` complete + `3` in flight. Read as a **floor** — the events tool
returned an identically-sized payload for `limit=400` and `limit=5000`, so I cannot prove the
slice is the whole window. The 8/8/7/7 shape matches **C3**'s prediction; tracking to ~75-80.

### Closed, with a regression test and a mutation check

| Ref | What | Test |
|---|---|---|
| A1 | Quote-budget overflow does not fail over | `test_quote_bundle_budget_not_failed_over.py` |
| A2 | `GI_INSIGHT_TOKENS_EACH` 150 -> 50 | `test_gi_cost_and_overgeneration.py` |
| A3 | GI cost billed once | `test_gi_cost_emitted_once.py` |
| A4 | ffprobe duration fallback | `test_audio_duration_ffprobe_fallback.py` |
| A5 | `skip_existing` honoured | `test_skip_existing_negative_flag.py` |
| C1 | Deadline relabelled to metadata-generation scope | (log/metric rename) |
| C2 | Index and search agree; `_RUN_TS_RE` matches prod | `test_corpus_index_agrees_with_search.py`, `test_run_recency_epoch.py` |
| D1 | `run_timing` concurrency-normalised | `test_run_timing_concurrency.py` |
| D2 | Watcher fails loudly | (rewritten off SSH) |
| F3 | `gi-repair --episode-ids --force-healthy` | `test_gi_repair_targeting.py`, `test_gi_repair_cli_args.py` |

Every fix above was mutation-tested: the pre-fix code was re-introduced and the suite had to
go red. Three mutations initially came back **green** and the tests were rewritten — see
`test_pipeline_stage_prevalidation.py`'s `TestExplicitFlagBeatsTheFile` docstring for the one
that mattered (a test that mirrored the logic instead of exercising it).

### Found while writing those tests — fixed here, not in the original census

- **`_on_disk_guid_index` had a fallback worse than the failure it handled.** The newest-run
  dedupe import was wrapped in `except ImportError` + a WARNING. With the dedupe gone,
  first-wins over ascending-sorted globs resolves a reprocessed episode to its **oldest** run,
  every time — not "may resolve to a superseded run" as the warning claimed. That idx drives
  the `{idx} - *.txt` transcript glob, so `relabel_only` / `rediarize_only` would have
  re-derived a superseded transcript over the current one, behind a warning and a zero exit.
  The swallow is gone; the import raises. `test_on_disk_guid_index_determinism.py`.
- **`gi-repair` ids-file did not dedupe.** Every id is a paid re-derivation, so a duplicated
  line spends DGX time overwriting an artifact with a second sample of itself. Now
  order-preserving deduped, with the drop count logged rather than collapsed silently.
- **Broken anchor** in this guide's own sibling (`CORPUS_REPROCESSING.md` linked
  `#re-name--re-diarize-only`; the generated id is `#re-name-re-diarize-only`).

### NOT closed — carried forward

| Ref | What | Why it is still open |
|---|---|---|
| B1 | Model ignores the insight count (#1891) | Ratio is down only because truncation bites sooner. Behaviour unchanged. Needs guided decoding / stricter prompt / bounded chunks. |
| B2 | Qwen3-30B JSON reliability | Now the visible failure mode. Blocked: GI returns a newline list, not JSON, so guided JSON decoding needs the output contract changed first. |
| B3 | Salvage fires ~1.2x/episode | Works, but load-bearing. Wants an explicit metric, not a log line. |
| C3 | `episode_offset` drifts on live feeds | Design decision: a "next N not-yet-ingested" selection mode keyed on guid. Not a mid-batch change. |
| C4 | 21% value-gate rejection under a self-grading rater | Quality question, not a mechanical one. Needs a disjoint-vendor rater to even measure. |
| F | GPU-with-GPU concurrency | The 4-7x expectation is **unmeasured**. This is the actual #1888 lever. |

---

## H. The B-series (2026-09-01) — where the GI cost actually is

Stage cost, measured over 72 episodes of the running batch (`pipeline_stage` events):

| stage | median/ep | max | share of pipeline time |
|---|---|---|---|
| **gi** | **824.7s** | 3446.6s | **57.4%** |
| asr | 470.6s | 1167.7s | 24.8% |
| summary | 236.4s | 634.7s | 15.4% |
| kg | 36.4s | 77.5s | 2.4% |

LLM call volume over the same window (`llm_cost` events):

| stage | calls | calls/ep | input tokens | output tokens |
|---|---|---|---|---|
| extract_quotes | 1334 | 15.0 | **17,152,856** | 1,342,948 |
| cleaning | 355 | 5.0 | 904,740 | 705,180 |
| gi | 254 | 4.0 | 1,747,830 | 656,017 |
| score_entailment | 1521 | 20.0 | 1,591,230 | 197,401 |

`extract_quotes` is ~72% of all input tokens. **Insight count is its multiplier**: over 71
episodes insight_count correlates r=0.58 with quote CALLS, r=0.60 with quote INPUT tokens,
r=0.76 with entailment calls. Cutting insights cuts the dominant stage proportionally — that is
the value argument for everything below.

### H0. The size of the prize, measured

GI is 57.4% of pipeline time. Within it, measured over 73 episodes / 255 GI calls:

| | measured |
|---|---|
| output tokens per GI call | **1880** median (ceiling was 2500 — calls run at **75% of the ceiling**) |
| insights KEPT per call | 23 |
| token cost of the kept text | ~460–575 (at the 20–25 tok/insight the config records) |
| **discarded** | **~73% of GI generation** |

That is the whole item in one line: the model is handed a 2500-token budget, spends 1880 of it,
and we keep about a quarter of what comes back because `cleaned[:max_insights]` throws the rest
away. Two independent causes, and they need different fixes:

1. **We ask for the cap once per chunk** (H1) — fixed by dividing the episode budget. This also
   lowers `max_tokens`, so it is the one that reclaims GPU time.
2. **The model overshoots whatever we ask** (H2a) — a serving-side issue, unfixed, on #1896.

### H1. The ceiling was multiplied by the chunk count — FIXED

`generate_chunked` handed every chunk the full episode ceiling. That ceiling is already
duration-scaled, and the chunk count is *also* duration-derived, so duration was counted twice:

| transcript | chunks | cap/chunk | effective ceiling |
|---|---|---|---|
| 52k | 2 | 50 | 100 |
| 120k | 4 | 125 | 500 |
| 200k | 6 | 200 | 1200 |

Production ran at median **79.5** insights/episode against a configured 50, max 157.
`gi_max_insights: 50` meant 50 nowhere. Now the episode budget is divided across passes
(`per_chunk_budget`). Per **ADR-135:59-61** the ceiling is "extraction/token-budget safety only,
never a corpus cutoff", so the merged list is deliberately NOT trimmed — a test pins that, and
the "tidy" trim is a mutation the suite rejects.

**Validated end to end** — the real `generate_chunked`, a real model
(`qwen3-30b-a3b-instruct-2507` via OpenRouter so the running batch was untouched), the real 52k
transcript, temperature 0.0, 3 runs per policy:

| policy | per-chunk ask | insights | vs cap 50 |
|---|---|---|---|
| OLD (every chunk gets the episode cap) | 50 | 95, 98, 94 -> **median 95** | 1.9x |
| NEW (episode budget divided) | 25 | 48, 49, 49 -> **median 49** | **1.0x** |

Lands on the configured ceiling, tight variance. An earlier extrapolation of mine predicted ~25
for this case and was WRONG: it assumed the dedupe keep-ratio stays constant, but dedupe
pressure falls when each chunk produces less, so the naive model under-predicts badly. Measured,
not extrapolated. This is a 2-chunk episode; production runs 2-7 chunks, where the old
multiplication was proportionally worse.

**GPU time reclaimed.** Lowering the per-chunk ask also lowers `max_tokens`
(`max(GI_INSIGHT_TOKENS_FLOOR, ask * GI_INSIGHT_TOKENS_EACH)`):

| chunks | OLD budget/call | NEW budget/call |
|---|---|---|
| 2 | 2500 | 1250 |
| 3-6 | 2500 | 1024 (the floor) |

Applied to the batch's real per-episode chunk counts, the GI output ceiling falls
**637,500 -> 265,640 tokens (-58.3%)** against 619,593 actually emitted — i.e. the batch was
running near its ceiling, exactly as "runaway calls use the FULL budget every time" predicts.

### H2a. THE CONTROL: the same model over-generates on our DGX and not on OpenRouter

`qwen/qwen3-30b-a3b-instruct-2507`, identical prompt, identical transcript, cap=50:

| host | median | ratio | truncated |
|---|---|---|---|
| OpenRouter | 54.0 | **1.08x** | **0/4** |
| our DGX | 54–126 | **1.08–2.52x** | yes, repeatedly |

Same model. Same prompt. So the over-generation is **our serving configuration**, not the model
and not the prompt. Two visible differences to chase on #1896:

* the DGX serves `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` — a **4-bit NVFP4 quantisation**;
  OpenRouter serves the standard weights. Degraded instruction-following (specifically
  "stop at N") is a known quantisation failure mode.
* vLLM `0.20.1+7124b12a.dev`, `max_model_len` 32768, and no `vllm_extra_body` sampling
  overrides in `prod_dgx_full`.

Also worth noting for any follow-up: production runs `gi_insight_temperature: 0.0` while these
A/B probes used 0.3. The comparison is internally valid (both arms at 0.3) but the absolute
ratios are not production's.

### H2b. What the chunk fix actually buys — count, not tokens

Worth being precise, because the two are easy to conflate. The provider already trims with
`cleaned[:max_insights]`, so the model's overshoot never inflates the STORED count — it is
bounded per call. Therefore:

* **Count inflation** (median 79.5 vs configured 50) came from the chunk multiplication, and is
  what H1 fixes: 2 chunks x 50 kept = 100 pre-dedupe -> ~79.5. With the budget divided, 2 x 25
  = 50 pre-dedupe -> roughly the configured number. This lands regardless of the overshoot.
* **Token waste** (generating ~120 to keep 25) is the DGX serving issue in H2a, and H1 does
  nothing for it. That is the remaining lever, and it is on #1896.

### H2. Over-generation is Qwen-specific, not a universal prompt defect

Same prompt, same transcript, cap=50, reasoning disabled:

| model | median insights | ratio |
|---|---|---|
| Qwen3-30B-A3B-Instruct (DGX) | 54–126 | 1.08–2.52x |
| podcast-flash-0731 (DeepSeek v4, prod cloud) | 38 | 0.76x |
| homelab-kimi | 28 | 0.56x |

DeepSeek and Kimi land comfortably under the cap. Only Qwen overshoots. So the prompt
contradiction was real but is not what makes Qwen overshoot.

### H3. MEASUREMENT CONFOUND — the DGX A/B is not trustworthy as run

The same CURRENT prompt, same five seeds, same transcript, same model, measured **twice**:

    run A: 130, 123, 123,  86, 139   median 123  (2.46x)
    run B:  67, 126,  54,  68,  61   median  67  (1.34x)

vLLM's `seed` does not give determinism under continuous batching while production traffic
shares the engine. The 1.8x swing in the BASELINE is larger than the effect I attributed to the
prompt earlier in the session — so **the "2.46x → 1.60x" figure I reported was load drift, not
a measured prompt effect.** Retracted. A paired design (both arms issued concurrently so they
see the same batch state) is the only valid way to measure this here.

### H4. Reasoning models and the token budget — no live exposure, latent trap

Reasoning tokens bill against `max_tokens` and never appear in `content`. Production's alias
on a real transcript through the gateway:

| extra_body | budget | reasoning_tokens | insights | finish |
|---|---|---|---|---|
| profile (reasoning off) | 2500 | 0 | 35 | stop |
| profile (reasoning off) | 7500 | 0 | 41 | stop |
| omitted | 2500 | 2500 | **0** | length |
| omitted | 7500 | 2765 | 17 | stop |

I briefly reported this as "production is broken by the 150→50 cut". **That was wrong** — my
probe had omitted the profile's `litellm_extra_body`. All five LiteLLM profiles routing a
reasoning alias set `reasoning: {enabled: false}`, and production produces 35 insights at the
current budget. The real gap is narrower: `LiteLLMProvider` has no headroom FALLBACK, unlike
`deepseek_provider` (`_REASONING_TOKEN_HEADROOM = 2048`) and `groq_provider`. Closed today with
a profile guard; the provider-side fallback is on #1896.

### H5. Sixteen GI counters were never written anywhere — FIXED

`Metrics` is a dataclass but `finish()` builds an explicit dict literal, so a counter must be
BOTH a declared field AND named in that literal. Sixteen were neither — seven added earlier in
this session, nine pre-existing, including **every value-gate counter**. That is why C4's 21%
rejection rate had to be reconstructed by grepping WARNING lines. All declared + exported, with
`test_bumped_metrics_are_exported.py` scanning bump call sites via AST so the next one cannot
repeat it.

---

## I. `enrich_only` — the reprocess mode was a silent no-op (2026-09-01)

This is the mode the operator asked for in plain words: *"when we reprocess an episode, skip
download and ASR and diarization, and just do the LLM part."* `enrich_only` is this codebase's
existing name for exactly that. The name is a poor one and is being changed (see I3).

### I1. Why it did nothing

`enrich_only` coerces `transcribe_missing=false` — correct, it must never call an ASR provider.
But the only other exit from `process_episode_download` was the
`if cfg.transcribe_missing and temp_dir:` gate, so the function returned `(False, None, None, 0)`,
the caller queued no `ProcessingJob` (it requires a non-None `transcript_source`), and the run
exited **0**. Documented as broken in the reprocessing guide and the Makefile rather than fixed.

Its two siblings dodge this by setting `transcribe_missing=true` **specifically to reach** the
transcription stage and intercept there — the config comments say so outright. That route is
closed to enrich_only: `transcribe_missing=true` is also what makes the Deepgram validator
demand an ASR key for a stage that calls no ASR. So the fix resolves the transcript in
`process_episode_download` instead, with no audio, no temp dir, and no ASR credential.

### I2. A SECOND no-op was hiding behind the first

`enrich_only` alone selects nothing. It coerces `skip_existing=true`, so without
`--reprocess-existing-only` the work list is built from the LIVE feed and every on-disk episode
is then skipped. Measured: `Episodes to process: 0 of 0`, exit 0. The Makefile target had this
second defect even after the first was fixed.

**This is why unit tests were not enough.** Three end-to-end attempts were needed:

| attempt | outcome | cause |
|---|---|---|
| 1 | branch never reached, `0 of 0` | no `--reprocess-existing-only`; invented RSS URL |
| 2 | `ValueError: no on-disk episode GUIDs found` | feed dir is a HASH of the RSS URL; no URL tried hashed to the corpus's |
| 3 | **works** | fixture slug aligned to the real URL |

Attempt 2 failing LOUDLY is `_on_disk_guid_index` behaving correctly — the opposite of the bug.

**Verified working** (real CLI, real corpus, homelab gateway):

    [1] enrich_only: re-deriving from existing transcript .../run_20260830-144405/transcripts/...
    result: episodes=1 ok=1 failed=0 skipped=0
    transcribe_sec_total=0.0   download_media_sec_total=0.0   gi_sec_total=165.0

| | insights | content sha |
|---|---|---|
| original run dir | 64 | `523b50359e37` (untouched) |
| new run dir | 33 | `6816bcaa9a21` |

The same run also confirmed **H1** live — `chunked extraction: episode ceiling 50 over 2 passes
-> 25 insights per pass` (it would have been 50/pass before) — and exercised the value gate
(`dropped 8/41 insights below tier 2`), whose counters were among the sixteen writing nowhere.

### I3. Two follow-ups the operator called

1. **Write in place, not a new run dir.** Every other reprocess path writes in place
   (`relabel_only` "in place", `rediarize_only` "in place", `gi-repair` in place and explicitly
   "no new run dir, no index split-brain"). enrich_only spawning `run_<ts>/` is the outlier, and
   C2 above measured that pattern's cost: 12,608 duplicate-guid warnings. The change lands in
   ONE function — `_determine_metadata_path` — because GI/KG/bridge/context all derive their
   paths from it, and `run_index.episode_metadata_rel_in_corpus` already resolves the existing
   location. NOT YET CONFIRMED: whether relabel/rediarize write their ARTIFACTS in place or only
   their transcript; a relabel run is settling that before the "every other path" claim stands.
2. **Rename the stage.** `enrich_only` collides with `make enrich` (corpus-level topic clusters
   / co-appearance), a different operation entirely. Candidate `rederive_only`, matching the
   `<verb>_only` convention of its siblings, with `enrich_only` kept as a deprecated alias
   because profiles, the Makefile and docs reference it.

---

## J. Batch complete — the closing numbers (2026-09-01)

The 100-episode DGX batch finished at **83 episodes** past `kg` across 18 runs. Not 100: see
**C3** — `episode_offset` is positional and the feeds published between runs, so each job landed
short. That is the defect `episode_selection=unprocessed` now fixes.

Final stage profile, unchanged in shape from the mid-run census:

| stage | median/ep | share |
|---|---|---|
| **gi** | **820.3s** | **56.4%** |
| asr | 487.7s | 26.0% |
| summary | 231.1s | 15.1% |
| kg | 39.4s | 2.5% |

**insights/episode: median 80, min 22, max 157 — against a configured cap of 50.** This is the
whole batch confirming H1: the per-chunk multiplication, not model overshoot, is what put the
stored count at 1.6x the configured ceiling. The fix is measured to bring it to 1.0x (95 -> 49
on a real model), but note that number comes from a controlled single-episode run, NOT from
this batch — no episode here ran with the fix.

### The three GI levers, separated

Worth stating plainly because they were conflated for most of this investigation:

1. **Count inflation** (median 80 vs 50) — chunk multiplication. FIXED (H1).
2. **Token waste** (~73% of GI output discarded) — the model overshoots whatever it is asked
   for. NOT fixed; it is serving-side, and the control experiment (H2a) shows the same model
   on OpenRouter does not do it. On #1896.
3. **Call count** — unchanged by anything in this change set. The chunk fix makes calls
   SHORTER (output ceiling -58%), not fewer. The only call-count reduction is B2's, which
   removes the bisect retries that truncation was causing.

### NOT verified — claims I have not tested

- **A1-A5 at 100 episodes.** Validated on the 31-episode census only. The end-of-batch
  re-census has not run.
- **C1's rename** has no test. It is a log string and a metric name; I asserted neither.
- **D2's watcher** was rewritten but its failure path was never exercised — I did not kill the
  transport and confirm it screams.
- **Every fix in this file is unpushed and unexercised in production.** The running batch is
  `sha-a624e77`, which contains none of them.
- The **B2 JSON-reliability rate** is a count from one 31-episode census, not a rate I have
  re-measured since the token-budget fix changed what reaches the parser.
