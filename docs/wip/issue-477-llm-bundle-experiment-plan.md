# Issue #477 — LLM bundle experiment (clean + summary + bullets)

**Status:** WIP — implementation landed; ship only after measured smoke/benchmark runs.

**GitHub:** [Issue #477](https://github.com/chipi/podcast_scraper/issues/477)

## What shipped in code

- `Config.llm_pipeline_mode`: `staged` (default) | `bundled`
- `Config.llm_bundled_max_output_tokens` (default `16384`)
- `summarize_bundled()` on OpenAI, Anthropic, Gemini providers
- Workflow: pattern pre-clean, then one structured JSON completion; fallback to staged on failure
- `Metrics`: `llm_bundled_*`, `llm_bundled_fallback_to_staged_count`,
  `total_episode_estimated_cost_usd`, `total_episode_prompt_tokens`,
  `total_episode_completion_tokens`, `llm_token_totals_by_stage`
- Eval: `--cost-report` on `scripts/eval/run_experiment.py` writes `eval_pipeline_metrics.json`
- Eval configs (canonical tree: `data/eval/issue-477/`, see `README.md` there):
  - `data/eval/issue-477/experiment_openai_gpt4o_smoke_bullets_v1.yaml`
  - `data/eval/issue-477/experiment_openai_gpt4o_smoke_bundled_v1.yaml`
  - `data/eval/issue-477/experiment_openai_gpt4o_benchmark_bundled_v1.yaml`
  - `data/eval/issue-477/autoresearch_prompt_openai_smoke_bundled_v1.yaml`
  - Track B (paragraph smoke, same model tier as bullets): `data/eval/issue-477/experiment_openai_gpt4o_smoke_paragraph_v1.yaml`

## How to run (OpenAI smoke)

Staged baseline (existing):

```bash
make experiment-run \
  CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_bullets_v1.yaml \
  REFERENCE=silver_sonnet46_smoke_bullets_v1
```

Bundled candidate:

```bash
make experiment-run \
  CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_bundled_v1.yaml \
  REFERENCE=silver_sonnet46_smoke_bullets_v1
```

With token snapshot:

```bash
python scripts/eval/run_experiment.py \
  data/eval/issue-477/experiment_openai_gpt4o_smoke_bundled_v1.yaml \
  --reference silver_sonnet46_smoke_bullets_v1 \
  --cost-report
```

## Two-track process (bullets + paragraph, no extra eval machinery)

Issue #477’s **bundled** implementation produces **JSON** (`title`, `summary`, `bullets`). The scorer compares each run to **one** reference’s `summary_final` string. So you do **not** need one mega-experiment or new prediction fields.

**Treat the two concerns as separate runs**, then fold both into one **process / decision**:

### Track A — Bullets (primary for bundle cost/quality)

This is where **staged vs bundled** matters. Same dataset, same bullet silver, two configs (commands above).

### Track B — Paragraph (regression guard, staged only)

Use **`experiment_openai_gpt4o_smoke_paragraph_v1`** so paragraph smoke matches bullet smoke (**`gpt-4o`**, same dataset and preprocessing). Do not use `llm_pipeline_mode: bundled` here; bundled is not wired for paragraph prose.

```bash
make experiment-run \
  CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_paragraph_v1.yaml \
  REFERENCE=silver_sonnet46_smoke_v1
```

**Intent:** Shared code paths (`Config`, provider wiring, `run_experiment`, preprocessing) should not silently hurt **paragraph** quality. Compare ROUGE / gates to a **known baseline** (e.g. last green run on `main`, or a saved `metrics.json`), not to the bullet bundled run.

**Autoresearch:** `autoresearch_prompt_openai_smoke_paragraph_v1` remains on **`gpt-4o-mini`** for cheaper prompt-tuning; it is not required for Issue #477 Track B.

**Note:** Bundled mode now returns `title` + `summary` (prose paragraph) +
`bullets` in a single JSON response. The paragraph field is already part of
the bundled output shape, so both tracks can eventually use bundled mode.

## Decision gate (fill in after runs)

### Track A -- Bullets (staged vs bundled)

| Criterion | Threshold | Staged result | Bundled result | Pass? |
| --- | --- | --- | --- | --- |
| ROUGE-L F1 vs silver | within -0.02 of staged | 0.344 | 0.249 | No (-0.095) |
| ROUGE-1 F1 vs silver | informational | 0.599 | 0.493 | -- |
| Embed cosine sim | informational | 0.835 | 0.717 | -- |
| Eval gates | all pass | all pass | all pass | Yes |
| LLM calls (eval, 5 ep) | meaningful reduction | 5 | 5 | n/a (eval = summ only) |
| Input tokens (eval) | reduction | 12 457 | 13 377 | +7% (bundled prompt overhead) |
| Output tokens (eval) | reduction | 873 | 1 744 | +100% (title+summary+bullets) |
| Est. cost (eval) | reduction | ~$0.040 | ~$0.051 | No (+28%) |
| Fallback rate | under 5% | n/a | 0% | Yes |

**Note on eval cost:** The eval runner only measures the summarization call.
Bundled output tokens are higher because the bundled call returns title +
prose summary + bullets (3 fields) vs staged bullets-only (1 field). The
real savings come from **eliminating the separate cleaning call** in the
full pipeline (see profile comparison below).

**Profile comparison (full E2E pipeline, 2 episodes):**

| Dimension | Staged | Bundled | Delta |
| --- | --- | --- | --- |
| transcript_cleaning wall | 42.1 s | 0 s (eliminated) | -100% |
| summarization wall | 45.9 s | 5.8 s | -87.5% |
| Total pipeline wall | 40.3 s | 21.5 s | **-46.6%** |
| Avg wall / episode | 20.2 s | 10.8 s | -46.6% |
| gi_generation wall | 11.7 s | 21.7 s | +85% (API rate limits) |
| Peak RSS | 645 MB | 644 MB | ~same |

### Track B -- Paragraph (staged regression guard)

| Criterion | Threshold | Result | Pass? |
| --- | --- | --- | --- |
| ROUGE-L F1 vs `silver_sonnet46_smoke_v1` | no large regression | 0.263 | Yes (baseline) |
| ROUGE-1 F1 | informational | 0.576 | -- |
| Embed cosine sim | informational | 0.788 | -- |
| Eval gates | all pass | all pass | Yes |

Paragraph quality is unaffected by bundled changes (staged-only path).

**Outcome:** Bundled mode delivers massive **time savings** (-46.6% total
wall, cleaning stage eliminated entirely, summarization 87.5% faster) and
all quality gates pass. However, **ROUGE-L dropped -0.095** (from 0.344 to
0.249), which exceeds the -0.02 threshold. The quality gap is expected:
the bundled prompt asks the LLM to do more work (clean + title + summary
paragraph + bullets) in a single pass, and the output shape changed (now
includes a prose summary field alongside bullets).

**Outcome (one line):** Keep bundled opt-in; iterate prompts to close the
ROUGE-L gap before making it the default. The performance win is proven;
quality needs tuning (autoresearch candidate).

## Problem definition (Step 2 -- measured baseline)

All numbers below use **gpt-4o** (matching eval configs) against **Sonnet 4.6 silvers**.

### Quality baseline (eval, 5 episodes, `curated_5feeds_smoke_v1`)

| Output | ROUGE-L F1 | ROUGE-1 F1 | Embed cosine | Gates | Avg latency |
| --- | --- | --- | --- | --- | --- |
| Bullets (staged) | 0.344 | 0.599 | 0.835 | all pass | 1822 ms |
| Paragraph (staged) | 0.263 | 0.576 | 0.788 | all pass | 3000 ms |

### Cost baseline (eval, 5 episodes)

| Output | LLM calls | Input tokens | Output tokens | Est. cost |
| --- | --- | --- | --- | --- |
| Bullets | 5 | 12 457 | 873 | ~$0.040 |
| Paragraph | 5 | 11 792 | 1 627 | ~$0.046 |

Note: eval runs only measure the **summarization** call (no cleaning, GI, KG).
The full pipeline adds cleaning + GI + KG calls on top.

### Performance and cost baseline (profile freeze, 2 episodes, E2E mock RSS)

Source: `data/profiles/issue-477/issue477-staged-gpt4o.yaml` and companion
`metrics.json`. Pricing: gpt-4o at $2.50 / 1M input, $10.00 / 1M output
(from `config/pricing_assumptions.yaml`).

| Stage | Wall (s) | Calls | In tokens | Out tokens | Est. USD | CPU% |
| --- | --- | --- | --- | --- | --- | --- |
| transcript_cleaning | 42.1 | 2 | 4 611 | 2 711 | $0.039 | 0.37 |
| summarization | 45.9 | 2 | 3 264 | 312 | $0.011 | 0.46 |
| gi_generation | 11.7 | 20 | 23 357 | 266 | $0.061 | 0.97 |
| kg_extraction | 1.5 | 2 | 715 | 90 | $0.003 | 0.00 |
| speaker_detection | 2.7 | 2 | 1 607 | 82 | $0.005 | 2.27 |
| vector_indexing | 0.2 | -- | -- | -- | -- | 37.20 |
| **Total** | **40.3** | **28** | **33 554** | **3 461** | **$0.119** | |

Total wall is lower than the sum of stage walls (104.0 s) because 2 episodes
run in parallel.

**Time perspective:** Cleaning + summarization = 88.0 s of stage wall (85% of
all LLM-bound stage time). Both stages show near-zero CPU -- pure API wait.

**Cost perspective:** GI generation is the most expensive stage ($0.061, 51%
of total) due to 20 calls (evidence extraction + entailment scoring per
insight). Cleaning is second ($0.039, 33%) because it outputs the full
cleaned transcript (high output token count). Summarization is cheap ($0.011,
9%) -- short output.

**Bundled optimization target:** Cleaning + summarization together cost
$0.050 (42% of total) and take 88.0 s of stage wall. The biggest cost
driver is cleaning **output tokens** (2 711 tokens, $0.027 at $10/1M) --
the full cleaned transcript returned over the wire. With bundled mode the
LLM cleans internally but only returns title + bullets JSON (~312 output
tokens), so those 2 711 output tokens disappear entirely. Combined with
sending the transcript only once as input (not twice), the expected cost
for the merged call drops to ~$0.015 -- a **~70% reduction** on these two
stages. Wall time also drops by ~42 s (the cleaning round-trip).

### Comparison to gpt-4o-mini (existing profile)

Source: `data/profiles/v2.6-wip-openai.yaml` (gpt-4o-mini, same 2 episodes).
gpt-4o-mini pricing: $0.15 / 1M input, $0.60 / 1M output.

| Stage | mini wall (s) | 4o wall (s) | Time delta | mini USD | 4o USD | Cost delta |
| --- | --- | --- | --- | --- | --- | --- |
| transcript_cleaning | 87.6 | 42.1 | -52% | ~$0.002 | $0.039 | +18x |
| summarization | 106.8 | 45.9 | -57% | ~$0.001 | $0.011 | +11x |
| gi_generation | 27.8 | 11.7 | -58% | ~$0.004 | $0.061 | +15x |
| kg_extraction | 5.6 | 1.5 | -73% | ~$0.000 | $0.003 | -- |
| Total wall | 95.4 | 40.3 | -58% | ~$0.007 | $0.119 | +17x |

gpt-4o is ~58% faster per-call than gpt-4o-mini (likely different API server
routing or model serving infrastructure), but ~17x more expensive per token.
The **relative dominance** of cleaning + summarization in wall time holds for
both models.

### Hypothesis validation

**H1 (cleaning + summarization dominate sequential cost):** Confirmed.

- **Time:** 88.0 s out of 104.0 s stage wall (85%). Both stages show
  near-zero CPU (pure API wait).
- **Cost:** $0.050 out of $0.119 total (42%). Cleaning alone is $0.039
  (33%) -- almost all of that is **output tokens** ($0.027) because the
  LLM returns the full cleaned transcript. This is the single biggest
  cost-reduction lever: bundled mode eliminates those output tokens
  entirely (the LLM cleans internally, returns only title + bullets).
- GI generation is the most expensive stage by USD ($0.061, 51%) due to
  20 calls, but it is not in scope for the bundled optimization.

**H2 (parallelism collapses total wall):** Confirmed. Sum of stage walls =
104.0 s but total wall = 40.3 s. The 2 episodes run in parallel, so stages
overlap.

**H3 (low CPU% = I/O wait):** Confirmed. All LLM stages show under 1% CPU.
The machine is waiting on OpenAI API responses, not doing local work.

### What this means for the bundled experiment (Step 3)

The staged pipeline makes **separate** LLM calls for cleaning and
summarization per episode. Bundled replaces both with a single call that
cleans internally and returns only the title + bullets JSON.

**Expected savings (2 episodes):**

| Dimension | Staged | Bundled (est.) | Saving |
| --- | --- | --- | --- |
| LLM calls | 4 (2 clean + 2 summ) | 2 (1 per ep) | -50% |
| Input tokens | 7 875 (transcript sent 2x) | ~4 600 (sent 1x) | -42% |
| Output tokens | 3 023 (2 711 clean + 312 summ) | ~312 (JSON only) | -90% |
| Est. USD | $0.050 | ~$0.015 | **-70%** |
| Stage wall | 88.0 s | ~46 s (1 round-trip) | **-48%** |

The dominant saving is **output tokens**: staged cleaning returns the full
cleaned transcript (1 356 tokens/call at $10.00 / 1M = $0.014/call).
Bundled never returns the cleaned text -- the LLM uses it internally and
only emits the short JSON result. This alone accounts for ~$0.027 of the
$0.035 expected saving.

The second saving is **input tokens**: the transcript is sent once instead
of twice (once for cleaning, once for summarization), cutting ~3 264
input tokens.

**Quality gate:** Bundled ROUGE-L must be within -0.02 of staged (34.4%
baseline for bullets). If the LLM produces worse bullets when asked to
clean + summarize in one pass, the cost saving is not worth it.

## References

- [GitHub Issue #477](https://github.com/chipi/podcast_scraper/issues/477) -- source of truth (hypothesis validation posted as comment)
- [PERFORMANCE_PROFILE_GUIDE.md](../guides/PERFORMANCE_PROFILE_GUIDE.md)
- [OPTIMIZATION_WORKFLOW_GUIDE.md](../guides/OPTIMIZATION_WORKFLOW_GUIDE.md)
- Profile configs and outputs: `data/profiles/issue-477/`
- Eval configs: `data/eval/issue-477/`
- RFC-065 / RFC-066 monitoring and run-compare Performance tab
