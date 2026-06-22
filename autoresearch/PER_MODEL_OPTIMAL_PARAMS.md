# Per-model optimal vLLM-on-GB10 params — #1016 observations

Live notes collected while Round 2 runs the harness fixes. Used as input to #1022
(systematic vLLM-on-GB10 tuning exploration). Each model row captures:

- **VLLM_GPU_MEM_UTIL** used (the env var that scaled `--gpu-memory-utilization`)
- **max-model-len** required (from compose)
- **max-num-batched-tokens** required (added per Gemma 4 multimodal constraint)
- **trust-remote-code** required (yes/no)
- **GPU memory at load** (from `Model loading took XX GiB memory`)
- **Boot wall-clock** (download + load + compile + autotune)
- **Boot stage cache hits** (torch.compile cache hits on second boot)
- **KV cache % observed during inference** (from `/metrics` poll)
- **TTFT / TPOT** (from histogram buckets)
- **Best chars/episode / s/episode** observed (gate run)
- **Required prompt adaptations** (strict / anti-think / etc.)
- **Failure modes if any**

| Model | UTIL | max-model-len | max-num-batched | trust-remote | Load mem | Boot | KV % obs | s/ep | chars/ep | Prompt adapt | Failure mode |
| ----- | ---: | ------------: | --------------: | :----------: | -------: | ---: | -------: | ---: | -------: | :----------- | ----------- |
| Qwen3-30B-A3B-Instruct-2507 | 0.65 | 32768 | default 2048 | no | ~60 GB | ~12 min | tbd | 25.6 | 4472 (over-spec) | tight prompts | floor bug |
| Qwen3.5-35B-A3B | 0.65 | 32768 | default 2048 | no | 65.5 GB | ~15 min | tbd | 21.7 | 3779 (over-spec) | tight prompts | floor bug |
| Mistral-Small-3.2-24B-Instruct-2506 | 0.55 | 32768 | default 2048 | no | tbd | ~10 min | tbd | 72.6 | 2057 | none | none |
| Magistral-Small-2509 | 0.55 | 32768 | default 2048 | no | tbd | ~10 min | tbd | 64.8 | 1762 | none (reasoning_parser=mistral) | none |
| Ministral-3-14B-Instruct-2512 | 0.65 | 32768 | default 2048 | no | tbd | ~6 min | tbd | 30.9 | 2678 | none | none |
| Gemma-4-26B-A4B-it | 0.65 | 32768 | **4096 (required for multimodal)** | no | 48.5 GB | ~32 min | tbd | 14.6 | 2009 | none | needs max-num-batched bump |
| Moonlight-16B-A3B-Instruct | 0.55 | **8192 (max_position_embeddings limit)** | 4096 | no | tbd | ~12 min | tbd | 8.8 | 1674 | none | needs max-model-len reduce |
| Mistral-Small-4-119B-NVFP4 | 0.80 | 32768 | 4096 | no | ~67 GB | ~32 min | OOM | n/a | n/a | n/a | OOM on inference — needs max-num-seqs=4 + max-model-len=8192 |
| Llama-3.3-70B-Instruct-NVFP4 | 0.70 | 32768 | 4096 | no | tbd | ~18 min | tbd | 85 (3 eps) | 2283-2595 | strict prompt | verbosity at 800 cap |
| DeepSeek-V2-Lite-Chat | 0.55 | 32768 | 4096 | no | tbd | ~9 min | tbd | **4.8 (decoded)** | **983 (decoded)** | strict prompt + decode_r1_byte_level postprocessor | round 1: BPE + verbosity |
| DeepSeek-R1-0528-Qwen3-8B | 0.55 | 32768 | 4096 | no | tbd | ~5 min | tbd | n/a | n/a | strict + reasoning-off prompt (TBD) | round 1: reasoning consumed budget |
| Kimi-Linear-48B-A3B-Instruct | 0.80 | 32768 | 4096 | **YES** (tokenization_kimi.py) | tbd | ~38 min | tbd | 10 (2 eps) | 1318 | strict prompt | round 1: verbosity at 800 cap |
| Nemotron-Super-49B-v1_5-FP8 | 0.65 | 32768 | 4096 | no | tbd | ~22 min | tbd | n/a | n/a | "/no_think" or "detailed thinking off" system msg | round 1: reasoning consumed budget |

Updated as runs land. Cross-reference: vllm_kv_metrics.log in same dir.

## Phase 2c Round 3 v1 — actual collected metrics (2026-06-17)

Replaces the speculative `tbd` columns above with real polling data. Source:
`vllm_metrics_*_phase2c.log` files in this directory (one per candidate).

| Candidate | Samples | **KV peak** | TTFT p50 | TPOT p50 | Gen TPS avg | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-35B-A3B | 77 | 1.1% | 250ms | 50ms | 29 tok/s | `--reasoning-parser=qwen3` flag wired |
| Gemma-4-26B-A4B-it | 120 | 0.9% | 250ms | 50ms | 22 tok/s | `--max-num-batched-tokens=4096` retained |
| Moonlight-16B-A3B-Instruct | 72 | 0.2% | 250ms | 50ms | 30 tok/s | `--max-model-len=8192` (hard floor) |
| Qwen3-30B-A3B-Instruct-2507 | 128 | 1.4% | 250ms | 50ms | 27 tok/s | No parser flag (2507 non-thinking only) |
| Ministral-3-14B-Instruct-2512 | 300 | 0.8% | 250ms | 75ms | 15 tok/s | 4 mistral tokenizer flags |
| **Llama-3.3-70B-NVFP4** | 354 | **2.5%** | **500ms** | **200ms** | **5 tok/s** | NVFP4 quant throughput bottleneck |
| DeepSeek-V2-Lite-Chat | 62 | 0.2% | 500ms | 50ms | 33 tok/s | GI/KG output had BPE bug (task #111) |

### Key findings for #1022

1. **Single-stream eval barely touches GPU memory budget.** Peak KV cache
   = 2.5% (Llama 70B); most models < 1.5%. We are running with
   `--max-num-seqs=64` but could drop to `--max-num-seqs=4` and free
   20-40 GiB of KV-cache reservation. That would let tighter-fit models
   (like Kimi-Linear, Mistral-Small-4-119B) potentially boot.

2. **Llama-3.3-70B-NVFP4 is throughput-bound, not memory-bound.**
   - TPOT 200ms (4× cohort avg of 50ms) → 5 tok/s steady-state
   - TTFT 500ms (2× cohort avg of 250ms)
   - KV peak 2.5% means there's massive memory headroom going unused
   - Hypothesis: NVFP4 kernels may not be optimized for our small batches.
     Try `--max-num-batched-tokens=16384+` and `--enable-prefix-caching`.

3. **TTFT/TPOT are remarkably consistent across the small/MoE cohort.**
   - All small/MoE: TTFT 250ms, TPOT 50ms
   - Compute-bound regime; prefill of ~10k input tokens dominates
   - Smaller `max_input_tokens` would cut TTFT but pipeline naturally caps
     summary input at ~10k

4. **Boot times suggest CUDA graph capture is the dominant cost.**
   - Multimodal Gemma 4 + 30B+: 7-8 min cached boot
   - 8B/14B dense: 2-4 min
   - For eval-loop iteration: `--enforce-eager` saves 2-4 min/boot at
     ~10-15% inference perf cost (acceptable for iteration; not for prod)

### Compose defaults updated (2026-06-17, levers A+C applied)

Following the Phase 2c metrics analysis showing KV cache peak 2.5% even at
`max-num-seqs=64`, the autoresearch compose at
`~/agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml` now defaults to:

- `--max-num-seqs=4` (down from 64) — frees 20-40 GiB KV cache reservation
  per the data above. Safe for single-stream eval (we never use concurrent
  seqs). May allow Kimi-Linear / Mistral-Small-4-119B-NVFP4 to boot.
- `--enforce-eager` — added. Skips CUDA graph capture, saving 2-4 min per
  boot at ~10-15% inference perf cost. Acceptable for the eval-loop swap
  cadence; not for prod (where graph capture amortizes).

These are persistent compose defaults. To bypass for a specific run (e.g. a
prod-mirror benchmark), remove `--enforce-eager` and bump `--max-num-seqs`
back to 64 in the compose before that swap.

### Original `tbd` columns now filled in (per-model)

Updated PER_MODEL_OPTIMAL_PARAMS table values (replacing the `tbd` placeholders
in the original Round 2 table at top of this file):

- **KV % obs**: all candidates 0.2-2.5% during single-stream eval, peak 2.5%
- **Boot wall-clock** (cached): see § "Boot times" in EVAL_1016_FINAL_REPORT_2026_06_17.md
- **Gen TPS**: 5-33 tok/s, mostly clustered around 20-30 except Llama 70B at 5

## RFC-097 chunk-7 workaround sweep — new findings (2026-06-21)

The chunk-7 silver rebuild sweep (2026-06-21) exercised every candidate
under the v2-emitting KG+GI pipeline. The initial sweep bypassed this
compendium (`docker run` with default flags) and 5 of 8 candidates
failed. The retry sweep used per-model flags from this compendium plus
two new findings, captured here so future sweeps don't repeat the work:

### Finding 1: `--language-model-only` for Mistral3 + Pixtral models

Mistral-Small-3.2-24B, Magistral-Small-2509, and Ministral-3-14B-Instruct-2512
all use the `Mistral3ForConditionalGeneration` architecture, which embeds
a Pixtral vision tower in the same checkpoint as the text model. Our
autoresearch workload is text-only; the vision tower otherwise loads
into GPU memory and throttles text-generation throughput (Mistral-3.2
ran at 5 tok/s without this flag vs 50+ tok/s with).

**Add `--language-model-only` to vLLM serve flags for any
Mistral3-family text-only workload.** Source in
`vllm/config/multimodal.py`: *"If True, disables all multimodal inputs
by setting all modality limits to 0. Equivalent to setting
`--limit-mm-per-prompt` to 0 for every modality."* In
`encoder_cache_manager.compute_mm_encoder_budget`, the empty-modalities
branch returns `(0, 0)` and skips the
`disable_chunked_mm_input + max_tokens_per_mm_item` check that bit
Gemma-4-26B-A4B too (also confirmed compatible with the existing Gemma
fix `--max-num-batched-tokens 4096`).

### Finding 2: Magistral reasoning model needs `max_length ≥ 4096` in eval config

Magistral-Small-2509 is an explicit reasoning model that emits
`[THINK]…[/THINK]` blocks before its real response. Vendor sets
`max_tokens=131072`; our autoresearch KG/GI configs default to
`max_length: 800`. The THINK block consumes the budget before the
JSON extraction can fit. `--reasoning-parser=mistral` strips the
THINK trace from the chat response, but the trace **still counts
against `max_tokens`** during generation, so the parser alone is not
sufficient.

**Bump `params.max_length: 800 → 4096`** in any KG/GI config
targeting a reasoning model (Magistral, future R1-family). Already
applied to:

- `data/eval/configs/kg_autoresearch_prompt_vllm_magistral_small_2509_dev_v1.yaml`
- `data/eval/configs/gi_autoresearch_prompt_vllm_magistral_small_2509_dev_v1.yaml`

### Finding 3: Verify HF model id against the DGX cache before swapping

Ministral-3-14B's chunk-7 first-attempt failure was mis-attributed to
"MoE init fault" — actually we'd used HF id `mistralai/Ministral-3-14B`,
which **does not exist in HF**. The DGX cache only has
`mistralai/Ministral-3-14B-Instruct-2512`. The container died at HF
resolver instantly. A `ssh dgx-llm-1 'ls /opt/llm-models/huggingface/hub/'`
check before the swap would have caught this.

### Finding 4: 5 of 8 candidates failed when sweep bypassed this doc

`/tmp/sequential_runs.sh` (chunk-7 first sweep) issued `docker run`
with `--gpu-memory-utilization 0.60 --max-model-len 32768
--max-num-seqs 128` for every model — the homelab compose defaults
at the time, NOT the per-model flags this compendium records. Result:
Gemma, Mistral-3.2, Magistral, Moonlight, Ministral all failed in
ways the compendium's Phase 2c rows had already documented as
"working with these specific flags". Lesson is now an AGENTS.md hard
rule: *Read this doc before any multi-model sweep; never default-flag
a model with a documented row.*

### Updated final scoreboard (2026-06-21 v2 pipeline + workarounds)

Best candidates after the chunk-7 retries:

| Candidate | KG opus47 / sonnet46 | GI opus47 / sonnet46 |
| --- | ---: | ---: |
| `mistral_small_3_2_24b` | **82.8% / 81.9%** (KG leader) | 91.2% / 85.0% |
| `qwen3_5_35b_a3b` | 82.1% / 79.5% | 85.0% / 87.5% |
| `ministral_3_14b` | 79.9% / 81.9% | 83.8% / 85.0% |
| `gemini25_flash_lite` | 76.1% / 73.2% | **96.2% / 92.5%** (GI leader) |
| `gemma_4_26b_a4b` | 74.6% / 74.8% | 90.0% / 92.5% |
| `magistral_small_2509` | 72.4% / 72.4% | 90.0% / 87.5% |
| `qwen3_30b_a3b_instruct_2507` | 71.6% / 74.0% | 91.2% / 90.0% |
| `moonlight_16b_a3b` | 61.2% / 57.5% | 63.7% / 56.2% |
| `deepseek_v2_lite_chat` | 1.5% / 0.8% | 3.8% / 1.2% |

Full scoreboard + per-model fix details:
`docs/guides/eval-reports/EVAL_RFC097_V2_BASELINE_2026_06_20.md`,
`docs/guides/eval-reports/EVAL_RFC097_CHUNK7_VLLM_WORKAROUNDS_2026_06_21.md`.
