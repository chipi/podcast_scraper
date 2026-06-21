# Chunk 7 — per-model vLLM:26.05-py3 workarounds (2026-06-21)

Live research log for the 5 candidates that failed the chunk 7 silver rebuild
sweep on `nvcr.io/nvidia/vllm:26.05-py3`. One model at a time, deep research
first, then retry, then advance.

Pack-leader cohort (Qwen3-30B-A3B-Instruct-2507, Qwen3.5-35B-A3B, Gemini)
is already green — see `EVAL_RFC097_V2_BASELINE_2026_06_20.md`
(same directory).

## Canonical reference: `PER_MODEL_OPTIMAL_PARAMS.md`

**`autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` is the per-model
flag compendium** (lives in `autoresearch/` next to `README.md` and
`MODEL_PLAYBOOK.md` — the three-doc pair that describes how to run
autoresearch). Every model below has a row in its Phase 2c (2026-06-17)
table that proves vLLM-on-GB10 worked with specific flags. The chunk 7
sweep scripts (`/tmp/sequential_runs.sh`, `/tmp/swap_run.sh`) bypassed
the homelab compose and used `docker run` directly with DEFAULT flags —
which is the root cause of these failures.

**Working approach**: every retry below pulls flags from
`PER_MODEL_OPTIMAL_PARAMS.md`. The sweep's "discoveries" are not new —
they're rediscoveries of what the chunk 7 swap script omitted.

After all retries land, update `PER_MODEL_OPTIMAL_PARAMS.md` with any
new findings (e.g. `--language-model-only` as an alternative for Gemma,
if it turns out cleaner than `--max-num-batched-tokens 4096`).

---

## Model 1: `google/gemma-4-26B-A4B-it`

**Failure signature**

```text
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
is larger than max_num_batched_tokens (2048).
Please increase max_num_batched_tokens.
```

Container died at 20s on the previous sweep.

**Architecture (from `config.json` in the DGX cache)**

- `architectures: Gemma4ForConditionalGeneration` — multimodal model
  (image + audio + video)
- `model_type: gemma4`; text inner config `model_type: gemma4_text`
- Sparse MoE in the text tower:
  - `enable_moe_block: true`
  - `num_experts: 128`, `top_k_experts: 8`
  - `hidden_size: 2816`, `intermediate_size: 2112`, `moe_intermediate_size: 704`
  - `num_hidden_layers: 30` (mostly sliding_attention, every 6th = full_attention)
  - `max_position_embeddings: 262144` (256k context)
- Vision tower: `vision_soft_tokens_per_image: 280`
- Audio: `audio_seq_length: 750`, 40ms/token
- Video: `num_frames: 32`, `max_soft_tokens: 70` per frame
- **`use_bidirectional_attention: "vision"`** — this is the trigger

**Root cause (from vLLM source — `vllm/platforms/cuda.py` 26.05)**

vLLM detects `use_bidirectional_attention=="vision"` and **forces**
`scheduler_config.disable_chunked_mm_input = True` ("Forcing
--disable_chunked_mm_input for models with multimodal-bidirectional
attention.").

With chunked MM input disabled, `encoder_cache_manager.compute_mm_encoder_budget`
requires `max_num_batched_tokens >= max_tokens_per_mm_item`, where
`max_tokens_per_mm_item = max(image_seq_len, audio_seq_len, video_total_tokens)`.
For Gemma4 that's `max(280, 750, 32*70 + overhead) = 2496`.

The default `max_num_batched_tokens` is 2048 → ValueError. The error message's
suggestion ("Please increase max_num_batched_tokens") is correct but
sub-optimal for our workload: we send **text only**, so the encoder budget
should be 0.

**Two fixes vLLM exposes**

1. **`--language-model-only`** — single bool flag. Source:
   `vllm/config/multimodal.py`:
   > "If True, disables all multimodal inputs by setting all modality limits
   > to 0. Equivalent to setting `--limit-mm-per-prompt` to 0 for every
   > modality."
   In `encoder_cache_manager.compute_mm_encoder_budget` the branch
   `if not mm_max_toks_per_item:` returns `(0, 0)` and skips the failing
   `disable_chunked_mm_input` check entirely. **This matches our actual
   workload (text-only) — preferred.**
2. **`--max-num-batched-tokens 4096`** (or higher) — keeps MM machinery in
   memory; encoder cache holds unused video/audio/image slots. Wasteful for
   text-only, but a valid fallback if `--language-model-only` doesn't ship
   in this vLLM build for some reason.

**Known-good config (from `PER_MODEL_OPTIMAL_PARAMS.md` Phase 2c 2026-06-17)**

Already-evidenced on this hardware + image (22 tok/s, KV peak 0.9%, 120
samples polled):

| Flag | Value | Why |
| --- | --- | --- |
| `--max-num-batched-tokens` | `4096` | required for multimodal (≥ 2496 mm-item budget) |
| `--max-model-len` | `32768` | full Gemma4 working context |
| `--gpu-memory-utilization` | `0.65` | matches Phase 2c row |
| `--max-num-seqs` | `4` | Phase 2c lever-A update; single-stream eval workload |
| `--enforce-eager` | (flag) | Phase 2c lever-C update; skips CUDA graph capture (-2-4 min boot, ~10-15% steady-state perf cost) |
| `--dtype` | `bfloat16` | matches checkpoint |

**Alternative discovered during this research**

`--language-model-only` — single bool that vLLM source says "disables all
multimodal inputs by setting all modality limits to 0". For our text-only
workload this would be cleaner (frees encoder cache GPU memory). NOT in
PER_MODEL_OPTIMAL_PARAMS.md yet — defer adopting it until after the
known-good `--max-num-batched-tokens 4096` retry confirms the cohort.

**Plan**

Apply the Phase 2c known-good config above. 20-min readiness budget.

**Verify after retry**

- vLLM startup completes without the `max_tokens_per_mm_item` ValueError
- KG + GI runs produce 10/10 predictions
- Score against silver_opus47 / silver_sonnet46 (both KG + GI)
- Append result row to `EVAL_RFC097_V2_BASELINE_2026_06_20.md` v2 scoreboard

**Status: **✅ FIXED 2026-06-21****

Boot in 22min (cached weights, `--enforce-eager` saves CUDA graph
capture). KG: 10/10 episodes in 14.5 min (~80s/ep). GI: 10/10 in
~24 min. Scoring: opus47 KG **74.6%** (topics 67.3%, entities
**100%**) + sonnet46 KG **74.8%** (topics 67%, entities **100%**) +
opus47 GI **90.0%** (avg sim 0.808) + sonnet46 GI **92.5%** (avg sim
0.781). Massive jump from stale-cohort (30.6/32.3 KG, 42.5/45.0 GI).
(GI initially hung on episode 1 due to laptop location change →
Tailscale TCP drop; re-run cleanly via `gemma_gi_retry` step in the
chain orchestrator.)

---

## Model 2: `moonshotai/Moonlight-16B-A3B-Instruct`

**Failure signature**

`User-specified max_model_len (32768) > derived max_model_len
(max_position_embeddings=8192)`. Container died with pydantic
ValidationError in `vllm/engine/arg_utils.py:1475` `ModelConfig`
validator.

**Architecture (from `config.json` in the DGX cache)**

- `architectures: DeepseekV3ForCausalLM`
- `model_type: deepseek_v3`
- `max_position_embeddings: 8192` (HARD CAP)
- No top-level vision/audio config; pure text MoE

**Root cause**

Chunk-7 sweep template hardcoded `--max-model-len 32768`. Moonlight's
checkpoint declares 8k as its native cap; vLLM 26.05 refuses to override
without `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (which carries RoPE
out-of-bounds risk). Need to respect the cap.

**Fix applied**

Phase 2c known-good config applied: `--max-model-len 8192` (model's
hard cap), `--gpu-memory-utilization 0.55`, `--max-num-batched-tokens
4096`, `--max-num-seqs 4`, `--enforce-eager`. No extra
Mistral/Moonlight arch flags needed (DeepseekV3 has native vLLM
support).

**Status: **✅ FIXED 2026-06-21****

KG 61.2% / 57.5% (entities 83% — only cohort model below 100%, likely
context-cap related), GI 63.7% / 56.2%. Stale-cohort was 31.3/26.8 KG,
16.2/15.0 GI — large jump but Moonlight is genuinely a smaller MoE
than the leaders.

---

## Model 3: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`

**Failure signature**

vLLM came up fine, KG started, generation throttled at **~5 tok/s**
(vs ~50 for Qwen3-30B and ~80 for Qwen3.5-35B). Operator-killed.

**Architecture (from `config.json` in the DGX cache)**

- `architectures: Mistral3ForConditionalGeneration`
- `model_type: mistral3`
- `vision_config: present` (Pixtral vision tower)
- No audio

**Root cause**

Two-fold:

1. **Vision tower loaded for text-only workload**: Mistral3 includes the
   Pixtral vision encoder. vLLM allocates it even when no image input
   is sent, throttling text generation throughput.
2. **HF tokenizer instead of mistral-common**: Mistral-3.2 ships
   `SYSTEM_PROMPT.txt` + `tekken.json` in `mistral` format. Without
   `--tokenizer-mode mistral --config-format mistral --load-format mistral`
   we pay subtle tokenization drift on every chat call.

**Fix applied**

`--tokenizer-mode mistral --config-format mistral --load-format mistral
--language-model-only` (the last flag skips vision tower init entirely;
since we're text-only the encoder cache is empty per
`encoder_cache_manager.compute_mm_encoder_budget`'s
`if not mm_max_toks_per_item: return (0, 0)` branch).

**Status: **✅ FIXED 2026-06-21****

**KG 82.8% / 81.9% — best in the cohort.** Topics 78%/76%, entities
**100%**. GI **91.2% / 85.0%** (avg sim 0.823 / 0.773 — ties Qwen3-30B
for GI lead vs opus47). Stale-cohort was 39.6/38.6 KG, 25.0/35.0 GI —
+43pp KG, +66pp GI.

---

## Model 4: `mistralai/Magistral-Small-2509`

**Failure signature**

Container died at 10s with no captured stdout. (Same image and base
flags as Mistral-3.2 / Ministral.)

**Architecture (from `config.json` in the DGX cache)**

- `architectures: Mistral3ForConditionalGeneration` (same family)
- `model_type: mistral3`
- `vision_config: present` — but the **distinguishing trait** is that
  Magistral is a **reasoning model** that emits `[THINK]/[/THINK]`
  blocks before its real response

**Root cause**

Three-fold:

1. **Vendor REQUIRED flags missing**: mistral tokenizer trio (see
   Model 3) **plus** `--reasoning-parser=mistral` and
   `--tool-call-parser=mistral`. Chunk-7 sweep had none.
2. **Vision tower still loaded** (same Pixtral concern as Model 3).
3. **`max_tokens=800` would consume the THINK block**: vendor sets
   `max_tokens=131072` because the model thinks for ~600+ tokens before
   producing an answer. With our 800-token cap, the `[THINK]` block
   eats the budget and the JSON extraction never fits.

**Fix applied**

`--reasoning-parser mistral --tokenizer-mode mistral --config-format
mistral --tool-call-parser mistral --language-model-only` + bump the
eval config `max_length: 800 → 4096` in both
`kg_autoresearch_prompt_vllm_magistral_small_2509_dev_v1.yaml` and the
GI counterpart.

**Status: **✅ FIXED 2026-06-21****

KG 72.4% / 72.4% (entities **100%**), GI 90.0% / 87.5% (avg sim 0.826
/ 0.783). Stale-cohort was 40.3/37.0 KG, 25.0/22.5 GI — +32pp KG,
+65pp GI. The reasoning model produces solid extractions when given
breathing room.

---

## Model 5: `mistralai/Ministral-3-14B-Instruct-2512`

**Failure signature**

Container died **before any model-loading log lines reached stdout**
(initial sweep). Earlier in the sweep cycle the SAME failure was
mis-attributed to "MoE init fault" — actually a wrong-HF-id problem.

**Architecture (from `config.json` in the DGX cache)**

- `architectures: Mistral3ForConditionalGeneration` (same family)
- `model_type: mistral3`
- `vision_config: present` (Pixtral)
- No top-level MoE flag (despite the chunk-7 mis-diagnosis)

**Root cause**

Chunk-7 sweep used HF id `mistralai/Ministral-3-14B`, which **does not
exist in HF**. The DGX cache only has
`models--mistralai--Ministral-3-14B-Instruct-2512`. The container's
HF resolver fell over instantly, hence "died with no stdout". Phase 2c
table in `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` had the correct id
all along.

**Fix applied**

Corrected HF id `mistralai/Ministral-3-14B-Instruct-2512` + the same
mistral tokenizer trio + `--language-model-only` + Phase 2c flags
(util 0.65, max-num-batched 4096, max-num-seqs 4, enforce-eager).

**Status: **✅ FIXED 2026-06-21****

KG 79.9% / 81.9% (entities **100%**), GI 83.8% / 85.0% (avg sim 0.793
/ 0.768). Stale-cohort was 33.6/35.4 KG, 25.0/30.0 GI — +46pp KG,
+59pp GI. A 14B model holding its own at #3 KG in the cohort is
striking.

---

## Lessons (for the per-model compendium)

1. **`--language-model-only` is the right default for Mistral3+Pixtral**
   models on text-only workloads. Skips encoder cache + vision-tower
   init; no quality cost since we never send images.
2. **The mistral tokenizer trio** (`--tokenizer-mode mistral
   --config-format mistral --load-format mistral`) is mandatory for
   any `mistralai/*` checkpoint that ships `mistral`-format
   tokenizer/config files. Subtle tokenization drift otherwise.
3. **Reasoning models need a `max_tokens` budget that covers the
   `[THINK]` block AND the answer.** For our KG/GI extraction tasks,
   `max_length: 4096` is the floor. Below that, the THINK block
   consumes the budget and the JSON never fits.
4. **Always verify HF id against the DGX cache** before swapping. A
   typo'd id dies instantly with no useful error.
5. **`PER_MODEL_OPTIMAL_PARAMS.md` is the source of truth** (now
   `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`). Every model that
   booted on this hardware has a row with its known-good flags;
   bypassing it is what burned chunk 7 initially. New AGENTS.md
   rule: read it before any sweep; update it after every sweep.
   See also the new `feedback_deep_research_per_model` +
   `reference_per_model_optimal_params` memories.
