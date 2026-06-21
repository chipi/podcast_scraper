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

### Failure signature

```text
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
is larger than max_num_batched_tokens (2048).
Please increase max_num_batched_tokens.
```

Container died at 20s on the previous sweep.

### Architecture (from `config.json` in the DGX cache)

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

### Root cause (from vLLM source — `vllm/platforms/cuda.py` 26.05)

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

### Two fixes vLLM exposes

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

### Known-good config (from `PER_MODEL_OPTIMAL_PARAMS.md` Phase 2c 2026-06-17)

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

### Alternative discovered during this research

`--language-model-only` — single bool that vLLM source says "disables all
multimodal inputs by setting all modality limits to 0". For our text-only
workload this would be cleaner (frees encoder cache GPU memory). NOT in
PER_MODEL_OPTIMAL_PARAMS.md yet — defer adopting it until after the
known-good `--max-num-batched-tokens 4096` retry confirms the cohort.

### Plan

Apply the Phase 2c known-good config above. 20-min readiness budget.

### Verify after retry

- vLLM startup completes without the `max_tokens_per_mm_item` ValueError
- KG + GI runs produce 10/10 predictions
- Score against silver_opus47 / silver_sonnet46 (both KG + GI)
- Append result row to `EVAL_RFC097_V2_BASELINE_2026_06_20.md` v2 scoreboard

### Status: **research complete, ready to retry**

---

## Model 2: TBD (next after Gemma confirmed)

## Model 3: TBD

## Model 4: TBD

## Model 5: TBD
