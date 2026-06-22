# Chunk-7 v2 scoreboard audit (post retro re-fingerprint, 2026-06-22)

Generated from the v2 retro-fingerprints in `data/eval/runs/...`.
Source script: `scripts/eval/fingerprint/refingerprint_from_run.py`.
Each row's `fingerprint_hash` is a v2.0 sha256 over the full
fingerprint dict (excluding run_id), so two runs with materially
different configs now have different hashes (this used to be invisible —
see `FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md`).

## KG runs

| candidate | backing_model_id | temp | max_tok | postprocessor | inference_target | fingerprint_hash |
| --- | --- | ---: | ---: | --- | --- | --- |
| `mistral_small_3_2_24b` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 0.15 | 800 | `strip_r1_reasoning` | dgx-vllm | `ffdeaae33dd4` |
| `qwen3_5_35b_a3b` | `Qwen/Qwen3.5-35B-A3B` | 0.0 | 800 | `—` | dgx-vllm | `0ed0b3a0a8b4` |
| `ministral_3_14b` | `mistralai/Ministral-3-14B-Instruct-2512` | 0.05 | 800 | `—` | dgx-vllm | `9e1b4c10b834` |
| `gemini25_flash_lite` | `_(unknown)_` | 0.0 | 800 | `—` | cloud-gemini | `30c4fb3626bd` |
| `gemma_4_26b_a4b` | `google/gemma-4-26B-A4B-it` | 0.0 | 800 | `—` | dgx-vllm | `68d526b266c1` |
| `magistral_small_2509` | `mistralai/Magistral-Small-2509` | 0.7 | 4096 | `strip_r1_reasoning` | dgx-vllm | `0398eaeb4fc5` |
| `qwen3_30b_a3b_instruct_2507` | `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` | 0.0 | 800 | `—` | dgx-vllm | `e0044e09a434` |
| `moonlight_16b_a3b` | `moonshotai/Moonlight-16B-A3B-Instruct` | 0.0 | 800 | `—` | dgx-vllm | `12b4c3bbf4e3` |
| `deepseek_v2_lite_chat` | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 0.0 | 800 | `decode_r1_byte_level` | dgx-vllm | `a4e204efb973` |

## GI runs

| candidate | backing_model_id | temp | max_tok | gi_insight_src | gi_max_insights | inference_target | fingerprint_hash |
| --- | --- | ---: | ---: | --- | ---: | --- | --- |
| `mistral_small_3_2_24b` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 0.15 | 800 | `provider` | 12 | dgx-vllm | `465d17f31a9b` |
| `qwen3_5_35b_a3b` | `Qwen/Qwen3.5-35B-A3B` | 0.0 | 800 | `provider` | 12 | dgx-vllm | `16b6f8fd276d` |
| `ministral_3_14b` | `mistralai/Ministral-3-14B-Instruct-2512` | 0.05 | 800 | `provider` | 12 | dgx-vllm | `0fcbcc710644` |
| `gemini25_flash_lite` | `_(unknown)_` | 0.0 | 800 | `provider` | 12 | cloud-gemini | `579c5d99276d` |
| `gemma_4_26b_a4b` | `google/gemma-4-26B-A4B-it` | 0.0 | 800 | `provider` | 12 | dgx-vllm | `8a02abd78bec` |
| `magistral_small_2509` | `mistralai/Magistral-Small-2509` | 0.7 | 4096 | `provider` | 12 | dgx-vllm | `0aa699043306` |
| `qwen3_30b_a3b_instruct_2507` | `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` | 0.0 | 800 | `provider` | 12 | dgx-vllm | `daa30f1b0ee8` |
| `moonlight_16b_a3b` | `moonshotai/Moonlight-16B-A3B-Instruct` | 0.0 | 800 | `provider` | 12 | dgx-vllm | `5b369b10c566` |
| `deepseek_v2_lite_chat` | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 0.0 | 800 | `provider` | 12 | dgx-vllm | `6112ea538879` |

## Key audit findings

- **All 18 chunk-7 scoreboard runs now have v2.0 fingerprints** with full
  generation_params, backing_model_id, task_pipeline, inference_target,
  dataset_content_hash, and a sha256 fingerprint_hash over the full dict.
- **Vendor sampling drift is visible**: Mistral-Small-3.2 KG/GI carry
  `temperature: 0.15` (vendor recommendation, just landed in commit
  `a4bceb81`). Ministral carries `temperature: 0.05`. Other candidates are
  at `temperature: 0.0` per the chunk-7 greedy discipline.
- **Magistral max_tokens = 4096** (reasoning model needs headroom for the
  THINK block; bumped from 800 in commit `1d17d19d`). Every other
  candidate is at 800.
- **DeepSeek-V2-Lite-Chat is the ONLY candidate with a non-default
  postprocessor in KG.** The KG row shows
  `postprocessor: decode_r1_byte_level` while every other vLLM KG run
  shows `—` (no postprocessor or the default `strip_r1_reasoning`).
  This is the same postprocessor `EVAL_1016_FINAL_REPORT_2026_06_17.md`
  flagged as a HARNESS BUG:
  > *"Postprocessor `decode_r1_byte_level` is applied to summary text
  > but NOT to GI/KG `node.label` fields. Tasks GitHub issue to fix;
  > rerun is fast."*
  The retro fingerprint surfaces the config divergence; the historical
  report explains its failure mode. **This is the load-bearing piece of
  evidence for the DeepSeek-V2-Lite-Chat decision** — the 1.5% / 3.8%
  scores partly reflect a known harness wiring gap on a model that
  REQUIRES that postprocessor (vs models that don't carry one).
  Before judging DeepSeek's model quality, fix the wiring (apply
  `decode_r1_byte_level` to node.label fields in GI/KG pipeline output)
  and re-run. If scores remain at the floor after the fix, then it's
  genuinely a too-weak model.
- **All `_retro_audit.unknown_fields` lists are empty for vLLM candidates**
  (chunk-7 recovery table filled inference_args + image). Gemini KG/GI
  list `backing_model_id` / `inference_args` / `inference_image` as
  unknown — cloud provider, no equivalent recovery surface.

## Methodology

- `scripts/eval/fingerprint/refingerprint_from_run.py` reads the existing
  `fingerprint.json` + the eval YAML at `data/eval/configs/<task>_<id>.yaml`,
  reconstructs the missing fields, archives the original at
  `fingerprint.v1.original.json`, replaces the in-place file with the v2
  shape.
- Operator authorisation 2026-06-22 (recorded in
  `feedback_never_mutate_historical_artifacts` memory rule).
- `dataset_content_hash` is captured AT AUDIT TIME, not at original-run
  time. `dataset_content_hash_audited_at` records when. This means the
  hash captures TODAY's materialized dataset content; if transcripts have
  been edited since the run, the retro hash is the post-edit content.
  Operator accepted this caveat to avoid pretending we know the original
  content state.
