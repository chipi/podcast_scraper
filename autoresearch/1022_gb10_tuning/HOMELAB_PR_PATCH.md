# Homelab autoresearch compose — model swap for #1022 Cell F

Paste-ready patch for `chipi/agentic-ai-homelab` —
`infra/vllm/autoresearch/docker-compose.yml`.

**Goal**: swap the daily-driver model from `Qwen/Qwen3-30B-A3B-Instruct-2507`
(bf16, ~57 GB) to `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` (NVIDIA Model
Optimizer NVFP4, ~18 GB) per the #1022 Cell F champion finding.

## Evidence supporting the swap

- ~2× speed gain on dev_v1 (~38 s/ep end-to-end vs ~73 s/ep baseline)
- No measurable quality regression vs bf16 (same model architecture):
  - Summary embedding cosine 0.8011 vs 0.7948–0.8069 baseline range
  - GI cov 0.425 vs baseline mean 0.595 — actually higher
  - KG topic cov 0.41 vs baseline mean 42.5% — within variance
- Held-out validation on `curated_5feeds_benchmark_v2` (Sonnet 4.6 silver)
  confirmed cross-dataset + cross-vendor robustness:
  - Summary cosine 0.8297 (up from dev_v1)
  - GI cov 0.614 (up from dev_v1's 0.611)
  - KG topic cov 44% (up from dev_v1's 41%)
- Wins GI stage outright across the full #1016 Round 3 cohort:
  0.425 vs Gemma-4 0.413 vs Qwen3.5-35B-A3B 0.363
- 3.7× smaller weight footprint frees ~50 GB unified memory for KV cache
  / future co-residence

Full evidence chain:
`docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md` in
`chipi/podcast_scraper` (branch `feat/autoresearch-followups-2026-06-18`).

## Single-line patch

```diff
diff --git a/infra/vllm/autoresearch/docker-compose.yml b/infra/vllm/autoresearch/docker-compose.yml
--- a/infra/vllm/autoresearch/docker-compose.yml
+++ b/infra/vllm/autoresearch/docker-compose.yml
@@ -52,7 +52,7 @@ services:
       # — re-run #928 Cell C against this pin to confirm scoring parity
       # before treating it as the new canonical autoresearch baseline.
       # See README.md § Model selection for the full rationale.
-      - Qwen/Qwen3-30B-A3B-Instruct-2507
+      - NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4
       # Pin the revision SHA after first verified boot, same dance as
       # coder-next. Until then keep this commented to take the model's
       # current main revision.
```

Apply, commit, push:

```bash
cd ~/agentic-ai-homelab
git checkout -b feat/autoresearch-cell-f-nvfp4-swap
# Edit infra/vllm/autoresearch/docker-compose.yml line 55 (model: line)
git add infra/vllm/autoresearch/docker-compose.yml
git commit -m "feat(autoresearch): swap Qwen3-30B-A3B → NVFP4 (#1022 Cell F daily driver)

NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 — NVIDIA Model Optimizer quant
of the same base model. ~2× faster end-to-end on autoresearch eval at
no measurable quality cost. Wins GI stage outright in the #1016
cohort. 18 GB vs 57 GB weight footprint.

Evidence: chipi/podcast_scraper /
docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md.

For highest-stakes one-shot evals where summary or KG quality matters
more than wall-clock, manually edit this file back to
Qwen/Qwen3.5-35B-A3B-Instruct for that run."
git push -u origin feat/autoresearch-cell-f-nvfp4-swap
# Open the PR in github
```

After merge, bring the new model up on DGX:

```bash
ssh dgx-llm-1
cd ~/agentic-ai-homelab/infra/vllm/autoresearch
git pull origin main
gpu-mode-swap.sh idle && gpu-mode-swap.sh research
# wait ~2 min for /health (NVFP4 boots ~3× faster than bf16)
curl -sf -H "Authorization: Bearer $VLLM_API_KEY" \
  http://localhost:8003/v1/models | jq -r '.data[].id'
# Expect: autoresearch
```

## Rollback

If anything surprises in real use, revert the one line in the homelab repo
and restart the autoresearch slot. Cell F is fully reversible at the
compose level; no podcast_scraper changes need to revert.

## What else to think about (not blocking)

- The previously-cached `Qwen/Qwen3-30B-A3B-Instruct-2507` bf16 stays on
  disk (~57 GB at `/opt/llm-models/huggingface/hub/`). Don't garbage-
  collect it — it's the rollback model and the "highest-stakes one-shot"
  variant.
- Validate that `--reasoning-parser=qwen3` is NOT set in the compose (it
  isn't currently for Qwen3-30B-A3B-Instruct-2507, since that variant
  doesn't think — same applies to its NVFP4 quant).
- `VLLM_GPU_MEM_UTIL=0.65` in `.env` stays the same — it's not a Cell F
  decision (cells A/B/C/D all came back as noise).
