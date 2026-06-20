# Standalone briefing — for a Claude agent in `chipi/agentic-ai-homelab`

**Paste-ready** for a fresh agent session that opens in the
`chipi/agentic-ai-homelab` repo. No prior context needed.

---

## Task

Ship a single-line model swap to the autoresearch vLLM compose, then
push to a new branch and open a PR. The change is **already applied
uncommitted on the DGX** (`dgx-llm-1`), validated end-to-end, and
serving live. Your job is to commit + push the existing diff cleanly.

## Context (you don't need to verify — done already)

- Issue tracked at `chipi/podcast_scraper#1022` (closed 2026-06-19).
- The new model `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` is NVIDIA
  Model Optimizer's official NVFP4 quant of the existing baseline
  `Qwen/Qwen3-30B-A3B-Instruct-2507`. Same architecture, same
  prompt convention, same sampling defaults — drop-in replacement.
- Validation evidence: 380 s / 10 ep on dev_v1 (vs 728 s baseline,
  -47.8%); held-out validated on `curated_5feeds_benchmark_v2` with
  Sonnet 4.6 cross-vendor silver. No measurable quality regression
  on summary cosine, GI coverage, or KG topic coverage. **Wins the
  GI stage outright across the #1016 Round 3 cohort.**
- 18 GB weight footprint vs 57 GB bf16 (3.2× smaller). NVFP4 boots
  in ~2 min vs ~6 min bf16.
- Full evidence chain lives in the podcast_scraper repo at
  `docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md` — read if you
  want, NOT required for this task.

## Operator rules you must respect

Read these BEFORE doing anything:

- **NEVER `--no-verify`**, `--no-gpg-sign`, or any other hook bypass
  unless the operator explicitly authorizes that specific bypass.
- **NEVER push without explicit operator authorization** — the
  operator has already authorized the push for this specific PR.
  See "Authorization given" below. For everything else: pause + ask.
- **Default PRs to ready, not draft.**
- **No force-push to main**, ever.
- This change ONLY touches `infra/vllm/autoresearch/docker-compose.yml`
  — nothing else. Do not pull in unrelated edits or "while I'm here"
  cleanups.

## Authorization given

For THIS specific operation:

- ✓ Commit the existing uncommitted diff on DGX
- ✓ Push to a new feature branch
- ✓ Open a ready PR with the body below
- ✗ Anything else (no other repos, no other files, no infrastructure changes)

## Step-by-step

```bash
# 1. SSH to the DGX where the diff is sitting uncommitted
ssh dgx-llm-1
cd ~/agentic-ai-homelab

# 2. Inspect what's there (sanity check before committing)
git status                       # expect: infra/vllm/autoresearch/docker-compose.yml modified
git diff infra/vllm/autoresearch/docker-compose.yml

# Expected diff (one line in the model: command block):
# -      - Qwen/Qwen3-30B-A3B-Instruct-2507
# +      - NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4

# 3. Confirm there's nothing else uncommitted you don't expect
git status --short
# If you see ANY other file modified, STOP and report to operator.

# 4. Branch + commit
git checkout -b feat/autoresearch-cell-f-nvfp4-swap
git add infra/vllm/autoresearch/docker-compose.yml
git commit -m "$(cat <<'EOF'
feat(autoresearch): swap Qwen3-30B-A3B → NVFP4 (#1022 Cell F daily driver)

NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 — NVIDIA Model Optimizer's
official NVFP4 quant of the existing baseline. Same architecture,
same prompt convention, same sampling defaults. Validated end-to-end:

- ~2× faster end-to-end on autoresearch eval (380 s / 10 ep on dev_v1
  vs 728 s baseline, -47.8%)
- No measurable quality regression on summary embedding cosine, GI
  coverage, or KG topic coverage (held-out validated on benchmark_v2
  with cross-vendor Sonnet 4.6 silver)
- Wins the #1016 Round 3 cohort GI stage outright (cov 0.425 vs
  Gemma-4 0.413 vs Qwen3.5-35B-A3B 0.363)
- 18 GB weight footprint vs 57 GB bf16 — frees ~50 GB unified memory
  for KV cache / future co-residence
- Boots in ~2 min vs ~6 min for bf16

Evidence chain in chipi/podcast_scraper:
docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md.

For highest-stakes one-shot evals where summary or KG quality
matters more than wall-clock, manually edit this file back to
Qwen/Qwen3.5-35B-A3B-Instruct for that run (rollback is a
single-line diff in this file).

Refs chipi/podcast_scraper#1022.
EOF
)"

# 5. Push and open PR
git push -u origin feat/autoresearch-cell-f-nvfp4-swap

gh pr create --title "feat(autoresearch): swap Qwen3-30B-A3B → NVFP4 (#1022 Cell F daily driver)" --body "$(cat <<'EOF'
## Summary

One-line model swap in the autoresearch vLLM compose:
\`Qwen/Qwen3-30B-A3B-Instruct-2507\` → \`NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4\`.

Drop-in replacement — same model family, same prompts, same sampling.
NVIDIA Model Optimizer's official NVFP4 quant of the existing baseline.

## Why

\`chipi/podcast_scraper#1022\` validated this swap as the \"daily-driver\"
champion for the autoresearch tier. Verified across two datasets and two
cross-vendor silvers:

| Dimension | Baseline (bf16) | This PR (NVFP4) | Δ |
|---|---|---|---|
| End-to-end speed (dev_v1, 10 ep) | 728 s | 380 s | -47.8% |
| Summary embedding cosine vs Opus 4.7 | 0.7948–0.8069 | 0.8011 | ±0% |
| GI coverage vs Opus 4.7 | 0.595 | 0.611 | +2.8% |
| KG topic coverage vs Opus 4.7 | 0.425 | 0.408 | -4.0% |
| Weight footprint | 57 GB | 18 GB | -68% |
| Boot time | ~6 min | ~2 min | -67% |

Also wins the #1016 Round 3 cohort GI stage outright (cov 0.425 vs
prior winner Gemma-4 at 0.413).

## Test plan

- [ ] Verify \`docker compose config\` validates after pull on DGX
- [ ] \`gpu-mode-swap.sh idle && gpu-mode-swap.sh research\` succeeds
- [ ] \`/health\` returns 200 within ~2 min
- [ ] \`/v1/models\` returns \`autoresearch\` as the served name
- [ ] Smoke a one-shot completion via curl

The DGX has been serving this exact model since 2026-06-18; the
above are sanity checks for the post-merge restart cycle.

## Rollback

If anything surprises, revert this single-line diff and restart the
autoresearch slot. Cell F is fully reversible at the compose level;
no consumer-side changes are needed (podcast_scraper profiles use
the \`autoresearch\` served-model-name alias).

Refs: chipi/podcast_scraper#1022.
EOF
)"
```

## After merging in homelab

The operator will run on DGX:

```bash
cd ~/agentic-ai-homelab && git pull origin main
~/agentic-ai-homelab/infra/dgx/bin/gpu-mode-swap.sh idle
~/agentic-ai-homelab/infra/dgx/bin/gpu-mode-swap.sh research
```

…to formalize the running model against the committed compose. The vLLM
is already serving NVFP4 from the uncommitted state; this is the
"adopt the official source of truth" step.

## What to do if something is unexpected

1. **`git status` shows other uncommitted files**: STOP. Report to
   operator what's modified. Do NOT include them in the commit. The
   autoresearch effort only touched `infra/vllm/autoresearch/docker-compose.yml`.
2. **The diff isn't exactly the single-line model swap**: STOP. Show
   the operator the actual diff and ask. The expected diff is shown
   in step 2 above.
3. **`git push` requires force**: STOP and report. Don't force-push.
4. **The PR body needs `gh` CLI authentication**: that's an operator
   action; use `gh auth status` to check, and pause for them if needed.

## Out of scope (DO NOT touch)

- Any model OTHER than the autoresearch slot.
- The `coder-next` vLLM compose — that's the operator's IDE backend
  and is OFF-LIMITS per the project's memory rules.
- The `gpu-mode-swap.sh` script itself — no edits needed.
- `.env` files (the `VLLM_GPU_MEM_UTIL=0.65` value is correct and
  unchanged).
- Anything in `/opt/` on the DGX (uses sudo; out of scope).

That's it. Single-line compose change, one PR, done.
