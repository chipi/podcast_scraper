# Session briefing — 2026-06-18 / 2026-06-19

For the operator returning to the desk. Tells you exactly what landed,
what's parked, and what's blocking.

## Branch state

- Local branch: `feat/autoresearch-followups-2026-06-18`
- 3 commits ahead of `origin/main`, NOT pushed (per
  `feedback_never_push_early.md`). Operator chooses when to push.

```text
6d92d14b fix(providers): #1023 — bypass cloud_structured floor in plain-text summarize() for 5 cloud providers
5427587f feat(autoresearch): #1022 Cell F NVFP4 daily-driver champion + cells A-D negative result
c16856ff feat(eval): #912 validation — 140 trials, 2 hosts, 0 parse failures (closing as no-repro)
```

## Closed this session (3 GH issues)

### #912 — qwen3.5:9b bundled JSON reliability

Closed as no-longer-reproducible at the issue's reported 50–67% rate.
140 A/B trials across two hosts (MBP + DGX) + two Ollama versions
(0.19.0 + 0.30.5) returned 0 parse failures. P(0 fails | true rate
50%) ≈ 10⁻⁴³. Path A's reported 20× latency tax also did not
reproduce (1.03–1.07×). Validation script + raw trials preserved at
`autoresearch/912_validation/`.

### #1022 — vLLM-on-GB10 tuning

Cells A–D (runtime knobs) all came back as noise within the ±5% noise
floor on single-stream Qwen3-30B-A3B eval. The issue's premise (10–30%
runtime-knob headroom) doesn't manifest at this workload.

Cell F (model swap to `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`, NVIDIA
Model Optimizer official quant) delivered ~2× speedup at no measurable
quality cost. Held-out validation on `curated_5feeds_benchmark_v2`
(Sonnet 4.6 cross-vendor silver) confirmed cross-dataset robustness.

**Cross-cohort apples-to-apples** (vs #1016 Round 3 cohort):
- New **GI stage winner** (cov 0.4250 > Gemma-4 0.4125 > Qwen3.5-35B-A3B 0.3625)
- Cohort end-to-end speed leader (38 s/ep vs 64.9 for Qwen3.5-35B-A3B)
- Loses summary rouge1 by 9.8%, KG by 16% vs Qwen3.5-35B-A3B — bounded

**Single-model daily-driver decision**: Qwen3-30B-A3B-NVFP4 (Cell F)
replaces Moonlight in `prod_dgx_balanced` + `eval_default`, replaces
Qwen3.5-35B-A3B in `prod_dgx_full_with_fallback`. Qwen3.5-35B-A3B
retained as the highest-quality reserve for manual one-shot swaps.

Full evidence: `docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md`.

### #1023 — cloud-structured floor audit

Original openai_provider patch + 4 unit tests were already in #1023.
This session completed the 5-provider follow-up audit:

- gemini_provider, deepseek_provider, grok_provider, anthropic_provider,
  mistral_provider — all 5 had the same plain-text `summarize()` bug;
  all 5 patched with `structured=False` argument.
- Verified that bundled methods (`summarize_mega_bundled`,
  `summarize_extraction_bundled`, `summarize_bundled`) correctly keep
  `structured=True` and the floor still applies there.
- 132 tests pass (all integration LLM provider tests + cloud
  structured unit tests). No regressions.

## Operator follow-up actions queued

### 1. Homelab compose model swap

Paste-ready patch at `autoresearch/1022_gb10_tuning/HOMELAB_PR_PATCH.md`.
One-line model swap in `chipi/agentic-ai-homelab` →
`infra/vllm/autoresearch/docker-compose.yml`:

```diff
-      - Qwen/Qwen3-30B-A3B-Instruct-2507
+      - NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4
```

After merge, on DGX:

```bash
ssh dgx-llm-1
cd ~/agentic-ai-homelab/infra/vllm/autoresearch
git pull origin main
gpu-mode-swap.sh idle && gpu-mode-swap.sh research
```

NVFP4 boots in ~2 min (4 shards × ~30 s) vs ~6 min for bf16 (16 shards).

### 2. Decide on Task #115 — local_dgx_balanced workaround revert

`config/profiles/local_dgx_balanced.yaml` still on
`llm_pipeline_mode: staged` from commit c6a8982b (#912 workaround). With
#912 closed as no-repro, the workaround is no longer required. Flip
back to `bundled` or keep `staged` — operator config decision.

### 3. Push this branch + open PR

Per `feedback_never_push_early.md` I didn't push. Run:

```bash
git push -u origin feat/autoresearch-followups-2026-06-18
gh pr create --title "feat(autoresearch): close #912 + #1022 + #1023 — Cell F NVFP4 daily-driver champion" --body "..."
```

Recommended PR scope: all 3 closed issues in one PR (they share the
same evidence chain and are interlocked via the #1016 final-report
addendum). Per `feedback_pr_open_ready_not_draft.md`: default to ready,
not draft.

## Parked tasks (not blocked, just out of scope this session)

| Task | Why parked | Effort estimate |
|---|---|---|
| #108 | Add top_p / response_format to SummarizationParams — small but touches all 6 providers + needs operator alignment on API surface | ~1-2h |
| #109 | Wire mistral tokenizer flags + load SYSTEM_PROMPT.txt — needs operator's mistral spec validation | ~1-2h |
| #111 | Fix BPE postprocessor for GI/KG node labels (DSV2 0% bug) — concrete bug; needs DSV2 re-run after fix | ~1h fix + GPU time |
| #112 | Entity-focused KG re-experiment — operator-driven GPU time | ~2h GPU |
| #113 | Small-model standoff — large task, NVFP4 variants don't exist for Moonlight/Qwen3.5-35B-A3B (verified this session) | ~3-4h |
| #114 | Path D — bundled parse-failure counter in autoresearch eval framework | ~3-4h, touches run_experiment.py |
| #115 | Revert local_dgx_balanced workaround — config decision (above) | 1-line + operator decision |

## Tasks blocked

| Task | Why blocked | What unblocks it |
|---|---|---|
| #970 | Qwen3.6 bf16 — Ollama MoE bf16 + HF symlink + Modelfile sandboxing all gap | Wait for Ollama 0.31+ or operator approval to fork |
| #1002 | Guardrail thresholds — measurement-driven, needs 2–4 weeks production data | Wait for prod metrics |

## State of the DGX

After #1022 close-out:

- `autoresearch` compose REVERTED to committed `main` state (no
  uncommitted diff). Previous session's DeepSeek-V2-Lite-Chat config
  is preserved in `git stash` as `wip_pre_1022_baseline_deepseek_v2_lite`.
- `.env` has `VLLM_GPU_MEM_UTIL=0.65` (unchanged — Cell A's 0.85
  override was reverted).
- `autoresearch` slot **currently up serving Qwen3-30B-A3B-NVFP4**
  (from the Cell F validation runs). When the homelab PR ships, the
  compose will officially adopt this model; until then, this is the
  "ad-hoc" state from validation.

## Reference: files changed across the 3 closed issues

```text
autoresearch/912_validation/     — 8 files (validation script + 3 trial logs + 3 phase logs)
autoresearch/1022_gb10_tuning/   — 22 files (helper script + 6 phase logs + 7 compose snapshots + 3 held-out configs + runs.tsv + HOMELAB_PR_PATCH.md)
docs/wip/JSON-RELIABILITY-DEEP-RESEARCH-2026-06-18.md  — full #912 research chain
docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md     — full #1022 evidence
docs/wip/EVAL_1016_FINAL_REPORT_2026_06_17.md          — Cell F addendum
src/podcast_scraper/providers/{openai,gemini,deepseek,grok,anthropic,mistral}/*_provider.py  — #1023 5-provider patches
src/podcast_scraper/providers/ml/model_registry.py     — new StageOption + 3 ProfilePreset summary swaps
config/profiles/{eval_default,prod_dgx_balanced,prod_dgx_full_with_fallback}.yaml  — Cell F UPDATE notes
```

End of briefing. Ready for review.
