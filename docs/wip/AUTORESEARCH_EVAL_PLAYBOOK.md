# Autoresearch LLM Landscape Eval — Step-0 Playbook

**Purpose**: when running the next "compare N candidate LLMs across our 3 stages"
project (similar to #1016), this is the checklist that captures everything we
learned the hard way during #1016. Treat it as Step 0 to skip re-discovering
the same harness gaps + methodology pitfalls.

Last updated: 2026-06-17, distilled from #1016 (Qwen / Mistral / DeepSeek /
Llama / Moonshot / Google / NVIDIA cohort).

---

## Step 0a — HF documentation pass for EVERY candidate (before any code)

For each candidate in the cohort, before downloading weights, capture from
the HF model card / GitHub / arxiv:

- [ ] **Total params + active params** (for MoE) — sizes the GB10 fit math
- [ ] **Quantization** (BF16 / FP8 / NVFP4 / etc.) — multiplies model size
- [ ] **Vendor-recommended sampling**: temperature, top_p, top_k, min_p,
      presence_penalty (verbatim from `generation_config.json` or README)
- [ ] **Reasoning support** — is this a thinking-by-default model?
      - If yes: vLLM `--reasoning-parser=<name>` flag is REQUIRED
        (Qwen3/3.5/3-Next → `qwen3`; Mistral reasoning models like Magistral
        → `mistral`; DeepSeek-R1 family → `deepseek_r1`)
      - chat_template_kwarg overrides (`enable_thinking: false` for Qwen3
        family, etc.)
- [ ] **Tokenizer mode** — Mistral family requires 4 server flags:
      `--tokenizer_mode=mistral --config_format=mistral --load_format=mistral
      --tool-call-parser=mistral`
- [ ] **trust_remote_code** — required for: Kimi, Moonlight, DSV2, DSR1
- [ ] **`max_position_embeddings`** — hard ceiling on `--max-model-len`.
      Moonlight is 8192; Qwen3.5 is 262144. Set compose flag to match the
      smaller of (model ceiling, our needs).
- [ ] **Multimodal token rules** — Gemma 4 needs `--max-num-batched-tokens=4096`
      because `max_tokens_per_mm_item=2496` > default 2048. Other vision
      models likely have similar requirements.
- [ ] **`SYSTEM_PROMPT.txt` availability** — Mistral family ships one in
      their HF repo; should be loaded into the system prompt slot.
- [ ] **Known refusal/CAI patterns** — Kimi-Linear hit constitutional-AI
      refusals on strict negative-constraint prompts. Document any
      vendor mention of refusal training.

**Time budget**: ~2-3 hours for 10-15 candidates. **Saves**: 20+ hours of
later "why does this candidate fail" debugging.

**Output**: per-candidate row in MODEL_PLAYBOOK.md `## Vendor-recommended
sampling` table.

---

## Step 0b — Memory + boot fit check before any download

**GB10 unified mem budget** (128 GiB total):

- Weights cost (from HF disk size) — multiplies by quant tier
- Activations + CUDA graph + non-torch overhead: ~25-30 GiB minimum
- Effective ceiling: **~95-100 GiB BF16 weights fit**; > 90 GiB do NOT fit
- NVFP4 / FP8 quant pulls weight cost down ~2-4x

Quick fit table:

| Model class | Weight footprint | Fits? |
|---|---:|---|
| ≤30B BF16 (Qwen3-30B, Mistral-Small-3.2) | ~60 GB | ✅ comfortable |
| 35B MoE BF16 (Qwen3.5-35B-A3B) | ~67 GB | ✅ comfortable |
| 26B multimodal BF16 (Gemma 4-A4B) | ~51 GB | ✅ comfortable |
| 70B NVFP4 (Llama-3.3-70B-NVFP4) | ~36 GB | ✅ comfortable (but slow throughput) |
| 119B NVFP4 (Mistral-Small-4-119B-NVFP4) | ~70 GB | ⚠️ tight — needs `--max-num-seqs=4 --max-model-len=8192 --gpu-mem=0.70` |
| 48B BF16 (Kimi-Linear) | **91.5 GB** | ❌ does NOT fit, no FP8/NVFP4 variant exists |

**Decision rule**: BF16 > 80 GB is at-risk; > 90 GB don't try unless a
quant variant exists. Drop or substitute before sinking download bandwidth.

---

## Step 0c — Boot-flow harness contract

For each candidate's first boot:

1. Hand-edit `~/agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml`
   - Set model name
   - Add all per-vendor flags from Step 0a (reasoning-parser, tokenizer flags, etc.)
2. `gpu-mode-swap.sh idle && gpu-mode-swap.sh research`
3. Monitor `/health` (background command with curl loop until 200)
4. When `/health` flips 200: run `onboard_model_smoke.py` IMMEDIATELY
   - 5-token hello probe
   - 1-episode summary with the planned config
   - Verify: no BPE artifacts, no `<think>` leak, no refusal, no echo,
     finish_reason=stop, length in spec
5. Smoke PASS → kick the full 10-ep cohort run
6. Smoke FAIL → diagnose BEFORE burning 10-episode inference budget

**Compose defaults applied as of 2026-06-17** (lever A + C from #1016 §5):

- `--max-num-seqs=4` (down from 64; KV peak was 2.5% even at 64)
- `--enforce-eager` (saves 2-4 min boot at ~10-15% inference perf cost)

For prod-mirror benchmark runs (not eval-loop iteration): remove
`--enforce-eager`, bump `--max-num-seqs=64`.

---

## Step 0d — Round structure that worked

| Round | Purpose | Output |
|---|---|---|
| **Phase 2a — Candidate addition** | Add each candidate to the cohort one at a time, verify it boots + produces output | Per-candidate config + smoke pass |
| **Phase 2b — Summary scoring** | Score every candidate's summary against silvers (Opus 4.7 + Sonnet 4.6) using gate criteria | 4-criteria gate (mean<100s, p99<150s, cv<0.3, chars 800-3200) |
| **Phase 2c — GI + KG scoring** | Same cohort, but extraction tasks | Per-stage coverage rate vs silvers |
| **G-Eval judging (cloud)** | LLM-judge cross-vendor scoring on Phase 2b/2c predictions | Quality signal complementing ROUGE/BLEU/cosine |

**Drop rule**: candidates >60s/ep summary speed are dropped from Phase 2c
(they cost 2x for marginal cohort-level value). Their R1/2 GI/KG data
stands as their final landscape numbers.

---

## Step 0e — Sampling methodology

The single biggest #1016 finding:

### Vendor sampling helps generative tasks, hurts extraction tasks

- **Summary (generative)**: vendor sampling **improves** semantic alignment
  (+2.6 cosine for Qwen3.5). The output diversity matches the silver style
  better than greedy collapse.
- **GI + KG (extraction)**: vendor sampling **hurts** structured-output
  stability (−8 to −9 coverage for Qwen3.5). Higher temp jitters the
  insight/topic list.

### Sonnet-mimicry detection (dual-silver methodology)

- Always score against TWO silvers from disjoint vendors (Opus + Sonnet)
- `Δ = (Sonnet-score) − (Opus-score)` reveals style-mimicry
- `|Δ| < 1pt` = neutral; `Δ ≥ +3pt` = clear Sonnet-mimicry
- **Vendor sampling AMPLIFIES mimicry signal that greedy hides**
- **Style is TASK-DEPENDENT**, not just model-dependent. Same model can
  show Sonnet-lean on summary and Opus-lean on KG (Qwen3-30B-Instruct-2507).
  Check the delta on the stage being ranked.

### Cohort-comparison vs per-stage-optimization

Choose ONE methodology and document it before starting:

- **Cohort comparison** (apples-to-apples): use the same sampling for all
  candidates so the comparison is valid. Accept that no candidate is at its
  individual optimal. #1016 chose this path.
- **Per-stage optimization**: each candidate gets its individual optimal
  sampling per stage. Better headline numbers but harder to compare
  cross-candidate.

---

## Step 0f — Anti-patterns to avoid (we hit these in #1016)

1. **Don't disable a project-wide lint rule to avoid fixing violations**
   ([[feedback-no-lint-check-weakening]]). Just fix the violations.
2. **Don't claim a model is dropped after 1-2 attempts** — the operator
   decides drops; agent role is run / observe / report.
3. **Don't sudo-edit on the DGX without explicit operator authorization**
   ([[feedback-never-overstep-sudo-or-host-state]]). The autoresearch
   compose at `~/agentic-ai-homelab/infra/vllm/autoresearch/` is editable
   without sudo; `/opt/vllm-autoresearch/` is stale and off-limits.
4. **Don't try to tune R1-family reasoning models for summary** — they
   produce 4096+ token outputs by design. Use the chat variant from the
   same vendor (DSV2-Lite for DeepSeek). Drop the reasoning model.
5. **Don't run Phase 2c GI/KG without first verifying the postprocessor
   reaches the GI/KG node-label path** (the bug that gave DSV2-Lite 0%
   in #1016 — fixed as task #111).
6. **Don't use a single judge vendor when Sonnet-mimicry candidates are
   in the cohort** — at minimum Sonnet + GPT-5.4 + maybe Gemini 2.5 Pro.
   Validate that judge agreement holds; if it doesn't, the mimicry candidate
   is over-ranked.
7. **Don't skip the onboarding smoke test for "we ran this candidate
   before"** — model versions change, tokenizer versions change, prompt
   templates change. Always smoke first.
8. **Don't trust the metrics poller log without verifying it captures real
   data** — the original #1016 metrics script wrote timestamps but the
   metric prefix names didn't match vLLM 0.20.1+ output. Real metrics:
   `vllm:kv_cache_usage_perc`, `vllm:generation_tokens_total`,
   `vllm:time_to_first_token_seconds_bucket`,
   `vllm:request_time_per_output_token_seconds_bucket`.

---

## Step 0g — Tooling references (paths under this repo)

- **Smoke test**: `scripts/eval/onboard_model_smoke.py` — 13-check probe
- **Metrics poller**: `scripts/eval/poll_vllm_metrics.py` — 5s interval,
  writes to `docs/wip/EVAL_1016_metrics/vllm_metrics_<candidate>_<phase>.log`
- **Phase 2c sweep runner**: `scripts/eval/phase2c_sweep.py` — needs
  candidate registry update for the new cohort
- **GI scorer**: `scripts/eval/score/score_gi_insight_to_insight.py`
- **KG scorer**: `scripts/eval/score/score_kg_node_to_node.py`
- **Postprocessor registry**: `src/podcast_scraper/evaluation/output_postprocess.py`
  (decode_r1_byte_level, strip_r1_reasoning_and_decode, noop, …)

---

## Step 0h — Known harness gaps still open

These should be fixed BEFORE the next autoresearch cohort to avoid re-hitting
the same blockers:

| Gap | Task | Why fix before next cohort |
|---|---|---|
| `top_p`/`top_k`/`presence_penalty` silently dropped from `params:` block (SummarizationParams doesn't define them) | #108 | Every Round 1 config that set non-default sampling was silently broken. Workaround was routing via `backend.extra_body`; proper fix should land. |
| Mistral tokenizer flags + SYSTEM_PROMPT.txt fetch | #109 | Manual compose-edit each swap; fragile. |
| BPE postprocessor not applied to GI/KG node labels (DSV2 0% bug) | #111 | Any model with byte-level BPE quirks will silently get 0% on extraction. |
| Cohort-uniform prompt doesn't request entity extraction (5/7 candidates emit 0 entities) | #112 | KG coverage is artificially low; per-stage routing decisions may be misled. |
| `.markdownlint.json` audit (5 rules disabled — may be accumulated agent cheats) | #1028 | Doc quality has been silently degrading. |

---

## Step 0i — When to run the Step-0 checklist

- Adding ≥3 new candidates to an existing cohort
- Starting a new task-domain comparison (not just summary — e.g. translation,
  classification, code-gen)
- New vLLM major version (0.20 → 0.21 may change metric names, available flags)
- New DGX or hardware change (different SM version, different unified mem budget)

For incremental work (1-2 new candidate additions to an existing cohort),
this checklist is overkill — just apply the relevant per-candidate Step 0a/0b/0c.
