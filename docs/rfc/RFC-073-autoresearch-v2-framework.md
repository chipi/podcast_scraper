# RFC-073: Autoresearch v2 Framework — Dev/Held-Out Split, Fixed Judges, Cross-Approach Comparison

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team
- **Supersedes (in part)**: `docs/rfc/RFC-057-autoresearch-optimization-loop.md` (Track A prompt tuning methodology; RFC-057 remains authoritative for Track B ML parameter tuning)
- **Related RFCs**:
  - `docs/rfc/RFC-057-autoresearch-optimization-loop.md` — v1 framework (smoke/benchmark, pre-v2 judging)
  - `docs/rfc/RFC-015-ai-experiment-pipeline.md` — experiment configs and `data/eval/` layout
  - `docs/rfc/RFC-017-prompt-management.md` — Jinja prompt store
- **Related ADRs**:
  - `docs/adr/ADR-073-rfc057-autoresearch-closure.md` — RFC-057 closure record (v1 champions)
- **Related Documents**:
  - `autoresearch/openai_v2_comparison_2026-04-14.md` — first v2 reference card (OpenAI bundled vs non-bundled)
  - `autoresearch/JUDGING.md` — dual-judge system (rubric, contestation, extraction)
  - `autoresearch/README.md` — v2 operational guide

## Abstract

RFC-057 introduced the autoresearch ratchet loop (edit prompts, score, accept/reject) using a smoke (5 episodes) vs benchmark (10 episodes) dataset pair. Practical experience exposed three structural problems with that framework: (1) smoke ⊂ benchmark contaminated validation, (2) binary-OR judge contestation flipped entire runs to ROUGE-only on a single divergent episode, and (3) temperature=0 is not deterministic at the OpenAI API layer, producing 5% run-to-run score noise. Champions tuned under v1 showed overfitting signatures (e.g. bundled bullets dropped 5.4pp ROUGE-L from smoke to benchmark).

This RFC defines the **v2 framework**: a dev/held-out dataset split modeled on ML train/test conventions, fraction-based contestation, an Efficiency rubric dimension replacing Conciseness, deterministic-seed plumbing for OpenAI, and prose extraction before judging for bundled outputs. The framework is proven end-to-end on OpenAI (bundled + non-bundled, bullets + paragraph — all four champions validated on held-out content they were never tuned on) and is the template for evaluating other providers.

## Problem Statement

RFC-057 Track A produced working champions but the framework underneath had three defects that accumulated into untrustworthy numbers:

### Problem 1: smoke ⊂ benchmark

`curated_5feeds_smoke_v1` (5 ep) is a strict subset of `curated_5feeds_benchmark_v1` (10 ep) — same e01 episodes, benchmark adds e02 per feed. Any "benchmark validation" of a smoke-tuned champion was partially retesting on the tuning set. The bundled-bullets smoke champion (ROUGE-L 33.4%) dropped to 28.0% at benchmark scale — a textbook overfitting signature that we couldn't detect cleanly without a disjoint held-out.

### Problem 2: binary-OR contestation was too brittle

Original contestation rule: if *any* of 5 smoke episodes had judges diverging more than 0.25, the entire run fell back to ROUGE-only scoring. At smoke scale (N=5), one episode = 20% of the sample. A single stochastic divergence would flip the final scalar by ~47% (from `0.70*ROUGE + 0.30*judge` to pure ROUGE). Multiple prompt experiments during RFC-057 Round 4/5 were rejected under this rule despite producing real quality improvements; the rejections were judge-variance artifacts, not prompt regressions.

### Problem 3: temperature=0 is not deterministic

OpenAI's `temperature=0` parameter is documented as "approximately deterministic," not guaranteed. Same model, prompt, and seed produce different outputs. Two back-to-back identical calls during v2 diagnosis (model=gpt-4o, temp=0, seed=42, stable system_fingerprint) produced different summaries and final scores 0.478 vs 0.487 — a 1.8% swing from API alone, before judging variance. At smoke scale this noise made marginal prompt improvements indistinguishable from API jitter.

## Goals

1. **Introduce held-out validation** to detect overfitting. All future champions are measured on episodes never used during tuning.
2. **Reduce per-run noise** through seed plumbing (where providers support it) and a 40%-fraction contestation threshold.
3. **Recalibrate the rubric** to stop penalising valid longer-form outputs (Efficiency replaces Conciseness).
4. **Extract JSON prose before judging** so bundled outputs are judged on semantic content, not JSON formatting.
5. **Preserve full reproducibility** of v1 runs. All v1 dataset and silver artifacts remain untouched; v2 adds new artifacts alongside.
6. **Stay thin-harness**: the v2 changes are all small edits to existing code (`autoresearch_track_a.py`, provider, factory, config), plus new data artifacts. No new infrastructure or CLI surface.

## Non-Goals

- Track B (ML parameter sweep) is unchanged; v2 applies only to Track A prompt tuning.
- Multi-run averaging per experiment is not implemented in v2 (deferred — see §Future Work).
- Statistical confidence intervals on champion scores are not implemented (deferred).
- Held-out dataset size is not optimized; 5 episodes is a pragmatic start, not a formal power calculation.

## Design

### Dataset tiering (the train/test split)

| Tier | Role | ML analogue | Dataset ID | Size | Usage |
| ---- | ---- | ----------- | ---------- | ---- | ----- |
| **Dev** | Ratchet iteration; accept/reject decisions use this score | Training set | `curated_5feeds_dev_v1` | 10 ep (e01+e02 per feed) | Every experiment scores here |
| **Held-out** | Champion validation; touched once per committed champion | Test set | `curated_5feeds_benchmark_v2` | 5 ep (e03 per feed, ~32 min each) | Never scored during iteration |

**Why this shape:**

- **Disjoint episodes**: e01/e02 in dev, e03 in held-out. Zero overlap. This is the key invariant — without it, validation is contaminated and the whole exercise becomes theatre.
- **Distributional shift**: held-out episodes are ~3× longer than iteration episodes. A champion that only works on 10-minute podcast clips will fail here. This is the cheapest available form of generalisation testing.
- **Small held-out (N=5) is a pragmatic compromise**. Own noise ~±5%. Big enough to catch major overfitting (>10pp), too small to discriminate similarly-good champions. Grow if needed by sourcing more episodes.

**Preserved artifacts** (do not modify):

- `curated_5feeds_smoke_v1` (5 ep, e01 per feed) — still referenced by v1 runs/baselines.
- `curated_5feeds_benchmark_v1` (10 ep, e01+e02 per feed) — same.
- All `silver_sonnet46_{smoke,benchmark}_{bullets,}_v1` references.

### Silvers for v2

Four new references, all generated with Claude Sonnet 4.6 using the existing shared/anthropic summarisation prompts:

- `silver_sonnet46_dev_v1_bullets` (10 ep)
- `silver_sonnet46_dev_v1_paragraph` (10 ep)
- `silver_sonnet46_benchmark_v2_bullets` (5 ep)
- `silver_sonnet46_benchmark_v2_paragraph` (5 ep)

Each silver is produced via `silver_candidate_sonnet46_<dataset>_<track>.yaml` → `make experiment-run` → `promote_run.py`. The silver methodology is unchanged from v1; only the dataset targets are new.

### Judging changes

**Efficiency replaces Conciseness.** The Conciseness rubric rewarded short outputs. The Efficiency dimension rewards unique-information-per-sentence and explicitly does not penalise length — an episode with 10 distinct topics warrants a longer summary. This unblocks 4-6 paragraph summaries without rubric rejection.

**Fraction-based contestation.** Replaces binary OR with a fraction threshold:

```python
# Old (v1): any single episode contests the run
any_contested = any(abs(ja - jb) > 0.25 for ja, jb in zip(judge_a, judge_b))

# New (v2): 40% of episodes must contest before the run is contested
contested_count = sum(1 for ja, jb in zip(judge_a, judge_b) if abs(ja - jb) > 0.25)
any_contested = (contested_count / len(episodes)) > 0.40
```

Threshold rationale at each dataset size:

| N | Episodes needed to contest | Why |
| - | -------------------------- | --- |
| 5 (dev smoke) | 3 | 60% of sample |
| 10 (dev) | 5 | 50% of sample |
| 5 (held-out) | 3 | 60% of sample |

**JSON prose extraction before judging.** Bundled outputs are stored as JSON strings; v1 judges received the raw JSON blob and sometimes treated JSON formatting characters as length signal. v2 extracts `title + summary + bullets` into plain prose before the judge call. Judges now see the same semantic content regardless of output format.

**Per-dimension judge output.** Judges return `{coverage, accuracy, efficiency, score, notes}`. The `score` (mean of three) is the authoritative value; per-dimension scores are logged at DEBUG for interpretability.

### Seed plumbing

OpenAI's `seed` parameter is "approximately deterministic" and stabilises `system_fingerprint`. Not perfect but measurably reduces contestation flips on borderline configs. Implementation:

- `Config.openai_summary_seed: Optional[int]` (config.py)
- `SummarizationParams.seed: Optional[int]` (providers/params.py)
- Factory wiring: `openai_summary_seed=params.seed` (summarization/factory.py)
- Experiment runner: `openai_summary_seed=params_dict_raw.get("seed")` (scripts/eval/experiment/run_experiment.py)
- Provider: passed to both bundled and non-bundled API calls (openai_provider.py)

All v2 autoresearch configs set `params.seed: 42`.

**Empirically**: seed does not give byte-identical outputs. Two seeded runs of the same config still produce different predictions.jsonl hashes. But the variance is smaller and contestation flips are rarer. Seed is a partial mitigation, not a fix — see Future Work.

### New v2 autoresearch configs

| File | Role |
| ---- | ---- |
| `data/eval/configs/summarization_bullets/autoresearch_prompt_openai_bundled_dev_bullets_v2.yaml` | Bundled bullets ratchet (dev) |
| `data/eval/configs/summarization_bullets/autoresearch_prompt_openai_bundled_benchmark_bullets_v2.yaml` | Bundled bullets held-out validation |
| `data/eval/configs/summarization/autoresearch_prompt_openai_bundled_dev_paragraph_v2.yaml` | Bundled paragraph ratchet (dev) |
| `data/eval/configs/summarization/autoresearch_prompt_openai_bundled_benchmark_paragraph_v2.yaml` | Bundled paragraph held-out validation |
| `data/eval/configs/summarization_bullets/autoresearch_prompt_openai_dev_bullets_v2.yaml` | Non-bundled bullets ratchet (dev) |
| `data/eval/configs/summarization_bullets/autoresearch_prompt_openai_benchmark_bullets_v2.yaml` | Non-bundled bullets held-out validation |
| `data/eval/configs/summarization/autoresearch_prompt_openai_dev_paragraph_v2.yaml` | Non-bundled paragraph ratchet (dev) |
| `data/eval/configs/summarization/autoresearch_prompt_openai_benchmark_paragraph_v2.yaml` | Non-bundled paragraph held-out validation |

All use `gpt-4o` + `seed: 42` + `temperature: 0.0`. Bundled configs set `scoring_output_field: bullets|summary` to extract the relevant field for ROUGE scoring.

## Workflow

### During iteration (dev)

```bash
# Bundled bullets ratchet
make autoresearch-score-bundled \
  CONFIG=data/eval/configs/summarization_bullets/autoresearch_prompt_openai_bundled_dev_bullets_v2.yaml \
  REFERENCE=silver_sonnet46_dev_v1_bullets

# Non-bundled bullets ratchet  (no separate Makefile target; use the orchestrator directly)
AUTORESEARCH_EVAL_N=10 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization_bullets/autoresearch_prompt_openai_dev_bullets_v2.yaml \
  --reference silver_sonnet46_dev_v1_bullets
```

Accept rule: delta ≥ +1% on the target track. For prompts shared between tracks (bundled bullets+paragraph), dual-metric: target ≥ +1% **and** other ≥ −1% (no significant regression).

### After accepting a champion

Run held-out validation **once**:

```bash
AUTORESEARCH_EVAL_N=5 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization_bullets/autoresearch_prompt_openai_benchmark_bullets_v2.yaml \
  --reference silver_sonnet46_benchmark_v2_bullets
```

If held-out holds (roughly within dev ± noise floor), the champion generalises. If held-out collapses, you overfit dev — revert.

**Do not iterate on held-out.** Using held-out scores as a ratchet signal destroys its purpose. Use it once per committed champion.

## Results: OpenAI v2 (first complete application)

Held-out scores (ROUGE-L, final score):

| Approach | Bullets | Paragraph |
| -------- | ------- | --------- |
| Non-bundled (v2 champion) | **39.6% / 0.566** | **31.7% / 0.481** |
| Bundled (v2 champion) | 33.2% / 0.505 | 29.5% / 0.469 |
| Silver (Sonnet 4.6 reference) | 100% / 1.0 | 100% / 1.0 |

Non-bundled wins both tracks at v2 scale. This is a reversal from the v1 narrative — in v1, bundled appeared to win paragraphs — which turned out to be a contamination + judging artifact. See `autoresearch/openai_v2_comparison_2026-04-14.md` for the full reference card.

All four champions generalise cleanly on held-out (no overfitting detected). The v2 framework achieved what it was designed to do: produce numbers we can defend.

## Replication for other providers

To benchmark another provider (Anthropic, Gemini, Mistral, etc.) under v2:

1. **Create 8 configs** mirroring OpenAI's structure (bundled + non-bundled × bullets + paragraph × dev + benchmark). Reuse the existing Sonnet 4.6 silvers — they're provider-neutral reference quality.
2. **Port the OpenAI champion prompts** as a starting point for the provider's prompts. OpenAI's accepted changes (style narration, few-shot examples, anti-patterns for bullets; 4-6 paragraphs + opening sentence + coverage + verbatim for paragraphs) transfer cleanly across models. Use those as the initial templates rather than starting from stock.
3. **Establish baselines on dev, iterate on dev, validate on held-out.** Keep held-out sacred.
4. **Fill in the same scorecard**. Compare across providers using held-out numbers.

## Alternatives Considered

### 1-tier (just more data in one bigger set)

Rejected for this project. With 15 episodes available, a single larger pool has less noise than 10-ep dev but no overfitting detection at all. Since we iterate many times (20+ experiments per track), the human-in-the-loop selection process *is* a form of optimisation against the eval set, and information leaks without a held-out signal. Empirically, our v1 bundled bullets showed exactly the overfitting behaviour that a held-out would have caught.

1-tier would be right if we had 50+ episodes (enough that noise floor makes overfitting detectable without a formal split) or if iteration counts were small (1-2 experiments).

### 3-tier (smoke + dev + held-out)

Rejected. A dedicated smoke tier adds setup complexity without distinct job — a `--sample 2` on dev does the same pipeline check. And with only 15 episodes, splitting into 3 tiers shrinks dev or held-out below useful size.

### Multi-run averaging (N=3 per experiment, reported mean)

Considered; deferred. This is the principled fix for API non-determinism. Cost is 3× per experiment. Worth implementing if v2 seed plumbing proves insufficient over a larger tuning session. Tracked in §Future Work.

### Keeping v1 dataset names, just adding held-out

Rejected on clarity grounds. `benchmark` in v1 semantics was a validation set that was actually contaminated by smoke. Reusing the name would perpetuate the confusion. New v2 names (`dev_v1`, `benchmark_v2`) make the role explicit.

## Future Work

1. **Multi-run averaging**. When per-experiment reliability matters more than iteration speed, wrap the ratchet in N=3 averaging. One-time change in `autoresearch_track_a.py` + `score.py`. Gives trustworthy ratchet scores at the cost of 3× compute.

2. **Confidence intervals on champions**. Report `score ± 1.96 * stderr` rather than point estimates. Would require multi-run infrastructure from (1). Makes "champion X is better than Y" claims statistically defensible.

3. **Grow held-out dataset**. 5 episodes is the weakest link — validation noise ~±5%. Sourcing 5-10 more e04/e05 episodes per feed would tighten the held-out signal without affecting the training-set workflow.

4. **Extend v2 to other providers**. Anthropic is the natural next application — we have Sonnet 4.6 silvers and the OpenAI champion prompts as a strong starting point. After Anthropic, Gemini and Mistral to complete the comparison matrix.

5. **Apply v2 to Track B (ML parameter sweep)**. RFC-057 Track B is closed under v1 methodology. Re-running key ML-parameter champions on v2 dev/held-out would validate they don't overfit either.

6. **Hybrid mode evaluation**. Bundled-for-summary + separate-non-bundled-for-bullets might combine the best of both. Not a design change, just a config / wiring effort. Measure against pure bundled and pure non-bundled on the same v2 framework.

## References

- RFC-057: Original autoresearch loop — defines Track A / Track B and the v1 smoke/benchmark framework.
- ADR-073: RFC-057 closure — documents v1 champion results.
- `autoresearch/openai_v2_comparison_2026-04-14.md` — first complete v2 reference card.
- `autoresearch/JUDGING.md` — detailed judging system documentation.
- Session commits on `feat/eval-benchmark-v2`: `3b10eb6` (v2 framework scaffold), `71419a7` (non-bundled bullets champion), `798991f` (non-bundled paragraph champion), `dca85d5` (comparison doc).
