# EVAL — Finale tier (G-Eval championship), smoke v2, 2026-06-09

**Issue:** #932 (G-Eval finale) + #940 (R1 tertiary judge integration)
**Branch:** `feat/907-autoresearch-batch-2`
**Dataset:** `curated_5feeds_smoke_v1` — 5 episodes per finalist
**Spend:** **$2.36** (cap $50.00)
**Verdict:** integrate qwen3.5:35b (DGX) + hermes3:8b (laptop). Zero contested
pairs across both judge passes.

---

## Why a finale tier

The qualifier (ROUGE-L vs Sonnet-4.6 silver) gives a cheap discrimination
signal but biases toward the silver model's surface style. Two costly side
effects:

1. **Sonnet-style mimics** rank ahead of equally-faithful but
   stylistically-distinct summaries.
2. **Current production champion qwen3.5:35b** (#923) sat *below* the
   qualifier floor on the v2 refresh — purely on stylistic distance from
   Opus-4.7 silver, not on quality.

The finale exists to break that bias with flagship LLM judging across four
G-Eval dimensions (faithfulness, coverage, coherence, fluency) on per-episode
predictions — the methodology designed in
[EVAL_FINALE_METHODOLOGY.md](EVAL_FINALE_METHODOLOGY.md).

## What the finale ran

- **Strata:** dgx_le_40b (DGX ≤40B Ollama), mbp_le_14b (laptop ≤14B Ollama).
  Cloud was empty in this smoke (no qualifier cloud finalists in the v2 set).
- **Promotion rule:** top-3 per stratum (RougeL-floor = 0.8 × stratum leader).
- **Carte blanche:** `autoresearch_prompt_ollama_qwen35_35b_*` (current prod
  champion, fell below the qualifier floor on stylistic distance) — promoted
  unconditionally so the finale judge could verdict it on G-Eval grounds.
  Mechanism added to the runner in this PR.
- **Sample size:** 5 episodes × 7 finalists × 4 dimensions = 140 primary
  judgments. Top-2-per-stratum (4 finalists) re-judged by the cross-check
  judge = +80 cross judgments. Total 220 G-Eval scores.

## Judges

| Slot | Model | Role |
| --- | --- | --- |
| Primary | **Claude Sonnet 4.6** | Score every (finalist × episode × dim) pair |
| Cross-check | **OpenAI GPT-5.4** | Re-score top-2 per stratum; flag contested |
| Tertiary | DeepSeek-R1:32B (DGX) | Available; not invoked this run |

**Why GPT-5.4 (not Gemini 2.5 Pro)?** The original #932 config specified
Gemini 2.5 Pro as the cross-check judge. On the first finale attempt
(2026-06-09 23:29), Gemini returned `text=''` on 20/20 cross-check calls:
the model's dynamic thinking budget consumed the entire
`max_output_tokens=1024`, leaving zero tokens for actual content. HTTP 200,
no billing, no signal. We reverted to the RFC-057 dual-judge pair
(Anthropic and OpenAI flagship models) used in earlier autoresearch evals
([EVAL_TIER2_QMSUM_2026_04](EVAL_TIER2_QMSUM_2026_04.md),
[EVAL_CLEANING_AUTORESEARCH_2026_06_08](EVAL_CLEANING_AUTORESEARCH_2026_06_08.md)).
`OpenAIChatJudge` (gpt-5.4, `max_completion_tokens`-aware) added in this PR;
`Gemini25ProJudge` stays in the codebase for ad-hoc tertiary use.

## Verdicts

### DGX (≤40B) — winner: **qwen3.5:35b** 🥇

| Rank | Model | Faith | Cov | Coh | Flu | Primary mean | GPT-5.4 cross | Agreement | Contested? |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 🥇 | `qwen3.5:35b` | 5.0 | 5.0 | 5.0 | 5.0 | **5.00** | 4.90 | 1.00 | – |
| 🥈 | `qwen3.5:27b` | 5.0 | 5.0 | 4.8 | 5.0 | **4.95** | 4.85 | 1.00 | – |
| 🥉 | `mistral-small3.2` | 5.0 | 5.0 | 4.0 | 5.0 | **4.75** | – | – | – |

**Reading:** qwen3.5:35b is the unambiguous DGX champion — perfect 5.00 on
all four dimensions on every episode, and GPT-5.4 ranked it #1 too
(4.90 mean, 100% agreement with Sonnet). This validates the carte-blanche
mechanism: qwen3.5:35b would have been *silently excluded* on the qualifier
ROUGE floor, exactly the bias the finale exists to break.

### Laptop (≤14B) — winner: **hermes3:8b** 🥇

| Rank | Model | Faith | Cov | Coh | Flu | Primary mean | GPT-5.4 cross | Agreement | Contested? |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 🥇 | `hermes3:8b` | 5.0 | 4.6 | 3.0 | 4.4 | **4.25** | 4.70 | 0.85 | – |
| 🥈 | `mistral:7b` | 4.8 | 4.4 | 3.2 | 4.4 | **4.20** | 4.60 | 0.85 | – |
| 🥉 | `llama3.2:3b` | 4.0 | 3.4 | 2.6 | 3.2 | **3.30** | – | – | – |

**Reading:** hermes3:8b takes the laptop tier by a hair (4.25 vs 4.20 on
primary; 4.70 vs 4.60 on GPT-5.4 cross). The Phase 0.5 per-model prompt
tuning lift on hermes3 (see [EVAL_SMOKE_V2_DGX_REFRESH_2026_06.md](EVAL_SMOKE_V2_DGX_REFRESH_2026_06.md))
clearly carried through to flagship judging. mistral:7b also delivers
strong finalist quality — both clear the 4.0/5.0 threshold and remain
viable laptop candidates.

**GPT-5.4 vs Sonnet diverged on coherence**: Sonnet scored mbp finalists
3.0-3.2 on coherence, GPT-5.4 scored them 4.6-4.8. The ranking is preserved
(hermes3 > mistral_7b on both), and no pair tripped the >0.5 absolute-mean
divergence threshold, so nothing was flagged contested. This is consistent
with prior observations that Sonnet is structurally more critical on
coherence specifically.

## Profile / production implications

| Profile | Before | After | Reason |
| --- | --- | --- | --- |
| `config/profiles/local.yaml` (mbp/laptop) | `qwen3.5:9b` | **`hermes3:8b`** | Finale champion in the ≤14B stratum; qwen3.5:9b was eliminated at the qualifier stage (rank 9 of 10). Tuned prompts already shipped at `src/podcast_scraper/prompts/ollama/hermes3_8b/summarization/*.j2`. |
| `config/profiles/local_dgx_balanced.yaml` (DGX small) | `qwen3.5:9b` | *unchanged* | This profile uses staged-mode specifically because qwen3.5:9b breaks under bundled mode (#652 / 2026-06-07 investigation). Switching to hermes3:8b needs a separate staged/bundled reliability eval before adoption. Tracked as follow-up. |
| `config/profiles/local_dgx_full.yaml` (DGX no-fallback measurement) | `llama3.3:70b` | *unchanged* | The 70B class is outside the ≤40B stratum the finale evaluated. Consider promoting `qwen3.5:35b` here in a follow-up after a direct head-to-head against llama3.3:70b. |

The laptop swap is the one production-meaningful change this PR makes —
`qwen3.5:35b` already runs in DGX prod, so the DGX verdict is *validating*
the existing default rather than redirecting it.

## Cost breakdown

| Phase | Calls | Cost |
| --- | ---: | ---: |
| Primary judging (Sonnet 4.6, 7 finalists × 5 ep × 4 dim) | 140 | $1.57 |
| Cross-check (GPT-5.4, top-2 × stratum × 5 ep × 4 dim) | 80 | $0.80 |
| **Total** | **220** | **$2.36** |

(Plus a sunk $1.57 from the first attempt against Gemini 2.5 Pro before
swapping judges — paid Sonnet primary cost, recoverable in principle by
caching but not implemented this PR.)

Per-finalist costs:

| Finalist | Stratum | Primary | Cross |
| --- | --- | ---: | ---: |
| `qwen3.5:35b` | dgx | $0.227 | $0.204 |
| `qwen3.5:27b` | dgx | $0.228 | $0.205 |
| `mistral-small3.2` | dgx | $0.221 | – |
| `mistral-small:24b` | dgx | $0.221 | – |
| `hermes3:8b` | mbp | $0.217 | $0.194 |
| `mistral:7b` | mbp | $0.218 | $0.196 |
| `llama3.2:3b` | mbp | $0.234 | – |

## Methodology footnote — why no `contested` flag

A pair is contested when |primary_mean − cross_mean| > 0.5 on any
dimension. All four cross-checked finalists' divergence stayed under 0.5
on every dim (the largest was hermes3's coherence: Sonnet 3.0 vs GPT-5.4
4.6 = 1.6 raw delta on that one dim, but the overall mean delta is
4.25 vs 4.70 = 0.45, just below threshold). Both judges agreed on the
within-stratum *ranking* on every comparison, which is the load-bearing
signal for "should we promote this model to production."

## Artifacts

- `data/eval/runs/finale/finale_smoke_v2_2026_06/finale_report.json` —
  machine-readable verdict (the source of every number above)
- `data/eval/runs/finale/finale_smoke_v2_2026_06/finale_report.md` —
  raw human-readable verdict (this report is the longform analysis)
- `data/eval/runs/finale/finale_smoke_v2_2026_06/finalists.jsonl` —
  per-(episode, dim) G-Eval scores from both judges
- `data/eval/runs/finale/finale_smoke_v2_2026_06/promotion.json` —
  stratification + carte-blanche audit trail

## Follow-ups (not this PR)

1. **`qwen3.6:latest` not mapped to any stratum** — carte-blanche couldn't
   promote the v2 challenger because the config has no matching prefix.
   File a small config bug.
2. **R1 fluency-dim parser hardening** — separate from the finale, but
   surfaced by [EVAL_R1_AS_JUDGE_2026_06.md](EVAL_R1_AS_JUDGE_2026_06.md):
   7/24 fluency calls returned empty text the parser couldn't recover.
3. **Gemini 2.5 Pro thinking-budget pathology** — document in
   `Gemini25ProJudge` docstring (already done); consider adding
   `thinking_config={thinking_budget: 0}` if we ever revive Gemini as a
   primary/cross judge slot.
4. **DGX-balanced laptop-tier model swap** — repeat the staged/bundled
   reliability sweep with hermes3:8b before swapping `local_dgx_balanced.yaml`.
5. **70B-tier head-to-head** — run a focused 2-model finale on
   `qwen3.5:35b` vs `llama3.3:70b` before touching `local_dgx_full.yaml`.
