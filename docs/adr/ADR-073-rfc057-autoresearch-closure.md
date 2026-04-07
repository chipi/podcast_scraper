# ADR-073: RFC-057 Autoresearch Loop — Closure and Final State

- **Status**: Accepted
- **Date**: 2026-04-06
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md)
- **See Also**: [ADR-067](ADR-067-pegasus-led-retirement-podcast-content.md),
  [ADR-068](ADR-068-bart-led-as-ml-production-baseline.md),
  [ADR-069](ADR-069-hybrid-ml-pipeline-as-production-direction.md),
  [ADR-070](ADR-070-bart-base-as-hybrid-map-stage.md),
  [ADR-071](ADR-071-four-tier-summarization-strategy.md),
  [ADR-072](ADR-072-llama32-3b-as-tier3-local-llm.md)

## Context & Problem Statement

RFC-057 defined an autoresearch optimization loop with two parallel tracks:

- **Track A** — Prompt tuning: edit Jinja templates, score with `make autoresearch-score`,
  ratchet if scalar composite score improves ≥1% relative; early stop after 3 consecutive
  failures.
- **Track B** — ML inference parameter sweep: greedy one-parameter-at-a-time sweep on
  the hybrid ML pipeline; same ratchet rules.

This ADR documents what was decided, what was promoted to production, and closes the RFC
as complete. It also records the final state of the eval infrastructure that was built
or significantly improved as part of this work.

---

## Track B — ML Parameter Sweep (completed 2026-04-04)

### What was swept

Two pipeline families were swept in sequence:

1. **BART+LED** (pure HuggingFace): map `max_new_tokens`, reduce `max_new_tokens`,
   `num_beams`, `no_repeat_ngram_size`, `repetition_penalty`.
2. **Hybrid BART+Llama 3.2:3b** (HuggingFace MAP + Ollama REDUCE): `temperature`,
   `top_p`, `max_tokens`, `frequency_penalty`, map `num_beams`, map `max_new_tokens`.

### Champions promoted

| Mode | Config ID | ROUGE-L (benchmark) | Embed | Lat/ep |
| :--- | :--- | ---: | ---: | ---: |
| Tier 2 ML Prod | `ml_bart_led_autoresearch_v1` | 20.5% | 68.2% | 26s |
| Hybrid (archived) | `ml_hybrid_bart_llama32_3b_autoresearch_v1` | 21.1% | 76.6% | 15s |

`ml_bart_led_autoresearch_v1` is set as `PROD_DEFAULT_SUMMARY_MODE_ID` in
`config_constants.py`. The hybrid is retained in the registry but not the production
default — direct Llama outperforms it when Ollama is available.

### Key findings

- **temperature=0.5** was the single largest lever for the hybrid (+10% ROUGE-L vs
  hardcoded 0.3). BART chunk noise benefits from more diversity.
- **top_p=1.0** — no nucleus filtering; BART-extracted chunks are clean enough that
  filtering hurts.
- **frequency_penalty=0** — any penalty hurt BART chunk diversity downstream.
- **Instruction-following > size** — llama3.2:3b (3B) beat all tested 7-12B models on
  the hybrid REDUCE step.
- **Hybrid variance at benchmark scale** — temperature=0.5 sampling noise averages out
  over 10 episodes; hybrid advantage over ML-prod shrinks from +4.9pp (smoke) to +0.6pp
  (benchmark). Direct Llama at temp=0.3 is the more reliable Tier 3 choice.

---

## Track A — Prompt Tuning (completed 2026-04-05)

### Scope

Six cloud providers tuned on the paragraph track; Anthropic Haiku then tuned on the
bullets track. All tuning used smoke dataset (5 eps) for speed; wins verified at
benchmark scale (10 eps).

### Results summary

| Track | Provider | Start score | Final score | Key wins |
| :--- | :--- | ---: | ---: | :--- |
| Paragraph | Anthropic | 0.287 | 0.523 | Thesis opener (+0.201), vocab alignment (+0.024) |
| Paragraph | Gemini | 0.446 | 0.475 | Thesis opener + vocab + anchor |
| Paragraph | Grok | 0.437 | 0.456 | Thesis opener + vocab + anchor |
| Paragraph | DeepSeek | 0.502 | 0.502 | Saturated — no wins |
| Paragraph | Mistral | 0.468 | 0.480 | Cause-effect relationships |
| Paragraph | OpenAI | 0.474 | 0.474 | Saturated — judge divergence on all candidates |
| Bullets | Anthropic | 0.546 | 0.599 | No-fence JSON constraint (+3.9%), single-sentence bullets (+5.7%) |

### Template changes accepted (shared across all providers)

Both changes are in
`src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2`:

1. **Explicit JSON boundary constraint** — `"Your response must start with { and end
   with }. No markdown, no code fences, no prose."` Fixed Haiku code-fence wrapping
   that dropped ROUGE-L from 40.1% to 37.1%.
2. **Single-sentence bullet rule** — `"Write each bullet as a single, complete
   sentence; do not split a bullet into two sentences."` Structural alignment with
   the silver reference style (+5.7% for Anthropic bullets).

### Ceiling assessment

All tracks reached early stop (3 consecutive fails). Remaining quality gap vs the
silver reference is structural (reference model writes at a different density and
vocabulary register) — not addressable by prompt instruction. A new silver reference
from a better frontier model would shift the ceiling.

---

## Eval infrastructure built during RFC-057

The following infrastructure was created or significantly improved as part of this RFC
and is now the standard eval process:

### Silver references (paragraph + bullets × smoke + benchmark)

| Reference ID | Model | Dataset | Format | Status |
| :--- | :--- | :--- | :--- | :--- |
| `silver_sonnet46_smoke_v1` | Claude Sonnet 4.6 | smoke (5 eps) | prose | Active |
| `silver_sonnet46_benchmark_v1` | Claude Sonnet 4.6 | benchmark (10 eps) | prose | Active |
| `silver_sonnet46_smoke_bullets_v1` | Claude Sonnet 4.6 | smoke (5 eps) | JSON bullets | Active |
| `silver_sonnet46_benchmark_bullets_v1` | Claude Sonnet 4.6 | benchmark (10 eps) | JSON bullets | Active |

Selected via pairwise LLM judge (dual OpenAI + Anthropic judges): Sonnet 4.6 beat
GPT-5.4 (3-1-1) and swept Gemini 2.0 Flash (5-0) on the smoke dataset.

`silver_gpt4o_*` references are archived — retained for historical traceability.

### 2×2 eval matrix

Every provider now has 4 configs: smoke/benchmark × paragraph/bullets. The full
matrix (6 cloud + 12 Ollama = 18 providers × 4 = 72 configs) is documented in
`data/eval/configs/README.md` with explicit silver reference pairing rules and
trigger conditions.

### Benchmark-scale runs and report

The first full 10-episode benchmark sweep was completed:

- 6 cloud providers × {paragraph, bullets} = 12 runs
- 12 Ollama models × {paragraph, bullets} = 24 runs
- 3 ML/hybrid baselines × paragraph = 3 runs

Results documented in
`docs/guides/eval-reports/EVAL_BENCHMARK_V1_2026_04.md`.

### Prompt tuning loop infrastructure

`make autoresearch-score` and `scripts/eval/autoresearch_score.py` provide a
composite scalar (ROUGE-L × 0.70 + dual LLM judge × 0.30) used as the ratchet
criterion for Track A. The loop is documented in RFC-057.

---

## Final production state

| Component | Value |
| :--- | :--- |
| `PROD_DEFAULT_SUMMARY_MODE_ID` | `ml_bart_led_autoresearch_v1` (Tier 2, air-gap safe) |
| `OLLAMA_DEFAULT_SUMMARY_MODEL` | `llama3.2:3b` (Tier 3 pointer) |
| Active silver references | 4 × sonnet46 (smoke/benchmark × paragraph/bullets) |
| Paragraph prompt | `long_v1.j2` with thesis opener + vocab alignment instructions |
| Bullets prompt | `bullets_json_v1.j2` with explicit JSON boundary + single-sentence rule |
| Eval matrix | 72 configs — complete 2×2 for all active providers |
| Benchmark report | `EVAL_BENCHMARK_V1_2026_04.md` — first 10-ep sweep |

---

## Decision

RFC-057 is **closed**. No further autoresearch loop iterations are scheduled. The next
trigger for a new loop would be:

1. A new frontier model that qualitatively outperforms Sonnet 4.6 → replace silver
   reference, re-run all providers, run prompt tuning round.
2. A new output format added (e.g. structured JSON with metadata fields) → new silver
   reference and prompt tuning track needed.
3. A significant regression in production ROUGE-L (>3pp) detected in nightly metrics
   → diagnose root cause first; may or may not require a new loop.

## Consequences

- All four-tier summarization modes are stable. No planned changes.
- The eval matrix is now complete — new providers should follow the 4-config template
  documented in `data/eval/configs/README.md`.
- Silver references should not be replaced without a pairwise judge tournament
  (see `configs/README.md` § Silver reference selection).
- The hybrid mode (`ml_hybrid_bart_llama32_3b_autoresearch_v1`) remains registered
  for long-transcript use cases but is no longer the recommended Tier 3 default.
