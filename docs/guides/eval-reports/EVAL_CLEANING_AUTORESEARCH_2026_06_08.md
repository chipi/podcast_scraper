# Eval: Transcript cleaning autoresearch (#594)

**Date:** 2026-06-08
**Ticket:** [#594](https://github.com/chipi/podcast_scraper/issues/594)
**Parent epic:** [#907](https://github.com/chipi/podcast_scraper/issues/907)
**Companion:** [#921](https://github.com/chipi/podcast_scraper/issues/921) (v3 fixtures rebuild)

## TL;DR

Two temperature defaults bumped from 0.2 → 0.4:

| Provider | `*_cleaning_model` | `*_cleaning_temperature` |
| --- | --- | --- |
| anthropic | claude-haiku-4-5 (unchanged) | 0.2 → **0.4** |
| gemini | gemini-2.5-flash-lite (unchanged) | 0.2 → **0.4** |
| openai | gpt-4o-mini (unchanged) | **0.2** (unchanged — see "why not openai") |
| deepseek | deepseek-chat (unchanged) | **0.2** (unchanged — already optimal at 0.2) |

Cleaning model choices are **already correct** (gpt-4o-mini, haiku-4-5, gemini-2.5-flash-lite have been defaults since pre-#594; the "16× cost-reduction" case the ticket flagged for OpenAI already shipped). The leftover win was temperature tuning: 0.4 beats 0.2 on the v2 sweep AND prod validation for two of three cloud providers; OpenAI defended at 0.2 on prod despite a marginal v2 win.

## Method

1. **Generate silver** — Sonnet 4.6 cleans 5 v2 smoke episodes (`curated_5feeds_smoke_v2`). Silver at `data/eval/references/silver/cleaning_v1/`. Script: `scripts/eval/score/cleaning_silver_v1.py`.
2. **Sweep** — for each (provider, model, temperature) cell, direct API call with the canonical cleaning system prompt mirrors `providers/*/clean_transcript()` (decoupled from `LLMBasedCleaner`'s length-guard so we're measuring model behavior, not wrapper logic). Script: `scripts/eval/score/cleaning_sweep_v1.py`. Matrix:
   - openai: gpt-4o-mini, gpt-4o
   - anthropic: claude-haiku-4-5
   - gemini: gemini-2.5-flash-lite, gemini-2.0-flash *(deprecated mid-sweep)*, gemini-2.5-flash
   - deepseek: deepseek-chat
   - temperatures: 0.0, 0.2, 0.4
3. **Per-cell metrics** — similarity-to-silver (SequenceMatcher), sponsor pattern residual hits, length retention, latency.
4. **Pairwise judge tournament** (`scripts/eval/score/cleaning_judge_v1.py`) — Sonnet 4.6 picks winner between provider best-cells, episode by episode. Decouples ranking from "silver-style bias".
5. **Prod validation** (`scripts/eval/score/cleaning_validate_prod_v1.py`) — 3 episodes from `.test_outputs/manual/my-manual-run-10`; per-provider OLD(t=0.2) vs NEW(t=0.4) pairwise. Bump applies only if NEW wins or ties on real-prod content.

## Per-cell similarity-to-silver

| Provider | Model | T | Sim | Sponsor residual | Cleaned/silver | Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| anthropic | claude-haiku-4-5 | 0.0 | 0.660 | 0.0 | 116% | 10.6s |
| anthropic | claude-haiku-4-5 | 0.2 | 0.660 | 0.0 | 116% | 10.0s |
| anthropic | claude-haiku-4-5 | **0.4** | **0.663** | **0.0** | 116% | 10.2s |
| deepseek | deepseek-chat | 0.0 | 0.626 | 0.0 | 111% | 13.4s |
| deepseek | deepseek-chat | **0.2** | **0.628** | **0.0** | 111% | 14.2s |
| deepseek | deepseek-chat | 0.4 | 0.626 | 0.0 | 111% | 14.2s |
| gemini | gemini-2.5-flash-lite | 0.0 | 0.602 | 0.8 | 117% | 4.6s |
| gemini | gemini-2.5-flash-lite | 0.2 | 0.602 | 0.8 | 117% | 4.4s |
| gemini | gemini-2.5-flash-lite | **0.4** | **0.615** | **0.8** | 117% | 4.4s |
| gemini | gemini-2.5-flash | 0.0 | 0.564 | 0.4 | 117% | 18.3s |
| gemini | gemini-2.5-flash | 0.2 | 0.466 | 1.2 | 117% | 17.6s |
| gemini | gemini-2.5-flash | 0.4 | 0.426 | 0.4 | 103% | 21.5s |
| openai | gpt-4o-mini | 0.0 | 0.620 | 0.0 | 118% | 34.4s |
| openai | gpt-4o-mini | 0.2 | 0.619 | 0.0 | 119% | 28.2s |
| openai | gpt-4o-mini | **0.4** | **0.625** | **0.0** | 118% | 31.3s |
| openai | gpt-4o | 0.0 | 0.453 | 0.0 | 118% | 15.1s |
| openai | gpt-4o | 0.2 | 0.531 | 0.0 | 119% | 14.4s |
| openai | gpt-4o | 0.4 | 0.427 | 0.0 | 120% | 14.7s |

Notes:

- **Sponsor residual: 0** across every non-Gemini cell. The detector pattern fixes from PR #918 + the cleaning prompt do their job for OpenAI/Anthropic/DeepSeek. Gemini Flash Lite leaks ~0.8 sponsor pattern hits per episode on average — small but real, worth flagging for #905.
- **`gemini-2.5-flash` mid-tier is unstable** at higher temps (0.564 → 0.466 → 0.426 as temp climbs). Flash Lite is more consistent. **gemini-2.5-flash should NOT be chosen as a cleaning default** — the bigger model isn't better here.
- **gpt-4o is WORSE than gpt-4o-mini** on similarity-to-silver (0.45 vs 0.62 at the same temp). The more expensive model drifts further from Sonnet's preserve-content style. Cost-justified validation that the existing gpt-4o-mini default is correct.
- **Latency winner:** Gemini Flash Lite at ~4.5s — 7× faster than gpt-4o-mini. Big operational win for the cleaning stage, which runs on every episode.

## Pairwise tournament

Provider-best cells (best temp per provider, picked by similarity-to-silver) cross-judged pairwise, 5 episodes × 6 pairings = 30 verdicts:

| Provider | Wins | Losses | Ties | Net |
| --- | ---: | ---: | ---: | ---: |
| openai (gpt-4o-mini, t=0.4) | **12** | 1 | 2 | **+11** |
| anthropic (claude-haiku-4-5, t=0.4) | 9 | 4 | 2 | +5 |
| gemini (gemini-2.5-flash-lite, t=0.4) | 4 | 11 | 0 | −7 |
| deepseek (deepseek-chat, t=0.2) | 3 | 12 | 0 | −9 |

**OpenAI wins the tournament** even though Anthropic had higher similarity-to-silver. Interpretation: similarity-to-silver biases toward Sonnet-style cleaning (which silver came from); the independent judge prefers gpt-4o-mini's less-aggressive style. This is good evidence that the similarity-to-silver metric alone is not a complete proxy for cleaning quality — but it's directionally correct enough to pick per-provider best cells, then arbitrate with the judge.

## Real-prod validation

3 episodes from `manual-run-10`. Pairwise OLD(t=0.2) vs NEW(t=0.4) for each provider:

| Provider | OLD(0.2) wins | NEW(0.4) wins | Ties | Verdict |
| --- | ---: | ---: | ---: | --- |
| anthropic | 0 | **2** | 0 | NEW safe — bump applied |
| gemini | 0 | **1** | 2 | NEW safe (1 invalid parse) — bump applied |
| openai | **1** | 2 | 0 | mixed — bump **NOT** applied (see below) |

The OpenAI `0024` over-cleaning case: t=0.4 produced 4733c vs 11962c at t=0.2 — judge preferred OLD. v2 sweep advantage was only +0.6% (0.625 vs 0.619), too thin to justify a temp bump that exhibits over-cleaning variance on real content. OpenAI stays at 0.2.

## Why these changes are conservative

- Cleaning models: no changes (the "16× cost reduction" case the ticket flagged is already shipped — OpenAI defaulted to gpt-4o-mini before this work).
- Temperature: only bumped where prod-validated as safe (anthropic + gemini, both wins-or-ties on every prod episode).
- DeepSeek: t=0.2 was already best per the v2 sweep; no change.
- OpenAI: even though v2 sweep marginally preferred t=0.4 (+0.6%), prod validation showed over-cleaning risk — defer the bump until we have stronger signal (e.g. larger prod sample or until #905 lands a cleaning-profile selection harness).

## v3 fixtures contribution

Real cleaning failure modes surfaced for #921:

- **Gemini sponsor leakage** — Flash Lite leaves ~0.8 pattern hits per episode, mostly closing-CTA fragments the LLM doesn't recognise as sponsor-class. v3 should add episodes where sponsor content uses non-canonical phrasing (e.g. native-ad blocks with no obvious "brought to you by" marker) so cleaning evaluation isn't gated only on template-matching.
- **Over-cleaning at higher temps (OpenAI specifically)** — v3 could include episodes where a chunk of substantive content has tone/structure that resembles sponsor copy (e.g. host enthusiastic recommendation that's *content*, not a sponsor). Lets the cleaning-profile selection (#905) score "preserves real content" precisely.
- **The gpt-4o cleaning regression** is upstream of fixtures — but worth documenting as autoresearch evidence that "bigger model = better cleaning" is not a safe assumption.

Logged in `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md`.

## Acceptance

- [x] Silver cleaned transcripts committed (5 v2 episodes, Sonnet 4.6).
- [x] Per-cell sweep metrics committed (`data/eval/runs/baseline_cleaning_autoresearch_v1/metrics.jsonl`).
- [x] Pairwise judge tournament committed (`judge.json`).
- [x] Real-prod validation committed (`data/eval/runs/baseline_cleaning_validate_prod_v1/`).
- [x] Default temperature applied for anthropic + gemini in `src/podcast_scraper/config.py` and provider `__init__` fallback.
- [x] Eval report (this file).
- [x] v3 contribution logged in `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md`.

## Out of scope

- **Cleaning profile selection** (cleaning_v1 vs v2 vs v3 vs v4 vs hybrid_after_pattern) — that's #905 Tier 2's job. This issue tunes model + temperature WITHIN cleaning_v4.
- **Ollama variants** — daemon is user-managed per project convention; the ticket explicitly listed `llama3.2:3b`, `mistral:7b`, `qwen3.5:9b` but local-model autoresearch requires the user to start the right models. Tracking as a follow-up (could be folded into #905 or a dedicated mini-ticket).
- **Per-cell cost accounting** — total spend for the sweep + judge + validation was ~$3–4, an order of magnitude below the ticket's "$5–15" budget estimate. Cost ranking didn't differentiate between cells materially (all under $0.005/ep for cleaning).

## Reproduction

```bash
export $(grep -E '^(OPENAI|ANTHROPIC|GEMINI|DEEPSEEK)_API_KEY=' .env)

PYTHONPATH=. python scripts/eval/score/cleaning_silver_v1.py \
    --sources data/eval/sources/curated_5feeds_raw_v2 \
    --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \
    --output data/eval/references/silver/cleaning_v1

PYTHONPATH=. python scripts/eval/score/cleaning_sweep_v1.py \
    --sources data/eval/sources/curated_5feeds_raw_v2 \
    --silver  data/eval/references/silver/cleaning_v1 \
    --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \
    --output data/eval/runs/baseline_cleaning_autoresearch_v1

PYTHONPATH=. python scripts/eval/score/cleaning_judge_v1.py \
    --sweep-output data/eval/runs/baseline_cleaning_autoresearch_v1 \
    --sources data/eval/sources/curated_5feeds_raw_v2 \
    --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \
    --output data/eval/runs/baseline_cleaning_autoresearch_v1/judge.json

PYTHONPATH=. python scripts/eval/score/cleaning_validate_prod_v1.py \
    --prod-transcripts-dir '.test_outputs/manual/my-manual-run-10/run_20260421-190016_2606de6d/transcripts' \
    --output data/eval/runs/baseline_cleaning_validate_prod_v1
```
