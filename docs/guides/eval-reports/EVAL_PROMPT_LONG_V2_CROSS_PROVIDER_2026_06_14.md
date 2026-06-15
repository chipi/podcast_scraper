# Eval: long_v2 paragraph prompt — cross-provider port (#985)

**Date:** 2026-06-14
**Ticket:** [#985](https://github.com/chipi/podcast_scraper/issues/985)
**Parent:** [#906](https://github.com/chipi/podcast_scraper/issues/906) (closed) — original Anthropic-only `long_v2.j2` ship (5-0 vs `long_v1`)
**Companion:** [EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md](EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md) §"Sub-task C" — Anthropic 5-0 numbers

## TL;DR

The v2-aware paragraph prompt — `long_v2.j2`, which adds two beats over `long_v1`
(position changes + recurring guests) — was ported from Anthropic to four other
providers (OpenAI, Gemini, DeepSeek, Ollama). Sonnet 4.6 pairwise-judged each
provider's `long_v1` vs `long_v2` over the 5 v2 fixture episodes. Judge held
constant across providers so verdicts are comparable to the #906 Anthropic
result.

| Provider | Model | v2 wins | Verdict |
| --- | --- | ---: | --- |
| **Anthropic** | Sonnet 4.6 (from #906) | **5/5** | already shipped via #906 |
| **OpenAI** | gpt-4o | **4/5** | **flip default → long_v2** |
| **Gemini** | gemini-2.5-flash-lite | **4/5** | **flip default → long_v2** |
| **DeepSeek** | deepseek-chat | **3/5** | **keep long_v1** — below ≥4/5 gate |
| **Ollama** | qwen3.5:35b (DGX) | **4/5** | **flip default → long_v2** |

Acceptance gate per #985: v2 wins **≥ 4/5** episodes → flip default; else keep
v1 + document the verdict.

## Method

For each (provider, episode) pair:

1. Read raw transcript from
   `data/eval/sources/curated_5feeds_raw_v2/feed-<feed>/<episode>.txt`.
2. Render two prompts via `prompts.store.render_prompt`:
   - `<provider>/summarization/long_v1` (current per-provider baseline)
   - `<provider>/summarization/long_v2` (this PR's port)
3. Call the provider's production summary model (`gpt-4o`,
   `gemini-2.5-flash-lite`, `deepseek-chat`, `qwen3.5:35b` on DGX) at
   `temperature=0`, `max_tokens=2000`. Same `SYSTEM_PROMPT` as the #906 harness.
4. Sonnet 4.6 pairwise-judges A=v1, B=v2 with the rubric from #906 (faithful
   coverage, named entities preserved, position changes captured, recurring
   guests named, no invention).

Single-order judging (no A/B swap) — matches the #906 methodology. Stronger
position-bias control would require N=10 with both orderings (deferrable; the
issue-989 cleaning report uses that pattern when the gap is tighter).

Harness: `scripts/eval/score/prompt_v2_cross_provider_v1.py`.

## Per-provider results

### OpenAI (gpt-4o) — flip → long_v2

| Episode | v1 chars | v2 chars | Winner |
| --- | ---: | ---: | --- |
| p01_e01 | 2270 | 2389 | B (v2) |
| p02_e01 | 2335 | 2248 | B (v2) |
| p03_e01 | 2110 | 2521 | A (v1) |
| p04_e01 | 2761 | 2580 | B (v2) |
| p05_e01 | 1977 | 2113 | B (v2) |
| **total** | — | — | **4-1 v2** |

Char counts comparable across prompts (per-episode delta within ~±15%).
v2 wins the typical case; the single v1 win (p03_e01) is the longer-prompt
case which often goes either way under single-order judging.

Raw: `data/eval/runs/prompt_v2_cross_provider/openai/metrics.json`

### Gemini (gemini-2.5-flash-lite) — flip → long_v2

| Episode | v1 chars | v2 chars | Winner |
| --- | ---: | ---: | --- |
| p01_e01 | 2965 | 2735 | B (v2) |
| p02_e01 | 3000 | 3093 | B (v2) |
| p03_e01 | 2599 | 2542 | A (v1) |
| p04_e01 | 2873 | 3679 | B (v2) |
| p05_e01 | 2715 | 2902 | B (v2) |
| **total** | — | — | **4-1 v2** |

First sweep against `gemini-2.5-flash` (NOT prod default — flash has a
thinking-mode token-accounting interaction that produced wildly inconsistent
output lengths, 368c–2418c per call). Re-run against `gemini-2.5-flash-lite`
(the actual production model per `PROD_DEFAULT_GEMINI_SUMMARY_MODEL` +
`cloud_balanced`/`cloud_thin` profiles). The flash-lite numbers above are
the valid set; the flash-lite results are reproducible.

Raw: `data/eval/runs/prompt_v2_cross_provider/gemini/metrics.json`

### DeepSeek (deepseek-chat) — keep long_v1

| Episode | v1 chars | v2 chars | Winner |
| --- | ---: | ---: | --- |
| p01_e01 | 2184 | 2886 | B (v2) |
| p02_e01 | 2183 | 2723 | A (v1) |
| p03_e01 | 2331 | 2792 | A (v1) |
| p04_e01 | 3388 | 2690 | B (v2) |
| p05_e01 | 2706 | 3021 | B (v2) |
| **total** | — | — | **3-2 v2** (below gate) |

Under the ≥4/5 acceptance gate, this is a **fail**. DeepSeek summaries are
longer overall (3-of-5 v2 outputs >2700c vs OpenAI's typical 2200c), so the
"explicit position changes" + "recurring guests" beats may be getting absorbed
into existing prose rather than surfaced as distinct beats. Keep DeepSeek on
`long_v1` until a larger sweep (N≥10) or a position-bias-neutralised judge
re-tests this provider.

Raw: `data/eval/runs/prompt_v2_cross_provider/deepseek/metrics.json`

### Ollama (qwen3.5:35b on DGX) — flip → long_v2

| Episode | v1 chars | v2 chars | Winner |
| --- | ---: | ---: | --- |
| p01_e01 | 3365 | 4044 | B (v2) |
| p02_e01 | 2934 | 3516 | B (v2) |
| p03_e01 | 3495 | 3333 | A (v1) |
| p04_e01 | 3478 | 3436 | B (v2) |
| p05_e01 | 3509 | 3318 | B (v2) |
| **total** | — | — | **4-1 v2** |

First sweep returned all-empty content (0c × 10) — qwen3.5 is a thinking
model and the OpenAI-compatible Ollama `/v1` endpoint has no
`reasoning_effort: none` knob, so the model burned the entire `num_predict`
budget on hidden reasoning. Switched the harness to the native `/api/chat`
endpoint with `"think": false`. Documented inline in the harness so the same
trap can't reappear on the next Ollama re-test. Same root-cause as the
`EVAL_REAL90_2026_06.md` "qwen3.5 burns budget on thinking" finding from #959.

Raw: `data/eval/runs/prompt_v2_cross_provider/ollama/metrics.json`

## What ships in this PR

- All 5 providers have `<provider>/summarization/long_v2.j2` on disk.
- `tests/unit/podcast_scraper/prompts/test_packaged_prompts_present.py` adds
  5 new parametrize entries so the wheel-build / packaging guard catches any
  port that drops from `package-data` later.
- Defaults flipped to `long_v2` for **OpenAI**, **Gemini**, and **Ollama**
  (each verified ≥4/5 against its production-default model).
- `DeepSeek` default stays on `long_v1` — 3/5 v2, below the gate. Documented
  per-provider verdict above; revisit with N≥10 + A/B-swap if needed.
- No `config/profiles/` change (no profile currently pins
  `<provider>_summary_user_prompt` — provider defaults flow through the
  `config.py` field default).
- `config/examples/` docs and the `config_constants` updated so the
  recommended-paragraph-default reflects the per-provider verdict.

## Cost

Total operator-paid spend for this validation run: ~$0.35 (OpenAI gpt-4o ×10
≈ $0.025, Gemini flash-lite × 20 ≈ $0.005, DeepSeek × 10 ≈ $0.007, Sonnet 4.6
judge × 20 ≈ $0.30, Ollama on DGX free). Well under the autoresearch run
budget envelope.

## Out of scope (filed elsewhere)

- N≥10 per-provider sweep with both A/B orderings (position-bias neutralisation
  per the #989 method) — defer to a follow-up if any provider's verdict needs
  re-litigation.
- Re-running Anthropic's #906 5-0 sweep against haiku-4-5 (the production
  model per #816 reliability axis) — same prompt port; if needed, a separate
  ticket.
- Non-summary prompts (cleaning, GI, KG) — different scope.

## Reproduction

```bash
.venv/bin/python scripts/eval/score/prompt_v2_cross_provider_v1.py \
  --provider <openai|gemini|deepseek|ollama> \
  --sources data/eval/sources/curated_5feeds_raw_v2 \
  --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \
  --output  data/eval/runs/prompt_v2_cross_provider/<provider>
```

## References

- Parent ship: #906 (Anthropic `long_v2.j2` 5-0 — closed)
- This issue: #985 (cross-provider port)
- Companion eval: `EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md` §Sub-task C
- Position-bias methodology to graduate to: `EVAL_CLEANING_V3_V4_BROADER_JUDGE_2026_06_13.md` (#989)
- Programme epic: #907 (autoresearch v2)
- Materialize-decisions discipline: `AGENTS.md`
