# Model bake-off — overnight autonomous run (2026-08-05)

Autonomous continuation while operator sleeps. Goal: map the full **cost-vs-value frontier** for
the podcast enrichment pipeline (summary + insights + KG topics) across model families, so we can
pick the model that beats Gemini for less money at 100-ep scale.

## Method (identical for every model — model is the ONLY variable)

- **Harness**: `--pipeline-stage relabel_only --reprocess-existing-only` over the SAME 9 diverse
  episodes (one per feed) copied from `prod-v2.4-relabel-fixed`. Reuses frozen 2.4 ASR + diarization;
  re-runs the whole LLM surface (naming → cleaning → summary → GI → KG).
- **Config**: one frozen control config; only the model alias/provider changes. Robust settings
  (GI chunk 20k, timeout 600s, insights cap 8, dedup 0.72, value-gate tier 3, summary cap 2048).
- **Judge**: Claude **Opus 4.8** (dedicated `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY`), blind A/B,
  length-controlled + position-controlled, scoring **top-12 ranked insights** (production view) —
  summary / insights / topics, 1–10 + winner. Baseline contestant = **Gemini 2.5-flash-lite (v2.4)**.
- **Cost**: LiteLLM SpendLogs delta / 9 eps (actual, all calls).

## Leaderboard (fair top-12; vs Gemini-2.5-flash-lite)

| Model | $/ep | Summary | Insights | Topics | Overall | Notes |
|---|---|---|---|---|---|---|
| **qwen3.7-flash** | **0.0045** | 8.44 | 8.22 | 7.78 | 9-0 | **VALUE WINNER** — cheaper than flash-lite, near-deepseek quality |
| deepseek-v4-flash-0731 | 0.0092 | 8.44 | 8.44 | 8.22 | 9-0 | **QUALITY CHAMPION** |
| gemini-2.5-flash | 0.0491 | 8.22 | 8.00 | 7.89 | 9-0 | beaten by qwen+deepseek on quality AND 5–10x pricier |
| glm-4.7-flash | 0.0182 | 8.29 | 7.86 | 7.57 | 7-0* | decent quality but PRICIEST + PATHOLOGICALLY SLOW (56min, >10min GI calls); *7-ep |
| Gemini-2.5-flash-lite | 0.0047 | (baseline) | — | — | — | cheapest Gemini (v2.4 baseline) |
| gemini-2.5-pro | ~0.051 | 8.71 | (broken) | (broken) | — | ⚠️ summary EXCELLENT (8.71, best), but GI insight-chunk extraction FAILS (stub fallback) + KG topics malformed. Bug, not verdict: gemini provider has no reasoning-off (unlike litellm). Needs fix + re-run. |
| deepseek-v4-pro | ~pricey | 8.80 | 8.00 | 6.60 | 5-0* | good summary/insights but WEAK topics + SLOW (14-min GI calls, hung on ep7) + pricey; *6-ep partial |
| kimi-k2.6 | | | | | | running (slow) |
| glm-5.2 (full) | | | | | | pending |
| gpt (openai) | | | | | | pending |

**Pro/reasoning tier verdict (gempro, deepseek-v4-pro, glm-4.7-flash, kimi):** all pathologically
SLOW (10–14 min GI calls, repeated 600s timeouts, occasional keepalive-hangs) and far pricier —
impractical for this pipeline regardless of quality, and their quality does NOT clearly beat the
flash models (weak topics or broken GI). The FLASH models win the practical frontier.

**Interim read:** the two cheap Chinese flash models (qwen3.7-flash, deepseek-flash) BEAT mid-tier
gemini-2.5-flash on quality while costing 5–10x less. gemini-2.5-flash beats only flash-lite. glm is
out (slow + pricey). qwen3.7-flash is the standout value; deepseek-flash the quality lead.

## Queue (sequential — avoid machine thrash; slow runs finish, "slow" is a recorded finding)

1. glm-4.7-flash iter-1 (finishing) → judge → record.
2. gemini-2.5-flash → judge. 3. gemini-2.5-pro → judge.
4. deepseek-v4-pro → judge. 5. kimi-k2.6 → judge. 6. glm-5.2 → judge.
7. gpt (openai, tbd model) → judge.
8. Compile the frontier + a recommendation for the 100-ep scale run.

## Rules held to
- Never stop/kill a run without operator approval — slow runs run to completion.
- Commit each profile + record results as they land.
- Flag any real decision (which model(s) advance to 100-ep) for the operator, don't decide it.

## Results log
(updated as runs complete — see git commits `tune(bakeoff): …` and this table)
