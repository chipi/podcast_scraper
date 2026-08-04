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
| deepseek-v4-flash-0731 | 0.0092 | 8.44 | 8.44 | 8.22 | 9-0 | quality champion |
| **qwen3.7-flash** | **0.0045** | 8.44 | 8.22 | 7.78 | 9-0 | **value winner** (< Gemini cost) |
| glm-4.7-flash | 0.0131 | 8.14 | 7.71 | 6.86 | 7-0* | pricey + weak topics + PATHOLOGICALLY SLOW (>10min GI calls); *baseline 7-ep |
| Gemini-2.5-flash-lite | 0.0047 | (baseline) | | | — | the cheapest Gemini (v2.4) |
| gemini-2.5-flash | | | | | | pending |
| gemini-2.5-pro | | | | | | pending |
| deepseek-v4-pro | | | | | | pending |
| kimi-k2.6 | | | | | | pending |
| glm-5.2 (full) | | | | | | pending |
| gpt (openai) | | | | | | pending |

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
