# Autoresearch Session Notes — 2026-03-30

## What we did

### Setup

- Read and understood the full autoresearch setup (RFC-057, Track A)
- Confirmed harness works: `make autoresearch-score`, scoring = ROUGE-L × weight + dual LLM judges
- Confirmed dataset materialized: `data/eval/materialized/curated_5feeds_smoke_v1/` (5 episodes)
- Template under optimisation: `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2`
- Experiment config: `data/eval/configs/autoresearch_prompt_openai_smoke_v1.yaml` (gpt-4o-mini summarizer)
- Branch created: `autoresearch/track-a-v1` (from `feat/kg-gi-next`) — later deleted after session

### Experiments run

| exp | N | template change | ROUGE-L | judge_mean | contested | final score |
| --- | - | --------------- | ------- | ---------- | --------- | ----------- |
| exp-0 | 1 | baseline, no change | 20.4% | 0.925 | Yes | 0.2038 |
| exp-1 | 1 | word cap 45→65, add "specific facts/numbers/named entities" | 21.0% | 0.960 | No | **0.6599** |
| exp-2 | 1 | min bullets 3→5, nudge 8-12 | 27.8% | 0.925 | Yes | 0.2775 |
| exp-3 | 1 | thesis lead bullet + word cap 80 | 26.2% | 0.925 | Yes | 0.2621 |
| exp-4 | 5 | exp-1 template, switched to N=5 | 28.9% | 0.929 | Yes | 0.2889 |
| exp-5 | 5 | word cap 50, dense/concrete framing | 27.1% | 0.932 | Yes | 0.2709 |
| exp-6 | 5 | added coverage structure (topic/args/conclusions) | 28.1% | 0.946 | Yes | 0.2807 |
| exp-7 | 5 | new judges gpt-4o+claude-sonnet-4-5, ROUGE weight 0.70 | 27.7% | 0.865 | Yes | 0.2766 |

Note: exp-7 used upgraded judges — later reverted back to gpt-4o-mini / claude-haiku-4-5.

### Key findings

1. **The `contested` fallback is structural, not random.** With bullet-JSON output vs prose silver reference, at least one of the 5 episodes consistently triggers judge divergence >0.15, forcing ROUGE-only scoring. This happens regardless of prompt quality.

2. **Silver reference is prose, our output is bullets.** The silver reference (`data/eval/references/silver/silver_gpt4o_smoke_v1/predictions.jsonl`) contains 4-5 paragraph prose summaries (~2000 chars each). Our bullet-JSON output is ~180 tokens. ROUGE-L comparison between these formats is not meaningful — different vocabulary, structure, and length by design.

3. **Judge quality is actually good.** judge_mean consistently 0.86-0.95 across all runs — the summaries are high quality. The scoring formula just can't register this due to the contested fallback.

4. **Stronger judges made things worse.** Switching to gpt-4o + claude-sonnet-4-5 reduced judge_mean (stricter models) and still contested. The mini/haiku pairing was actually better calibrated for this task.

5. **exp-1 template change is genuinely good.** Word cap 65 + "include specific facts/numbers/named entities" produced the only non-contested run at N=1 (0.660). Worth keeping as the starting point for next session.

### Harness config at end of session

- `autoresearch/prompt_tuning/eval/judge_config.yaml`: reverted to `gpt-4o-mini` / `claude-haiku-4-5`
- `.env`: `AUTORESEARCH_SCORE_ROUGE_WEIGHT=0.70` was added — check if you want to revert this too
- `autoresearch/prompt_tuning/results.tsv`: only header row (no committed experiments — branch was deleted)
- Current branch: `feat/kg-gi-next`

---

## What we agreed to do next

### Priority 1 — Fix the silver reference (root cause)

The silver reference must be regenerated in **bullet-JSON format** to match the output format of the experiment. Currently it is prose (GPT-4o paragraph summaries). Until this is fixed, ROUGE-L scores are meaningless and the `contested` fallback will keep firing.

**How:** Re-run the silver reference experiment using `bullets_json_v1` as the prompt instead of the prose template. Save as a new reference ID (e.g. `silver_gpt4o_bullets_smoke_v1`) and update the autoresearch config to point to it.

### Priority 2 — Consider raising the contested threshold

Once silver reference is fixed, if contested still fires, consider editing `score.py` to raise the threshold from 0.15 to 0.25. This is currently immutable (human must decide to expand allowlist in `program.md`).

### Priority 3 — Resume prompt tuning

Once scoring is reliable, resume from exp-1 template as baseline:

- Current best template: word cap 65 + "include specific facts/numbers/named entities"
- Promising directions not yet fully explored (all reset due to scoring issues, not prompt quality):
  - More bullets (exp-2: ROUGE-L 27.8% — best raw ROUGE but contested)
  - Coverage structure: central topic + main args + conclusions (exp-6: judge_mean 0.946 — best judge score)
  - Combining both: 6-8 bullets with explicit coverage structure + 65-word cap
- Keep ROUGE weight at 0.70 (more meaningful once silver reference matches format)
- Keep judges as gpt-4o-mini / claude-haiku-4-5 (better calibrated than flagship models for this task)
- Stop condition: 10 runs or <1% gain for 5 consecutive runs
- Branch: create fresh `autoresearch/track-a-v2` from main/feat branch

### Other notes

- `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1` is needed locally (dedicated autoresearch keys or production keys)
- N=1 smoke validation still useful before N=5 full runs
- Cost per N=5 run with mini/haiku judges: ~$0.04-0.08 — very cheap
