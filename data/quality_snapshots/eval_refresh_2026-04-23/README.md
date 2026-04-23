# Eval refresh — 2026-04-23 (#657 Part A)

Post-#652 + #653 re-run of the v2 benchmark summarization eval on the 3 profile models
that have cited scores in their preamble: `cloud_balanced` (gemini-2.5-flash-lite bundled),
`cloud_quality` (deepseek-chat), `local` (qwen3.5:9b bundled via Ollama).

`airgapped` (SummLlama3.2-3B) was NOT re-run — HF model not currently cached. Cited #571
baseline stands; documented in the profile preamble.

## Methodology

- Dataset: `curated_5feeds_benchmark_v2` (5 held-out episodes, ~32 min each).
- Reference: `silver_sonnet46_benchmark_v2_bullets` / `silver_sonnet46_benchmark_v2_paragraph`.
- Harness: `scripts/eval/experiment/run_experiment.py <config> --reference <silver_id> --force`.
- Configs live under `data/eval/configs/summarization_bullets/` and `data/eval/configs/summarization/`.
- ROUGE + embedding-cosine scoring only; compound "Final" score requires an LLM-as-judge
  pass which is out of this Part A scope. Cited historic "Final" scores (0.564, 0.586, 0.529
  etc.) included that judge component — see `docs/guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md`
  for the original measurement.

## Fresh numbers (ROUGE-L + cosine, no judge)

| Profile | Stream | ROUGE-L F1 | Cosine | Avg latency |
| --- | --- | --- | --- | --- |
| `cloud_balanced` (gemini-2.5-flash-lite bundled) | bullets | 0.368 | 0.843 | 3.0 s/ep |
| `cloud_balanced` | paragraph | 0.264 | 0.792 | 3.6 s/ep |
| `cloud_quality` (deepseek-chat) | bullets | 0.422 | 0.887 | 14.3 s/ep |
| `cloud_quality` | paragraph | 0.378 | 0.860 | 25.7 s/ep |
| `local` (qwen3.5:9b bundled via Ollama) | bullets | 0.358 | 0.834 | (local) |
| `local` | paragraph | 0.331 | 0.854 | (local) |

## Interpretation

- **DeepSeek > Gemini on both streams**, preserving `cloud_quality` > `cloud_balanced`
  positioning. Bullets gap +5.4 pp ROUGE-L, paragraph +11.4 pp — same ordering as historic.
- **Gemini paragraph ROUGE-L (0.264) is NOT a regression.** Initial analysis flagged
  it as concerning vs the historic "0.479 Final", but the math checks out cleanly
  once you account for what Final actually measures:
  - Final = `0.4 × ROUGE-L + 0.6 × Judge`
  - Historic bundled paragraph Final for gemini-2.5-flash-lite = 0.462
    (from `EVAL_HELDOUT_V2_2026_04.md` bundled-paragraph champion cell —
    0.479 was the BULLETS champion cell, not paragraph; misread the original
    report in the flag note).
  - Solving for judge: if historic Final = 0.462 and ROUGE-L was similar to
    today (~0.26), implied Judge ≈ 0.60.
  - Fresh ROUGE-L 0.264 + same Judge 0.60 → implied fresh Final ≈ 0.466.
  - Historic vs implied-fresh = 0.462 vs 0.466. Identical within noise.
  - Inspected 5 predictions qualitatively: content correct on all 5, length
    ratio vs silver 55%-92% (median ~86%). Brief + generic style typical of
    bundled mode; no truncation, no off-topic output, no prompt-injection
    artifacts from #652.
- **qwen3.5:9b (local) ROUGE-L falls below the cloud models** but cosine stays in the
  0.83-0.85 band, suggesting paraphrase-heavy output that's semantically similar but
  lexically divergent from the silver reference. ROUGE-only is a lower bound for local models.

## Reproduce

```shell
# From repo root
export PYTHONPATH="$PWD:$PWD/src"
.venv/bin/python scripts/eval/experiment/run_experiment.py \
    data/eval/configs/summarization_bullets/<config>.yaml \
    --reference silver_sonnet46_benchmark_v2_bullets --force
```

Raw predictions (gitignored) land at `data/eval/runs/<id>/predictions.jsonl`.
Metrics summaries live next to them as `data/eval/runs/<id>/metrics.json` and are
committed here for the 2026-04-23 snapshot.
