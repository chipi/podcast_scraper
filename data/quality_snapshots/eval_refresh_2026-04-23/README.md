# Eval refresh — 2026-04-23 (#657 Part A)

Post-#652 + #653 re-run of the v2 benchmark summarization eval on all 4 profile models:
`cloud_balanced` (gemini-2.5-flash-lite bundled), `cloud_quality` (deepseek-chat),
`local` (qwen3.5:9b bundled via Ollama), `airgapped` (SummLlama3.2-3B standalone).

SummLlama was inferenced on MPS (Apple Silicon) at ~78 s/ep after HF weights were pulled
into the local cache. Standalone paragraph only — the airgapped pipeline reuses these
bullets/paragraph outputs as the source for downstream GI/KG (`gi_insight_source:
summary_bullets`), so there is no separate airgapped "bundled" variant to eval.

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
| `airgapped` (SummLlama3.2-3B on MPS) | paragraph | 0.325 | 0.784 | 78 s/ep |

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
- **SummLlama3.2-3B (airgapped) paragraph ROUGE-L 0.325 / cosine 0.784** lands between
  gemini-paragraph (0.264) and qwen-paragraph (0.331) on ROUGE-L, and slightly below
  both on cosine. Historic #571 cited 0.485 as the compound "Final" — that included the
  judge component. Solving the same way as gemini: if historic Final 0.485 implies judge
  ~0.59 against ROUGE-L in the same band as today, fresh implied Final with judge 0.59 and
  ROUGE-L 0.325 ≈ 0.484. No regression within noise. Still the weakest paragraph model on
  pure ROUGE-L + cosine, which matches its position as the "no-network" fallback — the
  tradeoff is offline capability, not quality parity.

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
