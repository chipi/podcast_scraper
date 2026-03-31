# GIL: ML vs OpenAI — outcome benchmark (WIP note)

**Status:** Working note — methodology for comparing grounded-insights **outcomes**
on the **same episodes**, not for aligning YAML **parameter numbers** across stacks.

**Related:** [GROUNDED_INSIGHTS_GUIDE.md](../guides/GROUNDED_INSIGHTS_GUIDE.md),
`scripts/tools/gil_quality_metrics.py`, PRD-017 / `src/podcast_scraper/gi/quality_metrics.py`.

## Goal

- Use **OpenAI** (or another LLM evidence path) as a **reference outcome** on a fixed
  episode set.
- Improve the **ML evidence stack** (extractive QA + local NLI, thresholds, models)
  so that **artifacts and metrics** get **as close as practical** to that reference.
- **Do not** tune ML so that `gi_qa_score_min` / `gi_nli_entailment_min` “match”
  OpenAI numerically — scores are **not comparable** across implementations
  (e.g. OpenAI `extract_quotes` uses `qa_score=1.0` after span resolution; ML uses
  real SQuAD-style probabilities and CrossEncoder entailment).

## Compare outcomes, not hyperparameters

**Same inputs**

- Same feed and **same episode list** (or frozen episode IDs / transcript hashes).
- Where possible, **same insight text** for both runs (e.g. reuse **summary bullets**
  from one summary generation, or document explicitly if insights differ).

**Two runs**

- **Reference:** e.g. `quote_extraction_provider` / `entailment_provider: openai`
  (and/or LLM `summary_provider` if comparing full stack).
- **Candidate:** best-effort **ML** path (`transformers` / `hybrid_ml` evidence).

**Diff the products**

- Per run: `metadata/*.gi.json`, run-level `metrics.json`.
- Compare **grounding coverage**, **quotes per insight**, **verbatim span validity**,
  not config literals.

## Operational metrics (suggested)

Use a **small fixed benchmark** (e.g. 20–50 episodes) and report **side by side**.

| Level | Metric | Notes |
| ----- | ------ | ----- |
| Episode | % episodes with ≥1 grounded quote | Coverage vs reference |
| Insight | % insights with ≥1 quote when reference has one | Recall-like vs OpenAI |
| Quote | Span overlap (IoU) or exact `[char_start, char_end)` match | Same insight index / aligned row |
| Quality | Verbatim slice check; optional human or embedding sanity | Precision / semantics |

**Existing tooling**

- Run-level aggregates: `make gil-quality-metrics DIR=<run_root>` or
  `scripts/tools/gil_quality_metrics.py` (see GROUNDED_INSIGHTS_GUIDE).

**Gap / follow-up**

- A **small paired-diff script** (load two run dirs, match by `episode_id`, align
  insights by order or stable text hash, compute overlap / agreement rates) would
  make “distance to OpenAI” explicit. Not required to start manually.

## Closing the ML ↔ OpenAI gap (order of leverage)

1. **Lock the benchmark** — frozen episode list + cached transcripts/summaries so
   comparisons are reproducible.
2. **Align insight text** — same bullets (or same provider-generated insights) for
   both evidence paths so QA/NLI see identical hypotheses where intended.
3. **Improve the stack** — QA/NLI model choice, windowing, ASR cleanup for evidence
   only, etc. **Re-score agreement vs reference** after each change.
4. **Tune thresholds last** — adjust `gi_qa_score_min` / `gi_nli_entailment_min`
   on the dev benchmark to **maximize outcome agreement** (or F1 vs reference),
   subject to floors on invalid / non-verbatim spans.
5. **Hybrid ceiling** — ML summaries + **LLM evidence** (`quote_extraction_provider` /
  `entailment_provider: openai`) is a practical way to approach OpenAI-like behavior
   without re-summarizing on the API.

## Expectations

- You can **systematically shrink** the outcome gap and report **how close ML gets**
  on the benchmark.
- **Perfect** per-insight parity with OpenAI on hard paraphrase + noisy ASR is **not**
  guaranteed with a purely local stack; some LLM in the evidence path (or richer ML)
  may be needed for a tight ceiling.

## Implementation note (CLI + YAML)

- As of the fix tracked in this branch, `_build_config()` in `cli.py` must **forward**
  GIL tuning fields from the merged argparse namespace into `Config`; otherwise YAML
  `gi_qa_score_min` / `gi_nli_entailment_min` / window fields are ignored on CLI runs.
  When running paired benchmarks, confirm `Config` reflects the file (e.g. log or
  `config_snapshot` in metadata).

## Implemented (v1 housekeeping)

- **Paired configs:** `config/manual/gil_paired_benchmark_openai_evidence.yaml` (OpenAI
  evidence, ML summaries) and `config/manual/gil_paired_benchmark_ml_evidence.yaml`
  (local evidence). Same feed and `max_episodes`; distinct `output_dir` under
  `.test_outputs/benchmark/`.
- **Compare script:** `make compare-gil-runs REF=<run_a> CAND=<run_b>` or
  `scripts/tools/compare_gil_runs.py` (logic in `src/podcast_scraper/gi/compare_runs.py`)
  — per-episode quote/grounded counts and episode-level “both have quotes” summary.
- **CLI:** `_build_config` forwards `gi_embedding_model` (and other GIL keys); INFO log
  line for effective `gi_qa_score_min` / `gi_nli_entailment_min` / evidence providers /
  `gi_embedding_model` when `generate_gi` is true.

## Next steps (optional)

- [ ] Run both benchmark configs on the same machine; record two `run_*` roots and archive metrics.
- [ ] Add span-level IoU or insight-text alignment in `compare_runs` (beyond quote counts).
