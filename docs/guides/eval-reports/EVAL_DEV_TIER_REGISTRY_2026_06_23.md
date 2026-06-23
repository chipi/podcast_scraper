# EVAL — Dev / airgapped / airgapped_thin stage options on smoke_v2 (#1060)

**Issue:** #1060 (promote YAML-only profiles to ProfilePreset)
**Branch:** `feat/rfc097-followups`
**Dataset:** `curated_5feeds_smoke_v2` (5 episodes, ~9 min each, TTS synthetic)
**Silver reference:** `silver_sonnet46_smoke_v2` (paragraph)
**Device:** Apple M4 Pro (laptop) — CPU for Whisper, MPS for transformers
**Status:** Done. Numbers below back the four new `StageOption` entries
landed alongside this report.

## TL;DR

Four `StageOption` rows shipped to `model_registry.py`:

| Stage | option_id | Headline |
| --- | --- | --- |
| transcription | `local_whisper_tiny_en` | mean WER 21.7%, 8.7 s/ep on M4 Pro CPU — dev/CI floor |
| transcription | `local_whisper_medium_en` | mean WER 13.2%, 83.9 s/ep on M4 Pro CPU — airgapped quality default |
| summary | `summllama_3_2_3b_paragraph` | ROUGE-L 0.251 / cosine 0.823 / 53 s/ep on MPS — airgapped paragraph quality |
| summary | `transformers_bart_small_long_fast_authority` | ROUGE-L 0.150 / cosine 0.655 / 18 s/ep on MPS — dev/airgapped-thin floor |

These unblock `airgapped`, `airgapped_thin`, and `dev` to opt into
`_PROFILE_PRESETS`. `test_default` stays YAML-only by design (it pins
MODEL across vendors without pinning PROVIDER — incompatible with the
6-tuple preset shape).

## Methodology

### Transcription (Whisper local)

Laptop (M4 Pro CPU):

```bash
.venv/bin/python scripts/eval/experiment/transcription_sweep.py \
  --audio-dir tests/fixtures/audio/v2 \
  --reference-dir data/eval/materialized/curated_5feeds_smoke_v2 \
  --episodes p01_e01,p02_e01,p03_e01,p04_e01,p05_e01 \
  --models <tiny.en|medium.en> \
  --clean-reference
```

DGX GB10 CUDA (via `dgx-llm-1` Tailscale + `podcast-whisper:0.1.0` image):

```bash
ssh dgx-llm-1 "sudo docker run --rm --gpus all \
    -v /tmp/wer_bench:/work -w /work podcast-whisper:0.1.0 \
    python transcription_sweep.py \
      --audio-dir audio --reference-dir refs \
      --episodes p01_e01,p02_e01,p03_e01,p04_e01,p05_e01 \
      --models <tiny.en|medium.en> --clean-reference"
```

Reference: the synthetic source transcripts that the audio was generated
from (TTS). WER is computed token-level vs the reference; the
`--clean-reference` flag (FU4, 2026-06-23) strips markdown headers,
speaker labels (`Maya:`, `Liam:`) and bracketed timestamps (`[00:00]`)
from the reference before comparison. Whisper does not emit any of
these, so the dirty reference inflates WER by a near-constant delta;
the clean-reference numbers are what the StageOption strings cite.

**For provenance against the very first registry write, the dirty
numbers are included in the results table below**; they should not be
used for new comparisons.

### Summarization

SummLlama (MPS, paragraph, max_new_tokens=600, do_sample=False):

```bash
.venv/bin/python scripts/eval/experiment/run_summllama_v2.py \
  --dataset curated_5feeds_smoke_v2 \
  --run-id summllama32_smoke_v2_paragraph_2026_06_23 \
  --style paragraph
```

bart-small + long-fast (ml_small_authority mode, full map-reduce):

```bash
PYTHONPATH=. .venv/bin/python scripts/eval/experiment/run_experiment.py \
  data/eval/configs/ml/baseline_ml_small_authority_smoke_v2.yaml \
  --reference silver_sonnet46_smoke_v2 \
  --dry-run
```

Both scored with the in-tree `score_run()`:

```python
from podcast_scraper.evaluation.scorer import score_run
score_run(
    predictions_path=run_dir / "predictions.jsonl",
    dataset_id="curated_5feeds_smoke_v2",
    run_id=run_id,
    reference_paths={"silver_sonnet46_smoke_v2": ref_dir},
    task="summarization",
)
```

ROUGE-L + cosine reported; judges intentionally not run (judge calls
would hit paid LLMs — violates the rule that benchmark-grade numbers go
through the structured score path, not ad-hoc judge passes). The
relative ranking from ROUGE+cosine is sufficient to back the
`headline_metric` field for these dev-tier options.

## Results — transcription

**Two measurement modes** for each model: the original (dirty reference)
numbers documenting the first-pass result, plus a clean-reference re-run
(#1060 FU4) that strips markdown headers + speaker labels + timestamps so
absolute WER is comparable to the existing `local_whisper_small_en` 2.9%
baseline. The clean-reference numbers are the ones cited in the
`StageOption.headline_metric` strings.

| Model | Device | Reference | mean WER | mean time/ep | Run artifact |
| --- | --- | --- | --- | --- | --- |
| tiny.en | M4 Pro CPU | dirty | 21.7% | 8.7 s | `data/eval/runs/_transcription_sweep/sweep_20260623_175255.json` |
| tiny.en | M4 Pro CPU | clean | 17.2% | 9.8 s | `data/eval/runs/_transcription_sweep/sweep_20260623_210655.json` |
| tiny.en | DGX GB10 CUDA | clean | 16.0% | 5.4 s | `data/eval/runs/_transcription_sweep/sweep_20260623_190759.json` |
| medium.en | M4 Pro CPU | dirty | 13.2% | 83.9 s | `data/eval/runs/_transcription_sweep/sweep_20260623_180021.json` |
| medium.en | M4 Pro CPU | clean | 8.1% | 82.7 s | `data/eval/runs/_transcription_sweep/sweep_20260623_211413.json` |
| medium.en | DGX GB10 CUDA | clean | 8.1% | 34.3 s | `data/eval/runs/_transcription_sweep/sweep_20260623_191158.json` |

**Clean-reference delta vs dirty**: tiny.en −4.5pp, medium.en −5.1pp. The
delta is roughly constant per model (each gives back the artifact
fraction it can't reconstruct: headers, speaker labels, timestamps). Clean
WER is what to cite externally; the dirty numbers are kept here for
provenance against the first registry write.

**Comparison to `local_whisper_small_en` (registry baseline 2.9% on
v2 fixtures, 2026-06-13)**: medium.en 8.1% on smoke_v2 is higher. Two
candidate explanations: (1) smoke_v2 is a TTS-synthetic corpus with
unusual phrasing ("Cascadia Alliance", "Singletrack Sessions") that
small.en's training distribution covers better than medium.en's
generalization predicts; (2) the small.en baseline was measured on a
different v2 fixture (the EVAL_WHISPER_SMALL_EN_2026_06_13 report uses
the held-out v2 set, not smoke_v2). Either way the smoke_v2 measurements
are internally consistent (tiny > medium ordering preserved); the
absolute numbers are smoke_v2-specific.

### Per-episode WER spread (clean reference)

```text
tiny.en  CPU  p01=13.7% p02=12.1% p03=14.8% p04=28.1% p05=17.2%
tiny.en  DGX  p01=13.7% p02=12.1% p03=14.8% p04=22.2% p05=17.2%
medium   CPU  p01=8.8%  p02=5.2%  p03=8.2%  p04=5.0%  p05=13.4%
medium   DGX  p01=11.4% p02=5.1%  p03=8.2%  p04=5.0%  p05=13.4%
```

### Latency portability — laptop CPU vs DGX GB10 CUDA (#1060 FU5)

| Model | CPU s/ep | CUDA s/ep | Speedup | Notes |
| --- | --- | --- | --- | --- |
| tiny.en | 9.8 | 5.4 | 1.8× | Tiny model — speedup capped by container/Docker overhead and model-load amortization |
| medium.en | 82.7 | 34.3 | 2.4× | Bigger model — bigger CUDA win; medium.en on CPU is the dominant wallclock cost in airgapped |

Profiles deploying these StageOptions can plan for either device. DGX
adds substantial speedup for `medium.en` (the airgapped quality default);
for `tiny.en` the CPU number is already fast enough that DGX deployment
buys mostly nothing.

**Summary-stage StageOptions intentionally not DGX-benched**: `summllama_3_2_3b_paragraph`
and `transformers_bart_small_long_fast_authority` are laptop-class options
by design (SummLlama is the airgapped-CPU summary, bart-small+long-fast
is the dev/CI floor). Operators with GPU headroom pick a larger Ollama or
vLLM model — the LocalScope DGX number for these two options would be
artificial. The M4 Pro MPS numbers stand as their portability anchor.

## Results — summarization

| Model | ROUGE-L | ROUGE-1 | Cosine | Coverage | Mean time/ep | Run artifact |
| --- | --- | --- | --- | --- | --- | --- |
| SummLlama 3.2-3B (paragraph) | 0.2505 | 0.4985 | 0.8226 | 1.072 | 53.3 s | `data/eval/runs/summllama32_smoke_v2_paragraph_2026_06_23/` |
| bart-small + long-fast (ml_small_authority) | 0.1500 | 0.3112 | 0.6548 | 0.385 | 18.3 s | `data/eval/runs/baseline_ml_small_authority_smoke_v2/` |

**Trade observed**: SummLlama wins ROUGE-L by +67%, cosine by +25%,
coverage by +179%, at 2.9× the latency of bart-small+long-fast. SummLlama
is the right floor for `airgapped` (paragraph summary as the user-visible
artifact); bart-small+long-fast is the right floor for `airgapped_thin`
and `dev` (summary feeds downstream extraction, latency budget tighter).

## What this report does NOT establish

- Judge-graded quality (subjective rubric). The numbers above are
  rouge/cosine only — sufficient for `headline_metric` framing but not
  for "best paragraph summary on smoke_v2 overall" rankings against
  judge-evaluated cohorts in `EVAL_HELDOUT_V2` or `EVAL_1016_*`.
- ~~Absolute transcription WER under typical pipeline cleaning (markdown +
  speaker-label stripping not applied here).~~ Addressed by FU4 (the
  `--clean-reference` flag); the clean-reference numbers in the table are
  the comparable yardstick.
- Performance on longer-than-9-min content. The smoke_v2 corpus is
  curated at ~9 min/ep; latency at 60-min content is ~6× these numbers
  for both transcription and summary.

These gaps are acceptable for the dev/airgapped/airgapped_thin tier the
options serve. None of these options is the production pipeline default
for any cloud or DGX profile.

## Provenance

- Dataset: `data/eval/datasets/curated_5feeds_smoke_v2.json`
- Materialized transcripts: `data/eval/materialized/curated_5feeds_smoke_v2/`
- Audio: `tests/fixtures/audio/v2/p0*_e01.mp3`
- Silver: `data/eval/references/silver/silver_sonnet46_smoke_v2/`
- Whisper local cache: `~/.cache/whisper/{tiny,medium}.en.pt`
- SummLlama HF cache: `~/.cache/huggingface/hub/models--DISLab--SummLlama3.2-3B/`
