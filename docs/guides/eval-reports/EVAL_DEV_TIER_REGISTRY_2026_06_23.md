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

```bash
.venv/bin/python scripts/eval/experiment/transcription_sweep.py \
  --audio-dir tests/fixtures/audio/v2 \
  --reference-dir data/eval/materialized/curated_5feeds_smoke_v2 \
  --episodes p01_e01,p02_e01,p03_e01,p04_e01,p05_e01 \
  --models <tiny.en|medium.en>
```

Reference: the synthetic source transcripts that the audio was generated
from (TTS). WER is computed token-level vs the cleaned reference text.

**Caveat — WER absolute level**: the reference includes the markdown
header (`# Singletrack Sessions...`), speaker labels (`Maya:`, `Liam:`),
and timestamps (`[00:00]`) — none of which a raw Whisper transcript
contains. This inflates absolute WER by a near-constant delta vs the
"true" content-WER, but the **relative ordering across models is
reliable** because every model is scored against the same reference. For
absolute laptop WER under cleaner conditions, see
`EVAL_TRANSCRIPTION_3WAY_2026_06.md` (large-v3 MPS, 0.096) and
`EVAL_WHISPER_SMALL_EN_2026_06_13.md` (small.en CPU, 0.029).

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

| Model | mean WER | mean time/ep | runtime ratio vs audio | Run artifact |
| --- | --- | --- | --- | --- |
| tiny.en | 21.7% | 8.7 s | ≈ 63× realtime (9 min audio → 8.7 s) | `data/eval/runs/_transcription_sweep/sweep_20260623_175255.json` |
| medium.en | 13.2% | 83.9 s | ≈ 6.6× realtime | `data/eval/runs/_transcription_sweep/sweep_20260623_180021.json` |

Per-episode WER spread:

```text
tiny.en   p01=18.2% p02=16.4% p03=19.7% p04=32.2% p05=22.0%
medium.en p01=13.6% p02=10.2% p03=13.4% p04=10.5% p05=18.4%
```

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
- Absolute transcription WER under typical pipeline cleaning (markdown +
  speaker-label stripping not applied here).
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
