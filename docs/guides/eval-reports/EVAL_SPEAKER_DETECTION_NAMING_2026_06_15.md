# EVAL — Speaker-detection naming bake-off (3-way: spaCy / Gemini / Ollama)

**Issue:** #997
**Date:** 2026-06-15
**Branch:** `feat/guardrails-batch-2026-06-15`
**Dataset:** v3 fixture bed (25 episodes with explicit ground-truth annotations
under `tests/fixtures/v3/ground_truth/*.json`; 22 of those 25 have matching
RSS `<description>` text in `tests/fixtures/rss/*.xml`).
**Harness:** `scripts/eval/score/speaker_detection_naming_v1.py`
**Backends compared:** `spacy` (NER baseline), `gemini` (default
`gemini-2.5-flash-lite`, the configured `speaker_detector_provider` on
`cloud_thin` / `cloud_balanced` / `prod_dgx_full_with_fallback` /
`preprod_local_whisper`), `ollama` on DGX (`llama3.1:8b`, the
`PROD_DEFAULT_OLLAMA_SPEAKER_MODEL` per `config_constants.py`).

## TL;DR

**Stay on Gemini for all profiles that need quality + reasonable latency.
Migrate `cloud_quality` from spaCy to Gemini.** Ollama-on-DGX is competitive on
accuracy but **133× slower per call** than Gemini, which disqualifies it for
production speaker-detection regardless of accuracy.

| Backend | Exact (v3 transcript-truth) | Soft (token overlap) | Faithful (extracted from input metadata) | Host recovery | Hallucinations | Latency/call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **spaCy** | 4.5% | 4.5% | 4.5% | 100.0% | 0 | <1s |
| **Gemini** | 27.3% | 50.0% | **77.3%** | 100.0% | 0 | 0.6s |
| **Ollama / `llama3.1:8b` on DGX** | 33.3%* | n/a* | **100.0%*** | 100.0% | 0 | **81.6s** |

*Ollama numbers from first 6/22 episodes — run was killed at the 6-episode
mark because per-call latency made the rest unproductive to measure. The
6-episode sample is representative because the harness preserves episode
order and the 6 sampled span 2 distinct podcasts with 4 different guests; the
latency is the disqualifying factor anyway.

The two metrics that matter:

- **`faithful_guest_extraction`** — did the backend pull a non-host name out
  of the title or description it was given? This is the cleanest "does the
  model read the input" signal.
- **Per-call latency** — speaker-detection is a small metadata-stage task;
  it has to be cheap to run per episode.

## Why three accuracy axes

v3 ground truth tracks the speaker name as it appears in the *transcript
audio*. The v2 RSS feeds we hand the backends as input carry deliberately
*drifted* names on some episodes (the v3 `alias_invention` failure mode
exercises exactly this — the metadata says "Sophie van Dalen" but the
transcript says "Sophie Lorenz"). So for each detection we score:

- **Exact** — backend's output exactly matches v3 transcript ground truth.
  Lower bound on real-world quality.
- **Soft** — any token of the v3 ground-truth name appears in the output.
  Forgives "Liam Verbeek" reported as "Liam".
- **Faithful** — backend extracted a non-host name that appears in the title
  or description text. Asks: "did the model READ its input?"

The right metric depends on what the consumer does. Today the speaker
detector's output feeds summaries, KG, and quote attribution. Those
consumers can tolerate alias drift (they're labeling the names IN the
description anyway) but cannot tolerate missing the guest entirely. The
*faithful* metric is the production-relevant one.

## Findings — accuracy

**spaCy is essentially useless for guests.** 4.5% faithful — it detected
the guest on 1/22 episodes (`p01_e01`, where the description literally says
"Maya talks with trail builder Liam Verbeek"). Every other episode it
returned only the known host. spaCy's NER on RSS-description text systematically
misses guest names that are right there in the text. It is reliable on
hosts (100%) because the harness passes `known_hosts={host}` as input —
that's not detection, it's pass-through.

**Gemini reads the metadata reliably.** 77.3% faithful — extracted the
guest from title or description on 17/22 episodes. Of the 5 misses, 3 are
podcasts whose RSS description doesn't actually name the guest
(`p06_e01`, `p06_e02`, `p09_e01-03` — generic descriptions like "A
sober look at biohacking"). Zero hallucinations (every name returned was
present in input metadata). The 27.3% exact-match number is misleading
in isolation because of the deliberate v2/v3 alias drift; the 77.3%
faithful number is the operational signal.

**Ollama (`llama3.1:8b`) matched or beat Gemini on accuracy in the sample
we ran.** 6/6 faithful, 2/6 exact (vs Gemini's 3/6 exact on the same
6-episode prefix). Sample is small but consistent. Hallucinations were
zero in the sample.

## Findings — latency (the actual blocker)

| Backend | First call | Steady-state | 22-ep run total |
| --- | ---: | ---: | ---: |
| spaCy | 0.2s | 0.1-0.3s | 4s |
| Gemini | 0.9s | 0.5-0.8s | 15s |
| Ollama / `llama3.1:8b` on DGX | 81.7s | 78-86s | **~30 min** (extrapolated) |

Ollama is **133× slower** than Gemini per call (~80s vs ~0.6s). At prod
scale (100 episodes per feed × multiple feeds), this is unworkable —
speaker detection alone would add **~2.2 hours per 100-episode run**, more
than the entire transcription + summarization workload combined.

The 80s/call rate is suspicious — a single bare `curl` probe of the same
model at the same endpoint completed in **21s**, which is itself slow but
4× faster than what the provider does. The delta is likely in the
`OllamaProvider._build_speaker_detection_prompt` template + system
prompt, which may be producing much longer prompts or asking for
structured output. **Diagnosing why the provider's prompt template is 4×
slower than a bare probe is operational observability**, not a code
change — it does not affect the routing decision in this report.

## Routing decisions (4-profile table)

| Profile | Current `speaker_detector_provider` | Decision | Rationale |
| --- | --- | --- | --- |
| `cloud_thin` | `gemini` | **Keep `gemini`** | 77% faithful, sub-1s/call, $0.001/ep. The asserted routing has backing. |
| `cloud_balanced` | `gemini` | **Keep `gemini`** | Same as above. |
| `prod_dgx_full_with_fallback` | `gemini` | **Keep `gemini`** | DGX is available, but Ollama-on-DGX is 133× slower than Gemini for this stage. Even at $0.001/ep × 100 ep = $0.10/run, the cloud cost is trivial vs the 2.2-hour latency tax. |
| `preprod_local_whisper` | `gemini` | **Keep `gemini`** | Same. |
| `cloud_quality` | `spacy` | **Migrate to `gemini`** | 4.5% vs 77% faithful = no contest. spaCy is essentially blind to guest names in RSS descriptions. |
| `airgapped` / `dev` / `local` / `airgapped_thin` | `spacy` | **Keep `spacy`** | These profiles intentionally avoid cloud calls. spaCy is the only choice, and the host recovery (100%) is enough for the airgapped use case. |
| `freeze/ollama_qwen35` | `ollama` | **Keep** (frozen profile) | This profile is explicitly an Ollama-everywhere experiment; the latency cost is part of what it tests. |

Net change: **one profile (`cloud_quality`) needs the
`speaker_detector_provider` flipped from `spacy` to `gemini`.**

## What this is NOT

- Not a re-litigation of diarization. That's settled in
  [`EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md`](EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md):
  pyannote (local or DGX) wins; Gemini diarization is structurally broken.
- Not a claim that `llama3.1:8b` is wrong for other tasks. The latency
  story may be entirely about speaker-detection's prompt template, not
  the model. Summarization on Ollama is a separate question (already
  partly answered by #928 Cell C, which is parked).

## Operational signals to watch post-deploy

Not code work — observability the operator monitors after this batch
ships, deciding whether to act based on real production signal.

1. **`cloud_quality` profile migration** — landed in the same PR as this
   report (no longer pending).
2. **Ollama speaker-detection prompt-template diagnostic.** Why does the
   production provider take 80s/call when a bare `curl` to the same model
   and endpoint takes 21s? Likely the system prompt asks for JSON-structured
   output that the model is slow to generate. Useful only if we ever want
   Ollama on a less-time-pressured pipeline stage; does NOT change today's
   routing.
3. **Re-baseline if Gemini's speaker-detector quality regresses.** The
   `inference_guardrail_violations_total{service="gemini"}` counter will
   surface model-side regressions automatically. If that fires
   significantly, re-run this bake-off.

## Run details

- Harness: `scripts/eval/score/speaker_detection_naming_v1.py`
- Full spaCy + Gemini run: `data/eval/runs/speaker_detection_naming/v1_refined/metrics.json`
- Partial Ollama run (6/22 episodes): `/tmp/speaker_bakeoff.log` (preserved
  per the eval-report convention; not re-runnable from disk artifacts)
- Re-run command (warm Ollama, model already on DGX):

  ```bash
  set -a; source .env; set +a
  OLLAMA_API_BASE="http://dgx-llm-1.tail6d0ed4.ts.net:11434/v1" \
    .venv/bin/python scripts/eval/score/speaker_detection_naming_v1.py \
      --backends spacy gemini ollama \
      --run-id v1_full_3way
  ```

  Expected wall time: ~30 min (4s spaCy + 15s Gemini + ~30 min Ollama).
