# Pipeline profiles

Two kinds of YAML live here — do not mix them.

## `config/profiles/*.yaml` — deployment profiles

Ready-to-use profiles you pass to the main CLI. Each describes **providers and
research-backed defaults only** — no feeds, no output paths, no per-run
operational fields. Supply those via CLI flags.

| Profile | Purpose | Provider shape |
| ------- | ------- | -------------- |
| [`cloud_balanced.yaml`](cloud_balanced.yaml) | Production default, best compound score | Gemini summary, spaCy trf NER, Whisper API |
| [`cloud_quality.yaml`](cloud_quality.yaml) | Maximum cloud quality, ~2x cost | DeepSeek summary, spaCy trf NER, Whisper API |
| [`local.yaml`](local.yaml) | Fully local / privacy-first, $0 | Whisper small.en + spaCy trf + qwen3.5:9b bundled |
| [`airgapped.yaml`](airgapped.yaml) | No network, no Ollama | Whisper medium.en + spaCy trf + SummLlama3.2-3B |
| [`dev.yaml`](dev.yaml) | Fastest/cheapest, CI-friendly, no GI/KG | Whisper tiny.en + spaCy sm + bart-led |

**Usage:**

```bash
python -m podcast_scraper.cli \
  --config config/profiles/cloud_balanced.yaml \
  --rss <feed_url> \
  --output-dir <output>
```

All numbers and picks trace back to
[`docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md`](../../docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md)
→ **Autoresearch-derived defaults** section and
[`docs/guides/eval-reports/`](../../docs/guides/eval-reports/).

## `config/profiles/audio/*.yaml` — audio preprocessing presets

Named bundles of audio-preprocessing fields (bitrate, sample rate, silence
trim, target loudness). Deployment profiles reference them via the
`audio_preprocessing_profile` field so one edit to a preset updates every
profile that uses it.

All 5 deployment profiles above reference
[`audio/speech_optimal_v1.yaml`](audio/speech_optimal_v1.yaml) (the default
preset, Whisper-tuned). See [`audio/README.md`](audio/README.md) for the full
workflow and merge-precedence rules.

**Orthogonal to `ml_preprocessing_profile`** (ML-only text cleaning like
`cleaning_v4`). Audio preprocessing and text cleaning are different pipeline
stages; both are now user-configurable from deployment profiles (#634).

## `config/profiles/freeze/*.yaml` — performance capture profiles

Companion profiles used by `make profile-freeze` (RFC-064) to capture
per-provider timing and cost profiles under a fixed E2E fixture. Each freeze
profile is **maximally oriented toward its provider** (filename = provider)
and is merged with [`freeze/_defaults.yaml`](freeze/_defaults.yaml) at
run time to supply operational fields.

See [`freeze/README.md`](freeze/README.md) for the workflow and matrix.

## Acceptance runner matrix (`config/acceptance/`)

Cross-link only: the acceptance runner merges
[`MAIN_ACCEPTANCE_CONFIG.yaml`](../acceptance/MAIN_ACCEPTANCE_CONFIG.yaml) with YAML under
[`fragments/`](../acceptance/fragments/). That layout is independent of
`make profile-freeze`; see [`acceptance/README.md`](../acceptance/README.md)
for stems, env vars, and CI usage.

## Legacy

[`profile_freeze.example.yaml`](profile_freeze.example.yaml) is a stub from
pre-2.6; superseded by the files in [`freeze/`](freeze/). Kept for one cycle
so older docs that link it still resolve.
