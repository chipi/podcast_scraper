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
| [`cloud_thin.yaml`](cloud_thin.yaml) | Minimal cloud stack for Compose | Gemini + OpenAI Whisper API |
| [`local.yaml`](local.yaml) | Fully local / privacy-first, $0 | Whisper small.en + spaCy trf + qwen3.5:9b bundled |
| [`local_dgx_balanced.yaml`](local_dgx_balanced.yaml) | Local DGX workstation | Whisper + spaCy trf + Ollama bundled + DGX pyannote |
| [`local_dgx_full.yaml`](local_dgx_full.yaml) | Local DGX full ML stack | Whisper + spaCy trf + hybrid ML + DGX pyannote |
| [`cloud_with_dgx_primary.yaml`](cloud_with_dgx_primary.yaml) | Prod: DGX Whisper + pyannote, cloud fallback | `tailnet_dgx_whisper` + `tailnet_dgx` diarize + Gemini summary ([ADR-096](../../docs/adr/ADR-096-dgx-spark-prod-primary-with-fallback.md), [#926](https://github.com/chipi/podcast_scraper/issues/926)) |
| [`airgapped.yaml`](airgapped.yaml) | No network, no Ollama | Whisper medium.en + spaCy trf + SummLlama3.2-3B |
| [`airgapped_thin.yaml`](airgapped_thin.yaml) | Minimal airgapped | Whisper tiny.en + spaCy sm |
| [`dev.yaml`](dev.yaml) | Fastest/cheapest, CI-friendly, no GI/KG | Whisper tiny.en + spaCy sm + bart-led |

### Registry status — two first-class shapes

Profile YAMLs come in two flavours; both are valid, neither is a smell.

1. **Preset-backed** — the YAML declares `profile: <name>` and matches a
   `ProfilePreset` in
   [`src/podcast_scraper/providers/ml/model_registry.py`](../../src/podcast_scraper/providers/ml/model_registry.py).
   Use this when every stage choice has a research-backed `StageOption` and the
   profile pins both PROVIDER and MODEL for every stage. Drift between the YAML
   and the preset is caught by
   [`test_profile_yaml_registry_drift.py`](../../tests/integration/providers/ml/test_profile_yaml_registry_drift.py).
2. **YAML-only** — the YAML carries no `profile:` field. The drift test
   skips it. Two legitimate reasons:
   - **Structural** — the profile only pins MODEL across vendors, not PROVIDER
     (e.g. `test_default` lets Field defaults pick the provider per code path).
     The 6-tuple `ProfilePreset` shape doesn't represent that.
   - **Tactical** — the profile is a transient mirror or pre-promotion stub
     (e.g. `preprod_local_whisper` mirrors `cloud_with_dgx_primary` with one
     stage swap).

   YAML-only profiles MUST carry a `Registry status: YAML-only` comment
   explaining why; the drift test asserts this. Closure as YAML-only is the
   resolution path when the structural mismatch is the right answer
   (#1060 Decision 1).

### Screenplay and diarization (Audio Wave 2)

Profiles that use **local Whisper** (`transcription_provider: whisper` or
`tailnet_dgx_whisper`) set **`screenplay: true`** and **`diarize: true`** by default
([RFC-058](../../docs/rfc/RFC-058-audio-speaker-diarization.md)). Cloud API transcription
profiles omit these keys — config validation coerces both off for OpenAI / Gemini / Mistral /
Deepgram.

Neural diarization requires **`HF_TOKEN`** (or `hf_token` in config) and `pip install -e ".[ml]"`.
Disable with `diarize: false` or CLI **`--no-diarize`**. See
[Audio Pipeline Guide](../../docs/guides/AUDIO_PIPELINE_GUIDE.md).

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

Local-whisper freeze profiles also set **`screenplay: true`** and **`diarize: true`**.

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
