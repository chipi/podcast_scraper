# Audio preprocessing presets (GitHub #634)

Named audio-preprocessing bundles. Referenced by deployment profiles via the
`audio_preprocessing_profile` field; merged under the deployment profile so
one edit here updates every profile that references the preset.

## How it works

`Config` has a `@model_validator(mode="before")` (`_resolve_audio_preprocessing_profile`)
that runs right after the main deployment-profile resolver. When it sees
`audio_preprocessing_profile: speech_optimal_v1`, it loads
`config/profiles/audio/speech_optimal_v1.yaml` and merges its fields under the
deployment profile — explicit deployment profile fields still win on overlap.

**Merge order (low → high priority):**

1. `Config` field defaults
2. `audio/<name>.yaml` preset values
3. Deployment profile YAML fields (`cloud_balanced.yaml`, `local.yaml`, ...)
4. Explicit `Config(...)` kwargs / CLI args

## Available presets

| Preset | Purpose | Notes |
| ------ | ------- | ----- |
| [`speech_optimal_v1.yaml`](speech_optimal_v1.yaml) | Default for all deployment profiles | 32 kbps / 16 kHz / mono, silence trim at current defaults, -16 LUFS. Tuned for Whisper / LLM transcription. |

## Adding a preset

1. Drop a new file `config/profiles/audio/<name>.yaml` with only
   `preprocessing_*` fields (no provider selection, no runtime state).
2. Reference it from a deployment profile:

   ```yaml
   # cloud_balanced.yaml
   audio_preprocessing_profile: <name>
   ```

3. Document the data backing the choice (link to eval-reports / experiment
   run IDs) in a comment block at the top.

## When to add variants

- New eval surfaces a better silence config → bump to `speech_optimal_v2.yaml`
  so the old preset remains reproducible.
- Hardware constraint (e.g. airgapped storage) requires a different bitrate →
  add a named variant (e.g. `archival_high_quality.yaml`).
- Product-specific need (e.g. music-heavy shows without aggressive silence
  trim) → add a variant instead of editing the shared preset.

## Related

- Audio preprocessing core: `src/podcast_scraper/preprocessing/audio/ffmpeg_processor.py`
- Text preprocessing (separate axis, ML-only): see
  `ml_preprocessing_profile` on `Config` and the cleaning registry at
  `src/podcast_scraper/preprocessing/profiles.py`
- Research backing: `docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md` →
  *Autoresearch-derived defaults (2026-04)*
