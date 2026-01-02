# Local ML Model Cache

This directory contains cached ML models for local development and testing.

## Structure

- `.cache/whisper/` - Whisper transcription models (e.g., `tiny.en.pt`, `base.en.pt`)
- `.cache/huggingface/hub/` - HuggingFace Transformers models (e.g., `facebook/bart-base`, `allenai/led-base-16384`)
- `.cache/spacy/` - spaCy NER models (if using local cache instead of package installation)

## Usage

The codebase automatically prefers models in this local cache directory over the standard user cache
locations (`~/.cache/whisper/`, `~/.cache/huggingface/hub/`).

## Preloading Models

To populate this cache, run:

```bash
make preload-ml-models
```

This will download and cache the test default models:

- Whisper: `tiny.en` (~72MB)
- Transformers MAP: `facebook/bart-base` (~500MB)
- Transformers REDUCE: `allenai/led-base-16384` (~1GB)

## Git Ignore

By default, `.cache/` is **not** ignored in `.gitignore` so you can optionally commit models to share
with your team. If you prefer to keep models out of git, uncomment the `.cache/` line in `.gitignore`.

## Benefits

- **Consistent cache location**: Models are stored in the project directory
- **Team sharing**: Models can be committed to git (if desired)
- **Faster CI**: CI can use the same cache structure
- **Easier debugging**: Cache location is predictable and visible
