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

## `config/profiles/freeze/*.yaml` — performance capture profiles

Companion profiles used by `make profile-freeze` (RFC-064) to capture
per-provider timing and cost profiles under a fixed E2E fixture. Each freeze
profile is **maximally oriented toward its provider** (filename = provider)
and is merged with [`freeze/_defaults.yaml`](freeze/_defaults.yaml) at
run time to supply operational fields.

See [`freeze/README.md`](freeze/README.md) for the workflow and matrix.

## Legacy

[`profile_freeze.example.yaml`](profile_freeze.example.yaml) is a stub from
pre-2.6; superseded by the files in [`freeze/`](freeze/). Kept for one cycle
so older docs that link it still resolve.
