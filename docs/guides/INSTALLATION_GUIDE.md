# Installation guide

**Canonical install paths** live in the repository **[README.md](https://github.com/chipi/podcast_scraper/blob/main/README.md)** (Quick Start,
pip / pipx / uv, extras, `.env`, verification).

**First pipeline run after install:** use a **named profile**, an **operator YAML**, and a **feed
list** together — same as [README.md — Typical run](https://github.com/chipi/podcast_scraper/blob/main/README.md#typical-run-profile-operator-config--feed-list):

```bash
python -m podcast_scraper.cli \
  --profile airgapped_thin \
  --config my_operator.yaml \
  --feeds-spec config/examples/feeds.spec.example.yaml
```

- **`--profile`** — Preset under `config/profiles/<name>.yaml` (merged as defaults). `airgapped_thin` runs local models only; `cloud_balanced` needs `DEEPGRAM_API_KEY` and a LiteLLM gateway (`LITELLM_API_BASE`).
- **`--config`** — Operator file (`output_dir`, `max_episodes`, flags, …); keys override the profile.
- **Feeds** — **`--feeds-spec`** (structured `{ feeds: [...] }`), **`--rss-file`** (one URL per line),
  or a URL on the command line / **`--rss`**. Do not mix `--feeds-spec` with `--rss-file` or explicit
  RSS URLs in one invocation.

CLI details: [CLI.md — Quick Start](../api/CLI.md#quick-start), [CLI.md — RSS and multi-feed](../api/CLI.md#rss-and-multi-feed). Config semantics: [CONFIGURATION.md — Multi-feed compose](../api/CONFIGURATION.md#multi-feed-compose).

## Optional: Deepgram

Neural speaker diarization is not part of this install: the `[ml]` extra carries no pyannote, and the diarization stage is provided by an installed extension. Without one the speaker stage does nothing.

**Deepgram transcription:**

```bash
pip install -e ".[llm]"   # includes deepgram-sdk
export DEEPGRAM_API_KEY=...
python -m podcast_scraper.cli --transcription-provider deepgram ...
```

Full narrative: [Audio Pipeline Guide](AUDIO_PIPELINE_GUIDE.md).
