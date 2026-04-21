# Installation guide

**Canonical install paths** live in the repository **[README.md](https://github.com/chipi/podcast_scraper/blob/main/README.md)** (Quick Start,
pip / pipx / uv, extras, `.env`, verification).

**First pipeline run after install:** use a **named profile**, an **operator YAML**, and a **feed
list** together — same as [README.md — Typical run](https://github.com/chipi/podcast_scraper/blob/main/README.md#typical-run-profile-operator-config--feed-list):

```bash
python -m podcast_scraper.cli \
  --profile cloud_balanced \
  --config config/manual/operator_defaults.yaml \
  --feeds-spec config/manual/feeds.spec.registry_10.yaml
```

- **`--profile`** — Preset under `config/profiles/<name>.yaml` (merged as defaults).
- **`--config`** — Operator file (`output_dir`, `max_episodes`, flags, …); keys override the profile.
- **Feeds** — **`--feeds-spec`** (structured `{ feeds: [...] }`), **`--rss-file`** (one URL per line),
  or a URL on the command line / **`--rss`**. Do not mix `--feeds-spec` with `--rss-file` or explicit
  RSS URLs in one invocation.

CLI details: [CLI.md — Quick Start](../api/CLI.md#quick-start), [CLI.md — RSS and multi-feed](../api/CLI.md#rss-and-multi-feed). Config semantics: [CONFIGURATION.md — Multi-feed compose](../api/CONFIGURATION.md#multi-feed-compose).
