# Performance Report: E2E WIP v1 (April 2026)

> **Informal RFC-064 sweep** — eleven frozen profiles on the same host, E2E fixture
> `podcast1_mtb`, two episodes per run. Intended to exercise `profile-freeze` presets
> and `profile-diff`, not as a release baseline.

| Field | Value |
| ----- | ----- |
| **Date** | April 2026 |
| **Dataset label** | `e2e_podcast1_mtb_n2` (2 episodes, `podcast1_mtb`) |
| **RSS** | Mock (`E2EHTTPServer`); configs under `config/profiles/capture_e2e_*.yaml` |
| **Warm-up** | **`SKIP_WARMUP=1`** for all runs below (faster WIP; not release-standard) |
| **Host** | `Markos-MacBook-Pro.local` (from profile YAML; your numbers will differ on other machines) |
| **Artifacts** | `data/profiles/v2.6-wip-*.yaml` in repo when committed |
| **Methodology** | [Performance reports index](index.md), [Performance Profile Guide](../PERFORMANCE_PROFILE_GUIDE.md) |

---

## Totals (headline)

Sorted by **`totals.wall_time_s`** ascending. **MB** = `totals.peak_rss_mb`.

| Release tag | Config (preset) | Wall (s) | Peak RSS (MB) | s/episode |
| ----------- | --------------- | -------- | ------------- | --------- |
| `v2.6-wip-gemini` | `capture_e2e_gemini.yaml` | 7.89 | 1134 | 3.94 |
| `v2.6-wip-anthropic` | `capture_e2e_anthropic.yaml` | 12.23 | 1068 | 6.12 |
| `v2.6-wip-mistral` | `capture_e2e_mistral.yaml` | 17.23 | 1149 | 8.61 |
| `v2.6-wip-deepseek` | `capture_e2e_deepseek.yaml` | 44.86 | 1109 | 22.43 |
| `v2.6-wip-ollama-llama32` | `capture_e2e_ollama_llama32.yaml` | 52.87 | 607 | 26.44 |
| `v2.6-wip-grok` | `capture_e2e_grok.yaml` | 92.62 | 1144 | 46.31 |
| `v2.6-wip-openai` | `capture_e2e_openai.yaml` | 95.38 | 617 | 47.69 |
| `v2.6-wip-ollama-llama31` | `capture_e2e_ollama_llama31_8b.yaml` | 96.65 | 1056 | 48.33 |
| `v2.6-wip-ml-dev` | `capture_e2e_ml_dev.yaml` | 108.12 | 3282 | 54.06 |
| `v2.6-wip-ml-prod` | `capture_e2e_ml_prod.yaml` | 111.99 | 7159 | 56.00 |
| `v2.6-wip-ollama-qwen35` | `capture_e2e_ollama_qwen35.yaml` | 123.57 | 995 | 61.79 |

**Rounding:** wall times shown to two decimals; YAML may carry more precision.

---

## Pairs we diffed (same host)

Illustrative **`make profile-diff`** comparisons from this campaign:

| From | To | Note |
| ---- | -- | ---- |
| `v2.6-wip-openai` | `v2.6-wip-anthropic` | Different API stacks; transcript cache on both |
| `v2.6-wip-ml-dev` | `v2.6-wip-ml-prod` | Prod stack much higher peak RSS (~+118% in-table delta) |
| `v2.6-wip-ollama-qwen35` | `v2.6-wip-ollama-llama32` | Smaller Ollama model faster, lower RSS |
| `v2.6-wip-gemini` | `v2.6-wip-grok` | Very different wall totals; check parallelism / stage attribution |
| `v2.6-wip-grok` | `v2.6-wip-mistral` | Grok preset uses `grok-3-mini` (x.ai model IDs evolve) |
| `v2.6-wip-mistral` | `v2.6-wip-deepseek` | Two “budget cloud text” stacks (`mistral-large-latest` vs `deepseek-chat`) |
| `v2.6-wip-ollama-llama31` | `v2.6-wip-ollama-llama32` | Guide “privacy default” 8B vs 3B |

Re-run `make profile-diff FROM=… TO=…` locally to reproduce the Rich tables.

---

## Caveats (read before drawing product conclusions)

1. **WIP / non-release** — `SKIP_WARMUP=1` skips the default cold-start scrub; official
   release captures should omit it unless debugging.
2. **Not apples-to-apples across rows** — presets differ (e.g. OpenAI uses API
   transcription; Anthropic uses Whisper + API text; ML prod loads Pegasus + `trf`).
3. **Transcript cache** — repeated runs hit disk cache; **transcription** stage may be
   absent or near-zero in YAML even though episodes were processed.
4. **Stage RSS/CPU** — proportional sampling; short stages may show **0** MB or **0%**
   CPU. See [Interpreting the profile](../PERFORMANCE_PROFILE_GUIDE.md#interpreting-the-profile).
5. **Grok models** — `capture_e2e_grok.yaml` uses **`grok-3-mini`** for summary and
   cleaning because **`grok-2` / `grok-beta` returned 400** on the capture API at the time.

---

## Related

- [Performance reports index](index.md)
- [RFC-064](../../rfc/RFC-064-performance-profiling-release-freeze.md)
- [`config/profiles/README.md`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/README.md)
