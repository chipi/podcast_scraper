# Ollama Provider Guide

Complete guide for using the Ollama provider with podcast_scraper, including installation, setup, troubleshooting, and testing.

## Overview

Ollama is a **local, self-hosted LLM solution** that runs entirely on your machine. It provides:

- ✅ **Zero API costs** - All processing happens locally
- ✅ **Complete privacy** - Data never leaves your machine
- ✅ **Offline operation** - No internet required after setup
- ✅ **Unlimited usage** - No rate limits or quotas
- ⚠️ **No transcription support** - Use `whisper` or `openai` for transcription

## Installation

### Step 1: Install Ollama

**macOS (Homebrew):**

```bash
brew install ollama
```

**macOS (Direct Download):**

1. Download from [https://ollama.ai](https://ollama.ai)
2. Open the `.dmg` file and drag Ollama to Applications
3. Launch Ollama from Applications

**Linux:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**

1. Download from [https://ollama.ai](https://ollama.ai)
2. Run the installer
3. Ollama will start automatically

### Step 2: Start Ollama Server

**Option A: Manual Start (Recommended for Testing)**

```bash
# Start server in foreground (see logs)
ollama serve

# Or start in background
ollama serve &
```

**Option B: Service Mode (macOS)**

```bash
# Check if installed as service
brew services list | grep ollama

# Start as service
brew services start ollama

# Check status
brew services info ollama
```

**Verify Server is Running:**

```bash
# Test connection
curl http://localhost:11434/api/tags

# Should return JSON (empty list if no models yet)
```

### Step 3: Pull Required Models

```bash
# Best for structured JSON output and GIL extraction (recommended default)
ollama pull qwen2.5:7b

# Same family, larger — higher quality; needs much more RAM/VRAM (see table)
ollama pull qwen2.5:32b

# Qwen 3.5 (three-tier checklist in this guide + repository alignment)
# ollama pull qwen3.5:9b
# ollama pull qwen3.5:27b
# ollama pull qwen3.5:35b

# General purpose, good all-rounder (recommended default for speaker detection)
ollama pull llama3.1:8b

# Fast inference, good for speaker detection
ollama pull mistral:7b

# Larger Mistral-family tags (see table below)
ollama pull mistral-nemo:12b
ollama pull mistral-small3.2

# Balanced quality/speed for summarization
ollama pull gemma2:9b

# Lightweight for development/testing (lowest RAM requirement)
ollama pull phi3:mini

# Verify models are available
ollama list
```

**Recommended Models:**

| Model | Size | RAM Required | Speed | Quality | Best For | Use Case |
| ----- | ---- | ------------- | ----- | ------- | -------- | -------- |
| **Qwen 2.5 7B** | 4.4GB | 8GB+ | Medium | High | Structured JSON, GIL extraction | Summarization, GIL extraction (best JSON output) |
| **Qwen 2.5 32B** | ~19GB | 32GB+ | Slower | Higher | Same use cases as 7B, more capacity | When 7B quality is not enough and hardware allows |
| **Llama 3.1 8B** | 4.7GB | 8GB+ | Medium | High | General purpose | Speaker detection (default), summarization |
| **Mistral 7B** | 4.1GB | 8GB+ | Fast | Good | Fast inference | Speaker detection (fastest), summarization |
| **Mistral Nemo 12B** | varies | 16GB+ | Medium | Good | Larger Mistral instruct | Summarization when 7B is tight |
| **Mistral Small 3.2** | varies | 16GB+ | Medium–slow | Higher | Instruct / long-context class | Summarization; confirm tag on [Ollama library](https://ollama.com/library/mistral-small3.2) |
| **Gemma 2 9B** | 5.5GB | 12GB+ | Medium | High | Balanced quality/speed | Summarization (balanced) |
| **Phi-3 Mini** | 2.3GB | 4GB+ | Fast | Acceptable | Lightweight | Development, testing, low-resource |
| `llama3.1:8b` | ~4.7GB | 6-8GB | Fastest | Good | Limited RAM (6-8GB), testing, development | N/A |
| `llama3.2:latest` | ~4GB | 8-12GB | Fast | Good | Standard systems (8-12GB), testing, development | N/A |
| `llama3.3:latest` | ~4GB | 12-16GB | Medium | Better | Production (12-16GB recommended) | N/A |
| `llama3.3:70b` | ~40GB | 48GB+ | Slow | Best | High-quality production (48GB+ RAM) | N/A |

### Qwen 3.5 (Ollama): three-tier checklist

**Qwen 3.5** is a newer Ollama library family than **Qwen 2.5** (multimodal-capable tags; large context on many variants). **Smoke evals and docs still use Qwen 2.5 7B as the primary baseline**, but this repo **ships model-specific prompts, integration tests, and optional hybrid smoke YAML** for the three checklist tiers (see [Repository alignment](#qwen-35-repository-alignment) below). After `ollama pull`, always confirm summaries and any **structured JSON / GIL** outputs still parse. Tags and on-disk sizes change over time—see [Ollama: qwen3.5](https://ollama.com/library/qwen3.5/tags).

Use the **same config fields** as other Ollama models (`ollama_summary_model`, `ollama_speaker_model`, and/or `hybrid_reduce_model` with `hybrid_reduce_backend: ollama`). Model-specific prompts live under `src/podcast_scraper/prompts/ollama/<dir>/` where `<dir>` replaces `:` with `_` (e.g. `qwen3.5:9b` → `qwen3.5_9b`). If you use a variant tag not covered below (e.g. `qwen3.5:27b-q4_K_M`), the provider falls back to **generic** `ollama/ner/` and `ollama/summarization/` prompts unless you add a matching `<dir>`.

For **Qwen 3.5** tags, chat requests use Ollama’s OpenAI-compatible field **`reasoning_effort: none`** so the assistant answer is returned in **`content`** (not only chain-of-thought in **`reasoning`**, which can consume `max_tokens` and look like “empty summaries”). See [Ollama: Thinking](https://docs.ollama.com/capabilities/thinking).

#### Tier 1 — `qwen3.5:9b` (fast default for Qwen 3.5)

Roughly **~6–7 GB** on disk for the common quantized tag (Ollama default `latest` often tracks this class).

- [ ] `ollama pull qwen3.5:9b` (or `ollama pull qwen3.5:latest` if you intend the library default)
- [ ] `ollama list` includes the pulled tag
- [ ] Point config at the exact tag (example: `hybrid_reduce_model: qwen3.5:9b`)
- [ ] Run one short episode or a smoke eval; verify summary quality and JSON validity if your pipeline expects structured output

#### Tier 2 — `qwen3.5:27b` (quality)

Roughly **~17 GB** for the common `q4_K_M`-class tag on the library page—suitable for **24 GB+** unified memory with headroom for the app and OS.

- [ ] `ollama pull qwen3.5:27b`
- [ ] `ollama list` includes `qwen3.5:27b` (or the specific variant you pulled, e.g. `qwen3.5:27b-q4_K_M`)
- [ ] Set `hybrid_reduce_model` / `ollama_summary_model` to that tag
- [ ] Re-run the same validation as Tier 1; compare latency vs Tier 1

#### Tier 3 — `qwen3.5:35b` (heavy / Apple Silicon 48 GB class)

**MoE-style** tags (e.g. `35b-a3b` on the library) are often **~24 GB** for a common quant—feasible on **48 GB** unified RAM if little else is memory-heavy; expect slower inference than Tier 1–2.

- [ ] `ollama pull qwen3.5:35b` (or the explicit MoE tag you want, e.g. `qwen3.5:35b-a3b`, from the library)
- [ ] `ollama list` shows the model; confirm Activity Monitor memory stays acceptable under load
- [ ] Set config to the **exact** pulled tag
- [ ] Validate outputs as in Tier 1; if quality gains are marginal, prefer Tier 2 for speed

#### Qwen 3.5 — repository alignment

Use this block **after** you work through Tier 1–3. It ties the operational checklist to what the repo provides.

- [ ] **Tier 1 prompts (9B):** `src/podcast_scraper/prompts/ollama/qwen3.5_9b/` — `ner/` + `summarization/` (same output contracts as Qwen 2.5 7B templates)
- [ ] **Tier 2 prompts (27B):** `src/podcast_scraper/prompts/ollama/qwen3.5_27b/`
- [ ] **Tier 3 prompts (35B):** `src/podcast_scraper/prompts/ollama/qwen3.5_35b/`; for tag `qwen3.5:35b-a3b`, use `src/podcast_scraper/prompts/ollama/qwen3.5_35b-a3b/`
- [ ] **Integration tests:** `pytest tests/integration/providers/ollama/test_qwen3_5_ollama_tiers.py` (or run the full Ollama integration slice via `make test-integration`)
- [ ] **Eval configs (`data/eval/configs/`):** **`llm_ollama_*_smoke_v1`** = pure Ollama summarization (no LongT5; same shape as `llm_mistral_smoke_v1`). Mistral-family Ollama smokes include `llm_ollama_mistral_7b_smoke_v1`, `llm_ollama_mistral_nemo_12b_smoke_v1`, `llm_ollama_mistral_small3_2_smoke_v1`. Llama 3.x smokes include `llm_ollama_llama32_3b_smoke_v1`, `llm_ollama_llama33_70b_q3km_smoke_v1`. **`hybrid_ml_tier2_*`** = LongT5 MAP + Ollama REDUCE (needs HF MAP cache + `ollama pull`). Qwen 3.5 tier extras: `hybrid_ml_tier2_qwen35_*b_authority_v1`, `hybrid_ml_tier2_qwen35_9b_smoke_tuned_v1`, `hybrid_ml_tier2_qwen35_9b_smoke_paragraph_v1`.
- [ ] **Acceptance configs (`config/acceptance/summarization/`, mirror Qwen 2.5):** hybrid — `acceptance_planet_money_hybrid_ollama_qwen35_9b.yaml`, `_27b`, `_35b`; full Ollama stack — `acceptance_planet_money_ollama_qwen3_5_9b.yaml`, `_27b`, `_35b`. See `config/acceptance/README.md`.

### Step 4: Install Podcast Scraper Dependencies

```bash
# Install with Ollama support
pip install -e ".[ollama]"

# Or install all optional dependencies
pip install -e ".[all]"
```

**Required Dependencies:**

- `openai` - Used for OpenAI-compatible API client
- `httpx` - Used for health checks

## Basic Setup

### Configuration File

```yaml
# config.yaml
speaker_detector_provider: ollama
summary_provider: ollama

# Ollama configuration
ollama_api_base: http://localhost:11434/v1  # Default, can be omitted
ollama_speaker_model: llama3.3:latest       # Production model
ollama_summary_model: llama3.3:latest       # Production model
ollama_temperature: 0.3                      # Lower = more deterministic
ollama_timeout: 300                          # 5 minutes for slow inference
```

### CLI Usage

```bash
# Basic usage
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider ollama \
  --summary-provider ollama

# With custom model
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider ollama \
  --summary-provider ollama \
  --ollama-speaker-model llama3.2:latest \
  --ollama-summary-model llama3.2:latest

# With custom timeout (for slow models)
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider ollama \
  --summary-provider ollama \
  --ollama-timeout 600  # 10 minutes
```

### Using Ollama as REDUCE Backend (hybrid_ml)

When using **hybrid MAP-REDUCE** summarization (`summary_provider: hybrid_ml`), you can set `hybrid_reduce_backend: ollama` so the REDUCE phase runs on a local Ollama model instead of transformers. The MAP phase (e.g. LongT5-base) still runs locally; only the final synthesis uses Ollama.

- **Config**: `summary_provider: hybrid_ml`, `hybrid_reduce_backend: ollama`, `hybrid_reduce_model: <ollama_tag>` (e.g. `llama3.1:8b`, `mistral:7b`, `qwen2.5:7b`, `qwen2.5:32b`).
- **No template file**: The reduce instruction is sent to Ollama as an inline prompt; no `custom.j2` or other template file is required.
- **Acceptance tests**: Example configs under `config/acceptance/` use Ollama for REDUCE (e.g. `acceptance_planet_money_hybrid_ollama_llama3_8b.yaml` with `llama3.1:8b`). Ensure the model is available (`ollama pull llama3.1:8b` or equivalent).

See [ML Provider Reference](ML_PROVIDER_REFERENCE.md#hybrid-ml-provider-summary_provider-hybrid_ml) for full hybrid_ml configuration.

### Environment Variables

```bash
# Set custom API base (if Ollama is on different host/port)
export OLLAMA_API_BASE=http://192.168.1.100:11434/v1

# Then use in CLI or config
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider ollama
```

## Checklist: add support for a new Ollama model (in this repo)

Use this when you introduce another **Ollama model tag** (e.g. a larger variant of the same family) and want it to be easy to run and document—not for one-off local experiments only.

1. **Runtime (enough to use it):** `ollama pull <tag>`. Point config or CLI at the tag (`ollama_speaker_model`, `ollama_summary_model`, and/or `hybrid_reduce_model` with `hybrid_reduce_backend: ollama`). There is **no** new provider type: the same Ollama integration applies; only the model string changes (same idea as switching OpenAI model names).

2. **Prompts (only if needed):** Optional model-specific templates live under `prompts/.../ollama/<dir>/`, where `<dir>` is the tag with `:` replaced by `_` (e.g. `qwen2.5:32b` → `qwen2.5_32b`). If that path does not exist, the provider **falls back** to generic `ollama/<task>/...` prompts.

3. **Eval configs (optional):** Add a new file under `data/eval/configs/` with a unique `id`, copied from a similar run; set `reduce_model` or the relevant Ollama fields to `<tag>` so multi-run / comparison workflows can name the run.

4. **Acceptance configs (optional):** Add something like `config/acceptance/summarization/acceptance_planet_money_hybrid_ollama_<model>.yaml` (mirror an existing hybrid or full-Ollama YAML). Update the table in `config/acceptance/README.md` if the config should be discoverable. **Do not** add Ollama-heavy stems to `config/acceptance/FAST_CONFIGS.txt` unless you want PR CI to run them.

5. **Tests (optional):** Extra integration tests are only warranted if you need to pin behavior for that tag; otherwise existing Ollama tests plus config strings usually suffice.

## Troubleshooting

### Issue: `ollama list` Hangs or Times Out

**Symptom:** Command hangs with no output.

**Cause:** Ollama server is not running.

**Solution:**

```bash
# 1. Start Ollama server
ollama serve

# 2. Keep that terminal open, then in another terminal:
ollama list

# 3. Verify server is responding
curl http://localhost:11434/api/tags
```

### Issue: "Ollama server is not running"

**Symptom:** Error message when trying to use Ollama provider.

**Cause:** Ollama server is not accessible at `http://localhost:11434`.

**Solution:**

```bash
# Check if server is running
curl http://localhost:11434/api/tags

# If connection fails, start server:
ollama serve

# If using custom host/port, set environment variable:
export OLLAMA_API_BASE=http://your-host:11434/v1
```

### Issue: "Model 'llama3.3:latest' is not available"

**Symptom:** Error message about model not found.

**Cause:** Model hasn't been pulled yet.

**Solution:**

```bash
# Pull the required model
ollama pull llama3.3:latest

# Verify it's available
ollama list

# Test the model directly
ollama run llama3.3:latest "Hello, test"
```

### Issue: HTTP 500 from Ollama / experiment seems stuck

**Symptom:** Logs or a proxy show `500` from `POST .../v1/chat/completions`, or a summarization experiment sits on one episode for a long time before failing.

**Typical causes:** Model out of memory (especially large quants on limited RAM), context length too large for the loaded model, or a transient Ollama bug. The OpenAI-compatible client used to **retry** 5xx errors; for **local** Ollama, **HTTP 500 is no longer retried** (fail fast). Timeouts can still wait up to `ollama_timeout` seconds (default **120**) per attempt.

**What to try:**

1. Check Ollama logs (macOS example): `cat ~/.ollama/logs/server.log` or run `ollama serve` in a terminal and watch stderr.
2. Run a tiny prompt: `ollama run <your-tag> "Say hello in one sentence."`
3. Use a smaller quant / tag, close other GPU-heavy apps, or set a longer read timeout for eval runs only:
   `EXPERIMENT_OLLAMA_READ_TIMEOUT=600 make experiment-run CONFIG=...`
4. For persistent 500s on one model only, try another tag from [ollama.com/library](https://ollama.com/library) or reduce transcript size (preprocessing / chunking).

### Issue: Connection Refused on Custom Port

**Symptom:** Error connecting to Ollama on non-default port.

**Cause:** Ollama is running on default port (11434) but config specifies different port.

**Solution:**

```bash
# Check what port Ollama is actually using
ps aux | grep ollama

# Or check Ollama config
cat ~/.ollama/config.json  # If exists

# Update config to match actual port:
# ollama_api_base: http://localhost:11434/v1  # Default
```

### Issue: Slow Performance

**Symptom:** Ollama inference is very slow.

**Causes & Solutions:**

1. **Model too large for hardware:**

   ```bash
   # Use smaller model (best for limited RAM)
   ollama pull llama3.1:8b
   # Update config: ollama_speaker_model: llama3.1:8b
   # OR for slightly better quality with more RAM:
   # ollama pull llama3.2:latest
   ```

2. **Insufficient RAM:**

   ```bash
   # Check available RAM
   free -h  # Linux
   vm_stat  # macOS

   # Use smaller model or add more RAM
   ```

3. **CPU-only inference (no GPU):**
   - Ollama will use CPU if no GPU available
   - Consider using GPU-accelerated models if available
   - Increase timeout: `ollama_timeout: 600`

4. **Multiple concurrent requests:**
   - Ollama processes requests sequentially by default
   - This is expected behavior for local inference

### Issue: Process Won't Die

**Symptom:** Can't kill Ollama process.

**Solution:**

```bash
# Kill by name
pkill ollama

# Or force kill
killall ollama

# Or find and kill manually
ps aux | grep ollama
kill <PID>

# If running as service (macOS)
brew services stop ollama
```

### Issue: Port Already in Use

**Symptom:** Error starting Ollama server - port 11434 already in use.

**Solution:**

```bash
# Find what's using the port
lsof -i :11434

# Kill the process
kill <PID>

# Or use different port (requires Ollama config change)
# Not recommended - better to kill existing process
```

### Issue: Permission Denied

**Symptom:** Permission errors when starting Ollama.

**Solution:**

```bash
# macOS: Grant network permissions in System Settings
# Settings > Privacy & Security > Network Extensions

# Linux: Check user permissions
# Ollama should run as your user, not root
```

## Testing with Real Models

### E2E Tests with Real Ollama

```bash
# 1. Ensure Ollama server is running
ollama serve  # In separate terminal

# 2. Pull required models
ollama pull llama3.3:latest

# 3. Run E2E tests with real Ollama
USE_REAL_OLLAMA_API=1 \
pytest tests/e2e/test_ollama_provider_integration_e2e.py -v

# 4. Test specific capability
USE_REAL_OLLAMA_API=1 \
pytest tests/e2e/test_ollama_provider_integration_e2e.py::TestOllamaProviderE2E::test_ollama_speaker_detection_in_workflow -v
```

### Test with Real RSS Feed

```bash
USE_REAL_OLLAMA_API=1 \
LLM_TEST_RSS_FEED="https://feeds.npr.org/510289/podcast.xml" \
pytest tests/e2e/test_ollama_provider_integration_e2e.py::TestOllamaProviderE2E::test_ollama_all_providers_in_pipeline -v
```

### Test Multiple Episodes

```bash
USE_REAL_OLLAMA_API=1 \
LLM_TEST_MAX_EPISODES=3 \
pytest tests/e2e/test_ollama_provider_integration_e2e.py::TestOllamaProviderE2E::test_ollama_all_providers_in_pipeline -v
```

## Performance Tips

### Model Selection

**By Use Case:**

- **Structured JSON / GIL Extraction:** Use `qwen2.5:7b` (best-in-class JSON output)
- **Speaker Detection (Default):** Use `llama3.1:8b` (strong all-rounder)
- **Fast Speaker Detection:** Use `mistral:7b` (fastest inference)
- **Summarization (Balanced):** Use `gemma2:9b` (balanced quality/speed)
- **Development/Testing:** Use `phi3:mini` (lightweight, lowest RAM)

**By RAM Available:**

- **4GB+ RAM:** Use `phi3:mini` (lightweight, dev/test only)
- **8GB+ RAM:** Use `qwen2.5:7b`, `llama3.1:8b`, or `mistral:7b` (recommended)
- **12GB+ RAM:** Use `gemma2:9b` (balanced quality/speed)

**Default Recommendations:**

- **Speaker Detection:** `llama3.1:8b` (general purpose, good quality)
- **Summarization:** `qwen2.5:7b` (best structured JSON, ideal for GIL extraction)
- **Development:** `phi3:mini` (lightweight, fast iteration)

### Timeout Configuration

```yaml
# For fast models (llama3.2)
ollama_timeout: 120  # 2 minutes

# For medium models (llama3.3)
ollama_timeout: 300  # 5 minutes

# For large models (70b)
ollama_timeout: 600  # 10 minutes
```

### Hardware Recommendations

| Model | Size | Minimum RAM | Recommended RAM | Speed | Best For |
| ----- | ---- | ----------- | ---------------- | ----- | -------- |
| **Phi-3 Mini** | 2.3GB | 4GB | 4GB+ | Fast | Dev/test, low-resource |
| **Mistral 7B** | 4.1GB | 8GB | 8GB+ | Fastest | Fast speaker detection |
| **Qwen 2.5 7B** | 4.4GB | 8GB | 8GB+ | Medium | Structured JSON, GIL extraction |
| **Llama 3.1 8B** | 4.7GB | 8GB | 8GB+ | Medium | General purpose (default) |
| **Gemma 2 9B** | 5.5GB | 12GB | 12GB+ | Medium | Balanced quality/speed |

## Common Workflows

### Development/Testing Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull model (choose based on your RAM and use case)
ollama pull phi3:mini        # Lightweight (4GB+ RAM, dev/test)
ollama pull mistral:7b       # Fast (8GB+ RAM, fast speaker detection)
ollama pull mistral-nemo:12b # Mistral Nemo 12B (16GB+ RAM typical)
ollama pull mistral-small3.2 # Mistral Small 3.2 (see library for exact size)
ollama pull llama3.1:8b      # General purpose (8GB+ RAM, default)
ollama pull qwen2.5:7b       # Best JSON (8GB+ RAM, GIL extraction)
ollama pull gemma2:9b        # Balanced (12GB+ RAM, summarization)

# 3. Configure (model-specific prompts are automatically selected)
# config.yaml:
#   ollama_speaker_model: llama3.1:8b  # or mistral:7b for speed
#   ollama_summary_model: qwen2.5:7b    # or gemma2:9b for quality
#   ollama_timeout: 120
```

### Production Setup

```bash
# 1. Start Ollama as service
brew services start ollama  # macOS
# Or: systemctl start ollama  # Linux

# 2. Pull production model
ollama pull llama3.3:latest

# 3. Configure for quality
# config.yaml:
#   ollama_speaker_model: llama3.3:latest
#   ollama_summary_model: llama3.3:latest
#   ollama_timeout: 300
```

### Hybrid Setup (Ollama + Other Providers)

```yaml
# Use Ollama for privacy-sensitive operations
# Use other providers for speed/cost optimization

transcription_provider: whisper      # Local transcription
speaker_detector_provider: ollama    # Local speaker detection (privacy)
summary_provider: ollama             # Local summarization (privacy)
```

## Verification

### Quick Health Check

```bash
# 1. Check server is running
curl http://localhost:11434/api/tags

# 2. List available models
ollama list

# 3. Test model directly
ollama run llama3.3:latest "What is a podcast?"

# 4. Test with podcast_scraper
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider ollama \
  --summary-provider ollama \
  --max-episodes 1
```

### Verify Provider is Working

```bash
# Run with verbose logging
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider ollama \
  --summary-provider ollama \
  --log-level DEBUG \
  --max-episodes 1

# Check metadata files for provider information
cat output/*/metadata/*.json | jq '.processing.config_snapshot.ml_providers'
```

## Related Documentation

- [Provider Configuration Quick Reference](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) - Configuration options
- [AI Provider Comparison Guide](AI_PROVIDER_COMPARISON_GUIDE.md) - Ollama vs other providers
- [E2E Testing Guide](E2E_TESTING_GUIDE.md) - Testing with real Ollama
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions

## Additional Resources

- **Ollama Documentation:** [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
- **Ollama Models:** [https://ollama.ai/library](https://ollama.ai/library)
- **Ollama API Reference:** [https://github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md)
