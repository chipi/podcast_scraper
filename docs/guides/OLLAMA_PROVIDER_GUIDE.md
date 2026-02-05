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
# For production/quality (larger, slower)
ollama pull llama3.3:latest

# For testing/speed (medium size)
ollama pull llama3.2:latest

# For limited RAM (smallest, fastest)
ollama pull llama3.1:8b

# Verify models are available
ollama list
```

**Recommended Models:**

| Model | Size | RAM Required | Speed | Quality | Use Case |
| ----- | ---- | ------------- | ----- | ------- | -------- |
| `llama3.1:8b` | ~4.7GB | 6-8GB | Fastest | Good | Limited RAM (6-8GB), testing, development |
| `llama3.2:latest` | ~4GB | 8-12GB | Fast | Good | Standard systems (8-12GB), testing, development |
| `llama3.3:latest` | ~4GB | 12-16GB | Medium | Better | Production (12-16GB recommended) |
| `llama3.3:70b` | ~40GB | 48GB+ | Slow | Best | High-quality production (48GB+ RAM) |

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

### Environment Variables

```bash
# Set custom API base (if Ollama is on different host/port)
export OLLAMA_API_BASE=http://192.168.1.100:11434/v1

# Then use in CLI or config
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider ollama
```

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

- **For Limited RAM (6-8GB):** Use `llama3.1:8b` (smallest, fastest)
- **For Standard Systems (8-12GB):** Use `llama3.2:latest` (good balance)
- **For Production (12-16GB):** Use `llama3.3:latest` (better quality, fits under 16GB)
- **For High Quality (48GB+):** Use `llama3.3:70b` (best quality, requires significant RAM)

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

| Model Size | Minimum RAM | Recommended RAM | Speed | Example Models |
| ---------- | ----------- | ---------------- | ----- | -------------- |
| 8B | 6GB | 8GB | Fastest | `llama3.1:8b` |
| 8B (latest) | 8GB | 12GB | Fast | `llama3.2:latest` |
| 8B (latest) | 12GB | 16GB | Medium | `llama3.3:latest` (fits under 16GB) |
| 13B | 16GB | 32GB | Medium | `llama3.1:13b` |
| 70B | 48GB | 64GB+ | Slow | `llama3.3:70b` |

## Common Workflows

### Development/Testing Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull fast model (choose based on your RAM)
ollama pull llama3.1:8b      # Smallest (6GB+ RAM)
# OR
ollama pull llama3.2:latest  # Medium (8GB+ RAM)

# 3. Configure for speed
# config.yaml:
#   ollama_speaker_model: llama3.1:8b  # or llama3.2:latest
#   ollama_summary_model: llama3.1:8b   # or llama3.2:latest
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
