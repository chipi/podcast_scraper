# Environment Variables

This document describes all environment variables supported by the podcast scraper.

## Overview

The podcast scraper supports configuration via environment variables, which can be set:

1. **System environment variables** (highest priority)
2. **`.env` file** (loaded automatically from project root or current directory)
3. **Config file fields** (lowest priority, used as fallback)

Environment variables are automatically loaded when the `podcast_scraper.config` module is imported using `python-dotenv`.

## Supported Environment Variables

### OpenAI API Configuration

#### `OPENAI_API_KEY`

**Description**: OpenAI API key for OpenAI-based providers (transcription, speaker detection, summarization).

**Required**: Yes, when using OpenAI providers (`transcription_provider=openai`, `speaker_detector_type=openai`, or `summary_provider=openai`).

**Example**:
```bash
export OPENAI_API_KEY=sk-your-actual-api-key-here
```

**In `.env` file**:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Security Notes**:

- Never commit `.env` files containing API keys
- Use `.env.example` as a template (without real keys)
- API keys are never logged or exposed in error messages
- Environment variables take precedence over `.env` file values

**See Also**: `docs/rfc/RFC-013-openai-provider-implementation.md`

## Usage Examples

### macOS / Linux

**Set environment variable for current session**:
```bash
export OPENAI_API_KEY=sk-your-key-here
python3 -m podcast_scraper https://example.com/feed.xml
```

**Set environment variable for single command**:
```bash
OPENAI_API_KEY=sk-your-key-here python3 -m podcast_scraper https://example.com/feed.xml
```

**Using `.env` file**:
```bash
# Create .env file in project root
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run command (env vars loaded automatically)
python3 -m podcast_scraper https://example.com/feed.xml
```

**Persistent environment variable** (add to `~/.bashrc` or `~/.zshrc`):
```bash
# Add to shell profile
export OPENAI_API_KEY=sk-your-key-here

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Windows

**Set environment variable for current session** (Command Prompt):
```cmd
set OPENAI_API_KEY=sk-your-key-here
python -m podcast_scraper https://example.com/feed.xml
```

**Set environment variable for current session** (PowerShell):
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
python -m podcast_scraper https://example.com/feed.xml
```

**Using `.env` file**:
```cmd
# Create .env file in project root
echo OPENAI_API_KEY=sk-your-key-here > .env

# Run command (env vars loaded automatically)
python -m podcast_scraper https://example.com/feed.xml
```

**Persistent environment variable** (Windows Settings):

1. Open "System Properties" → "Environment Variables"
2. Add new user or system variable:
   - Name: `OPENAI_API_KEY`
   - Value: `sk-your-key-here`
3. Restart terminal/IDE to apply changes

### Docker

**Using environment variable**:
```bash
docker run -e OPENAI_API_KEY=sk-your-key-here \
  podcast-scraper https://example.com/feed.xml
```

**Using `.env` file**:
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Docker Compose automatically loads .env
docker-compose up
```

**In `docker-compose.yml`**:
```yaml
services:
  podcast-scraper:
    image: podcast-scraper:latest
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./.env:/app/.env:ro
```

## .env File Setup

### Creating .env File

1. **Copy example template** (if available):
   ```bash
   cp .env.example .env
   ```

2. **Create `.env` file** in project root:
   ```bash
   # .env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Verify `.env` is in `.gitignore`**:
   ```bash
   # .gitignore should contain:
   .env
   .env.local
   .env.*.local
   ```

### .env File Location

The `.env` file is automatically loaded from:

1. **Project root** (where `config.py` is located): `{project_root}/.env`
2. **Current working directory**: `{cwd}/.env`

The first existing file is used. Project root takes precedence.

### .env File Format

```bash
# .env file format
# Comments start with #
# Empty lines are ignored
# No spaces around = sign

# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: Add other variables here
# LOG_LEVEL=DEBUG
```

## Security Best Practices

### ✅ DO

- **Use `.env` files for local development**
- **Add `.env` to `.gitignore`** (never commit secrets)
- **Use `.env.example` as template** (without real values)
- **Use environment variables in production** (more secure than files)
- **Rotate API keys regularly**
- **Use separate keys for development/production**
- **Restrict API key permissions** (if supported by provider)

### ❌ DON'T

- **Never commit `.env` files** with real API keys
- **Never hardcode API keys** in source code
- **Never log API keys** (they're automatically excluded from logs)
- **Never share API keys** in public repositories or chat
- **Never use production keys** in development

## Troubleshooting

### Environment Variable Not Found

**Problem**: `OPENAI_API_KEY` not found when using OpenAI providers.

**Solutions**:

1. **Check variable name**: Must be exactly `OPENAI_API_KEY` (case-sensitive)
2. **Check `.env` file location**: Should be in project root or current directory
3. **Check `.env` file format**: No spaces around `=`, no quotes needed
4. **Reload shell**: Restart terminal/IDE after setting environment variables
5. **Verify loading**: Check that `python-dotenv` is installed (`pip install python-dotenv`)

**Debug**:
```python
import os
print(os.getenv("OPENAI_API_KEY"))  # Should print your key (or None)
```

### .env File Not Loading

**Problem**: `.env` file exists but variables aren't loaded.

**Solutions**:

1. **Check file location**: Must be in project root (where `config.py` is) or current directory
2. **Check file name**: Must be exactly `.env` (not `.env.txt` or `env`)
3. **Check file permissions**: Must be readable
4. **Check file format**: No syntax errors, proper `KEY=value` format
5. **Verify `python-dotenv` installed**: `pip install python-dotenv`

**Debug**:
```python
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
print(f"Looking for .env at: {env_path}")
print(f"File exists: {env_path.exists()}")

if env_path.exists():
    load_dotenv(env_path, override=False)
    import os
    print(f"OPENAI_API_KEY loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
```

### Variable Precedence Issues

**Problem**: Config file value overrides environment variable.

**Note**: This is expected behavior. Priority order is:

1. Config file field (`openai_api_key`)
2. System environment variable (`OPENAI_API_KEY`)
3. `.env` file (`OPENAI_API_KEY`)

**Solution**: Remove `openai_api_key` from config file to use environment variable.

## Future Environment Variables

The following environment variables may be added in future versions:

- `OPENAI_ORGANIZATION` - OpenAI organization ID (for multi-org accounts)
- `OPENAI_API_BASE` - Custom API base URL (for proxies)
- `LOG_LEVEL` - Default log level (if not in config)
- `CACHE_DIR` - Custom cache directory for models

## Related Documentation

- `docs/rfc/RFC-013-openai-provider-implementation.md` - OpenAI API key management details
- `docs/prd/PRD-006-openai-provider-integration.md` - OpenAI provider requirements
- `docs/CUSTOM_PROVIDER_GUIDE.md` - Creating custom providers that may need API keys
- `README.md` - Quick start guide
- `CONTRIBUTING.md` - Development setup

## Support

If you encounter issues with environment variables:

1. Check this documentation
2. Review troubleshooting section above
3. Check related RFC/PRD documents
4. Open an issue on GitHub with:
   - Operating system and version
   - Python version
   - Error message or unexpected behavior
   - Steps to reproduce
