# Unit Testing Guide

> **See also:**
>
> - [Testing Strategy](../TESTING_STRATEGY.md) - High-level testing philosophy and test pyramid
> - [Testing Guide](TESTING_GUIDE.md) - Quick reference and test execution commands

This guide covers unit test implementation details: what to mock, isolation patterns, and testing practices.

## Overview

**Unit tests** test individual functions/modules in isolation with all dependencies mocked.

| Aspect | Requirement |
| -------- | ------------- |
| **Speed** | < 100ms per test |
| **Scope** | Single function or class |
| **Dependencies** | All mocked |
| **Network** | Blocked (enforced by pytest plugin) |
| **Filesystem** | Blocked except tempfile (enforced by pytest plugin) |
| **ML Models** | Not loaded (mocked before import) |

## What to Mock

### Always Mock

1. **HTTP/Network Calls**

   ```python
   @patch("podcast_scraper.downloader.requests.get")
   def test_download(self, mock_get):
       mock_get.return_value.status_code = 200
       mock_get.return_value.content = b"test content"
       # ...
   ```

2. **ML Models** (Whisper, spaCy, Transformers)

   ```python
   # Mock before importing dependent modules
   @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
   @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
   @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
   def test_provider_creation(self, mock_summary, mock_ner, mock_whisper):

```text
       # Test provider creation without loading real models
```python

3. **External API Clients** (OpenAI, etc.)

   ```python
   @patch("podcast_scraper.openai.openai_provider.OpenAI")
   def test_openai_provider(self, mock_client):
       mock_client.return_value.chat.completions.create.return_value = ...
   ```

1. **Filesystem Operations** (when testing logic, not file operations)

   ```python
   @patch("builtins.open", mock_open(read_data="test content"))
   def test_file_reading(self):

```text
       # Test file reading logic
```python

### Never Mock in Unit Tests

- **The function/class being tested** - That's what we're testing
- **Pure helper functions** - Test them directly
- **Data classes/models** - Create real instances

## Isolation Enforcement

Unit tests automatically enforce isolation via pytest plugins in `tests/unit/conftest.py`:

### Network Isolation

All network calls are blocked. If a test attempts network access:

```text
NetworkCallDetectedError: Attempt to make network call detected in unit test
```

**Blocked:**

- `requests.get()`, `requests.post()`, `requests.Session()` methods
- `urllib.request.urlopen()`
- `urllib3.PoolManager()`
- `socket.create_connection()`

### Filesystem Isolation

All filesystem I/O is blocked (except tempfile). If a test attempts I/O:

```text
FilesystemIODetectedError: Attempt to perform filesystem I/O in unit test
```

**Blocked:**

- `open()` for file operations (outside temp directories)
- `os.makedirs()`, `os.remove()`, `os.unlink()`, `os.rmdir()`, `os.rename()`
- `shutil.copy()`, `shutil.move()`, `shutil.rmtree()`
- `Path.write_text()`, `Path.write_bytes()`, `Path.mkdir()`, `Path.unlink()`

**Allowed:**

- `tempfile.mkdtemp()`, `tempfile.NamedTemporaryFile()`
- Operations within temp directories
- Cache directories (`~/.cache/`, `~/.local/share/`)
- Site-packages (read-only)
- Python cache files (`.pyc`, `__pycache__/`)

## ML Dependency Mocking

Unit tests must run **without ML packages installed** (for CI speed). Mock ML modules before importing dependent code:

```python
import sys
from unittest.mock import MagicMock

# Mock ML modules before import

sys.modules["whisper"] = MagicMock()
sys.modules["spacy"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Now import the module that uses these

from podcast_scraper import summarizer
```python

**CI Verification:** `scripts/check_unit_test_imports.py` verifies modules can import without ML deps.

## Test Structure

```python
class TestModuleName(unittest.TestCase):
    """Test module_name module."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("module.dependency")
    def test_function_success(self, mock_dependency):
        """Test successful function execution."""
        # Arrange
        mock_dependency.return_value = expected_value

        # Act
        result = function_under_test(input)

```text

        # Assert
        self.assertEqual(result, expected_result)
        mock_dependency.assert_called_once_with(...)

```python

    def test_function_error_handling(self):
        """Test function error handling."""
        with self.assertRaises(ExpectedError):
            function_under_test(invalid_input)

```

## Provider Testing Patterns

### Standalone Provider Tests

Test `MLProvider`/`OpenAIProvider` directly with mocked dependencies:

```python

class TestMLProvider(unittest.TestCase):
    """Test MLProvider standalone."""

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcription_initialization(self, mock_whisper):
        """Test transcription capability initialization."""
        provider = MLProvider(cfg)
        provider.initialize()
        mock_whisper.assert_called_once()

```

### Factory Tests

Test factories create correct unified providers:

```python

def test_create_transcription_provider_ml():
    """Test factory creates MLProvider for 'whisper'."""
    provider = create_transcription_provider(cfg)
    assert hasattr(provider, "transcribe")  # Protocol compliance

```python

**Key Principle:** Verify protocol compliance, not class names.

## Common Test Fixtures

```python

# Mock HTTP Response

class MockHTTPResponse:
    def __init__(self, content, status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

# Mock Whisper Model

mock_whisper_model = MagicMock()
mock_whisper_model.transcribe.return_value = {
    "text": "transcribed text",
    "segments": []
}

# Mock spaCy NLP

mock_nlp = MagicMock()
mock_nlp.return_value = [MagicMock(text="John", label_="PERSON")]

```yaml

## Test Files

| Module | Test File |
| -------- | ----------- |
| `config.py` | `tests/unit/podcast_scraper/test_config.py` |
| `filesystem.py` | `tests/unit/podcast_scraper/test_filesystem.py` |
| `rss_parser.py` | `tests/unit/podcast_scraper/test_rss_parser.py` |
| `downloader.py` | `tests/unit/podcast_scraper/test_downloader.py` |
| `service.py` | `tests/unit/podcast_scraper/test_service.py` |
| `summarizer.py` | `tests/unit/podcast_scraper/test_summarizer.py` |
| `speaker_detection.py` | `tests/unit/podcast_scraper/test_speaker_detection.py` |
| `metadata.py` | `tests/unit/podcast_scraper/test_metadata.py` |
| Provider factories | `tests/unit/podcast_scraper/*/test_*_provider.py` |
| MLProvider | `tests/unit/podcast_scraper/ml/test_ml_provider.py` |
| OpenAIProvider | `tests/unit/podcast_scraper/openai/test_openai_provider.py` |

## Running Unit Tests

```bash

# All unit tests

make test-unit

# Specific module

pytest tests/unit/podcast_scraper/test_config.py -v

# With coverage

pytest tests/unit/ --cov=podcast_scraper --cov-report=term-missing

```

## Provider Testing

For provider-specific testing patterns (unit tests for MLProvider, OpenAIProvider, factories):

→ **[Provider Implementation Guide - Testing Your Provider](PROVIDER_IMPLEMENTATION_GUIDE.md#testing-your-provider)**

Covers:

- Provider creation and initialization tests
- Mock API client patterns
- Factory tests and protocol compliance
- Testing checklist for new providers

## Coverage Targets

- **Overall:** >80%
- **Critical modules:** >90% (config, workflow, episode_processor)
- **Total tests:** 200+
