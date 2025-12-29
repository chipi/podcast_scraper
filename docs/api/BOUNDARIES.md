# API Boundaries and Architecture Assessment

This document describes the public API boundaries of `podcast_scraper` and identifies any
API creeps (internal details that shouldn't be exposed).

## Public API Surface

### Core Public API (`__init__.py`)

The main package exposes a minimal, clean API:

````python
from podcast_scraper import Config, load_config_file, run_pipeline, service

# Configuration

config = Config(rss_url="...", ...)
config = load_config_file("config.yaml")

# Direct pipeline execution

count, summary = run_pipeline(config)

# Service API (for daemon/service use)

result = service.run(config)
result = service.run_from_config_file("config.yaml")
```yaml

- `service.run(config: Config) -> ServiceResult` - Run pipeline with structured result
- `service.run_from_config_file(path: str) -> ServiceResult` - Convenience wrapper
- `service.main()` - CLI entry point for service mode (`python -m podcast_scraper.service`)

**ServiceResult** provides:

- `episodes_processed: int` - Number of episodes processed
- `summary: str` - Human-readable summary
- `success: bool` - Whether run completed successfully
- `error: Optional[str]` - Error message if failed

### CLI API (`cli.py`)

The CLI module is for interactive command-line use:

- `cli.main()` - Main CLI entry point
- `cli.parse_args()` - Argument parsing (internal, but may be used by tests)

## Internal Modules (Not Public API)

The following modules are **internal implementation details** and should not be imported directly:

- `workflow.py` - Core pipeline logic (use `run_pipeline` from `__init__.py`)
- `episode_processor.py` - Episode processing internals
- `rss_parser.py` - RSS parsing internals
- `downloader.py` - HTTP download internals
- `filesystem.py` - File system utilities
- `metadata.py` - Metadata generation internals
- `speaker_detection.py` - Speaker detection internals
- `summarizer.py` - Summarization internals
- `whisper_integration.py` - Whisper integration internals
- `metrics.py` - Metrics collection internals
- `progress.py` - Progress reporting internals
- `models.py` - Internal data models

## API Creeps Assessment

### ✅ Good: Clean Separation

1. **`workflow.run_pipeline`** - Properly exposed as public API
2. **`config.Config`** - Public API for configuration
3. **`config.load_config_file`** - Public API for config loading
4. **Internal functions prefixed with `_`** - Good convention

### ⚠️ Potential Issues

1. **`workflow.apply_log_level`** - Currently used by both CLI and service
   - **Status**: Acceptable - it's a utility function that both interfaces need
   - **Recommendation**: Could be moved to a `utils.py` module if it grows, but fine for now

2. **Direct imports from `workflow`** - CLI and service import `workflow` directly
   - **Status**: Acceptable - they're importing the public `run_pipeline` function
   - **Recommendation**: This is fine as long as we don't expose internal functions

3. **`progress` module** - May be used by external code
   - **Status**: Currently not in `__all__`, so not officially public
   - **Recommendation**: Keep as internal unless there's a clear use case

## Architecture Isolation

### Separation of Concerns

1. **CLI (`cli.py`)**:
   - Handles argument parsing
   - Formats output for terminal
   - Sets up progress bars
   - Calls `workflow.run_pipeline`

2. **Service (`service.py`)**:
   - Works with config files only
   - Returns structured results
   - No user interaction
   - Calls `workflow.run_pipeline`

3. **Workflow (`workflow.py`)**:
   - Core pipeline orchestration
   - No CLI or service concerns
   - Pure business logic

### Good Isolation Points

- ✅ CLI and Service both use the same `run_pipeline` function
- ✅ Configuration is separated into `Config` class
- ✅ Internal functions are prefixed with `_`
- ✅ Service API provides structured results suitable for automation

### Recommendations

1. **Keep `workflow.run_pipeline` as the single entry point** - Both CLI and service use it
2. **Don't expose internal modules** - Keep them out of `__all__`
3. **Service API is separate from CLI** - Good separation for different use cases
4. **Consider adding more utility functions to service API** if needed (e.g., health checks, status queries)

## Usage Patterns

### Interactive CLI Use

```bash
python -m podcast_scraper.cli https://example.com/feed.xml --max-episodes 10
```text

python -m podcast_scraper.service --config config.yaml

```text
```python

from podcast_scraper import service

result = service.run_from_config_file("config.yaml")
if not result.success:

    # Handle error

    pass

```yaml

### Versioning Policy

- **Major version (X.y.z)**: Breaking API changes
  - Function signatures change
  - Return types change
  - Required parameters added
  - Public classes/modules removed

- **Minor version (x.Y.z)**: New features, backward compatible
  - New functions/classes added
  - Optional parameters added
  - New return fields (with defaults)
  - Deprecation warnings (removed in next major)

- **Patch version (x.y.Z)**: Bug fixes, backward compatible
  - Bug fixes only
  - No API changes

### Version Access

```python

import podcast_scraper

# Access API version

api_version = podcast_scraper.__api_version__  # "2.3.0"
module_version = podcast_scraper.__version__   # "2.3.0"

# Versions are always the same

assert api_version == module_version

```yaml
- **Deprecation**: Features deprecated in minor versions are removed in next major version

## Future Considerations

1. **Health Checks**: Service API could expose health check endpoints
2. **Status Queries**: Could add functions to query pipeline status
3. **Event Hooks**: Could add callback/hook system for integration
````
