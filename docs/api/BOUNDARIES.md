# API Boundaries and Architecture Assessment

This document describes the public API boundaries of `podcast_scraper` and identifies any API creeps (internal details that shouldn't be exposed).

## Public API Surface

### Core Public API (`__init__.py`)

The main package exposes a minimal, clean API:

```python
from podcast_scraper import Config, load_config_file, run_pipeline, service

# Configuration
config = Config(rss_url="...", ...)
config_dict = load_config_file("config.yaml")
config = Config(**config_dict)

# Direct pipeline execution
count, summary = run_pipeline(config)

# Service API (for daemon/service use)
result = service.run(config)
result = service.run_from_config_file("config.yaml")
```

#### Key Components

- `Config` - Immutable Pydantic model for all runtime options.
- `load_config_file(path: str) -> Dict[str, Any]` - Loader for JSON/YAML configs.
- `run_pipeline(cfg: Config) -> Tuple[int, str]` - Orchestrates the entire job.
- `service.run(cfg: Config) -> ServiceResult` - High-level service wrapper.
- `service.run_from_config_file(path: str) -> ServiceResult` - Convenience wrapper for services.

### ServiceResult

Provides structured outcomes for automated processing:

- `episodes_processed: int` - Number of episodes processed.
- `summary: str` - Human-readable summary.
- `success: bool` - Whether the run completed successfully.
- `error: Optional[str]` - Error message if failed.

## CLI API (`cli.py`)

The CLI module is for interactive command-line use:

- `cli.main()` - Main CLI entry point.
- `cli.parse_args()` - Argument parsing (internal, but used by tests).

## Internal Modules (Not Public API)

The following modules are **internal implementation details** and should not be imported directly:

- `workflow.py` - Core pipeline logic (use `run_pipeline` from `__init__.py`).
- `episode_processor.py` - Episode processing internals.
- `rss_parser.py` - RSS parsing internals.
- `downloader.py` - HTTP download internals.
- `filesystem.py` - File system utilities.
- `metadata.py` - Metadata generation internals.
- `speaker_detection.py` - Speaker detection internals.
- `summarizer.py` - Summarization internals.
- `whisper_integration.py` - Whisper integration internals.
- `progress.py` - Progress reporting internals.
- `models.py` - Internal data models.
- `providers/` - Multi-provider implementations (accessed via factories).

## API Creeps Assessment

### ✅ Good: Clean Separation

1. **`workflow.run_pipeline`** - Properly exposed as public API.
2. **`config.Config`** - Public API for configuration.
3. **`config.load_config_file`** - Public API for config loading.
4. **Internal functions prefixed with `_`** - Good convention followed throughout.

### ⚠️ Potential Issues

1. **`progress` module** - Currently not in `__all__`, but may be used by advanced external integrations. Keep as internal unless a public hook is requested.
2. **Provider Protocols** - While internal, they define the contract for extensibility. If we allow user-defined providers, these would need to move to public API.

## Architecture Isolation

### Separation of Concerns

1. **CLI (`cli.py`)**: Handles argument parsing, formats terminal output, and manages interactive progress bars.
2. **Service (`service.py`)**: Optimized for daemons; works with config files only, returns structured results, and has no user interaction.
3. **Workflow (`workflow.py`)**: Core pipeline orchestration; contains business logic but is agnostic of CLI or Service interfaces.

### Recommendations

1. **Keep `workflow.run_pipeline` as the single entry point** for all interfaces.
2. **Maintain strict module boundaries** to prevent circular imports and reduce coupling.
3. **Version internal APIs separately** if they are exposed for plugin development in the future.

## Usage Patterns

### Interactive CLI Use

```bash
python -m podcast_scraper.cli https://example.com/feed.xml --max-episodes 10
```

### Automated Service Use

```bash
python -m podcast_scraper.service --config config.yaml
```

### Programmatic Python Use

```python
from podcast_scraper import service

result = service.run_from_config_file("config.yaml")
if not result.success:
    # Handle error
    print(f"Failed: {result.error}")
```

## Versioning Policy

- **Major (X.y.z)**: Breaking API changes (e.g., signature changes, removal of classes).
- **Minor (x.Y.z)**: New features, backward compatible (e.g., new optional fields, new providers).
- **Patch (x.y.Z)**: Bug fixes, backward compatible.

See [Versioning](VERSIONING.md) for full details.
