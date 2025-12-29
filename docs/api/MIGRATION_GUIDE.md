# API Migration Guide - v1.0 to v2.0

## Overview

Version 2.0 refactors the monolithic v1.0 into a clean modular architecture with a
**minimal public API of just 4 core items**, making the package easier to use and maintain.

## Changes

### Modular Architecture with Minimal Public API

**Before (v1.0):** Monolithic `podcast_scraper.py` file

- Single ~800-line file
- All functionality in one module
- No formal public API
- Difficult to test and maintain

**After (v2.0):** 8 focused modules with 4 public exports ⭐

````python
__all__ = ["Config", "load_config_file", "run_pipeline", "cli"]
```text
```python
import podcast_scraper

# Configuration

config = podcast_scraper.Config(
    rss_url="https://example.com/feed.xml",
    output_dir="./transcripts",
    max_episodes=10,
)

# Or load from file

config = podcast_scraper.load_config_file("config.yaml")

# Run pipeline

count, summary = podcast_scraper.run_pipeline(config)

# CLI access

sys.exit(podcast_scraper.cli.main())
```text
```python

# v2.0 - Explicit module imports for internals:

from podcast_scraper.filesystem import sanitize_filename  # ✅
from podcast_scraper.models import Episode  # ✅
from podcast_scraper.downloader import http_get  # ✅
from podcast_scraper.rss_parser import parse_rss_items  # ✅
```text

Tests in v2.0 use explicit module imports:

```python

# v2.0 test pattern:

from podcast_scraper import filesystem
from podcast_scraper import models
from podcast_scraper import downloader

result = filesystem.sanitize_filename("test")
episode = models.Episode(...)
```text

1. **Modular Architecture** - 8 focused modules vs 1 monolithic file
2. **Simpler API** - Only 4 public exports, clear and focused
3. **Clearer Intent** - API says "this is a pipeline tool"
4. **Easier Refactoring** - Internal changes don't break users
5. **Better Testing** - Tests are explicit about dependencies
6. **Maintainability** - Each module has a single responsibility
7. **Reduced Cognitive Load** - Clear separation between public/private

## Version History

- **v1.0.0**: Monolithic `podcast_scraper.py` (~800 lines)
- **v2.0.0**: Modular architecture (8 modules) with minimal API (4 exports) ⭐
- **v2.3.0**: Added service API (`service.run`, `service.run_from_config_file`) and API versioning

## API Versioning

The API follows semantic versioning tied to the module version:

- **API Version**: `podcast_scraper.__api_version__` (same as `__version__`)
- **Major version changes**: Breaking API changes (see migration guide)
- **Minor version changes**: New features, backward compatible
- **Patch version changes**: Bug fixes, backward compatible

```python
import podcast_scraper

# Check API version

api_version = podcast_scraper.__api_version__  # "2.3.0"
```yaml

- README: [https://github.com/chipi/podcast_scraper/blob/main/README.md](https://github.com/chipi/podcast_scraper/blob/main/README.md)
- Full documentation site (architecture, PRDs, RFCs, guides): [https://chipi.github.io/podcast_scraper/](https://chipi.github.io/podcast_scraper/)

````
