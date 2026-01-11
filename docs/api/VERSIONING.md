# API Versioning Policy

This document describes the versioning policy for the `podcast_scraper` public API.

## Semantic Versioning

The `podcast_scraper` API follows [Semantic Versioning 2.0.0](https://semver.org/):

```text
MAJOR.MINOR.PATCH
```

- **MAJOR** version: Breaking API changes (incompatible changes).
- **MINOR** version: New features, backward compatible (e.g., new providers, optional configuration fields).
- **PATCH** version: Bug fixes, backward compatible.

## API Version

The API version is exposed via:

```python
import podcast_scraper

print(podcast_scraper.__version__)      # Package version (e.g., "2.4.0")
print(podcast_scraper.__api_version__)  # API version (e.g., "2.4.0")
```

The public API consists of:

### Core API (Stable)

- `podcast_scraper.Config` - Configuration model.
- `podcast_scraper.load_config_file()` - Load config from file.
- `podcast_scraper.run_pipeline()` - Run the pipeline.
- `podcast_scraper.service.run()` - Service API.
- `podcast_scraper.service.run_from_config_file()` - Service API from config.
- `podcast_scraper.service.ServiceResult` - Service result dataclass.
- `podcast_scraper.cli.main()` - CLI entry point.

### Models (Stable)

- `podcast_scraper.models.RssFeed`
- `podcast_scraper.models.Episode`
- `podcast_scraper.models.TranscriptionJob`

### Internal API (Unstable)

Everything else is considered **internal** and may change without notice:

- Modules not explicitly exported in `__init__.py`.
- Functions/classes prefixed with `_`.
- Implementation details in submodules (e.g., `rss_parser`, `downloader`).

## Deprecation Policy

When we need to make breaking changes:

1. **Deprecation Warning** (Minor version)
   - Add deprecation warning to the old API.
   - Document the new API and migration path in [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).
   - Keep both APIs working for at least one minor version.

2. **Breaking Change** (Major version)
   - Remove deprecated API in next major version.
   - Update migration guide.
   - Increment major version number.

## Compatibility Promise

### What We Guarantee

✅ **Backward compatibility within major versions:**

- Minor and patch releases are backward compatible.
- Existing configuration files continue to work.
- Deprecation warnings are provided at least one minor version before removal.

✅ **Clear migration paths:**

- Breaking changes are documented in the [Migration Guide](MIGRATION_GUIDE.md).

### What We Don't Guarantee

❌ **Internal API stability:**

- Internal modules and private functions/classes may change.

❌ **CLI output format:**

- CLI output may change for better user experience. Use the **Service API** for programmatic parsing.

❌ **Filesystem layout:**

- The directory structure of the output may evolve (e.g., v2.4.0 introduction of `transcripts/` subfolder).

## Related

- [API Boundaries](BOUNDARIES.md) - What's public vs internal
- [API Migration Guide](MIGRATION_GUIDE.md) - Upgrading between versions
- [Release Notes](../releases/index.md) - Changes per version
