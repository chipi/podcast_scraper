# API Versioning Policy

This document describes the versioning policy for the podcast_scraper public API.

## Semantic Versioning

The podcast_scraper API follows [Semantic Versioning 2.0.0](https://semver.org/):

````text
MAJOR.MINOR.PATCH
```yaml

- **MAJOR** version: Breaking API changes (incompatible changes)
- **MINOR** version: New features, backward compatible
- **PATCH** version: Bug fixes, backward compatible

## API Version

The API version is exposed via:

```python
import podcast_scraper

print(podcast_scraper.__version__)      # e.g., "2.3.0"
print(podcast_scraper.__api_version__)  # e.g., "2.3.0"
```text

The public API consists of:

### Core API (Stable)

These are **stable** and follow semantic versioning:

- `podcast_scraper.Config` - Configuration model
- `podcast_scraper.load_config_file()` - Load config from file
- `podcast_scraper.run_pipeline()` - Run the pipeline
- `podcast_scraper.service.run()` - Service API
- `podcast_scraper.service.run_from_config_file()` - Service API from config
- `podcast_scraper.service.ServiceResult` - Service result dataclass
- `podcast_scraper.cli.main()` - CLI entry point

### Models (Stable)

- `podcast_scraper.models.RssFeed`
- `podcast_scraper.models.Episode`
- `podcast_scraper.models.TranscriptionJob`

### Internal API (Unstable)

Everything else is considered **internal** and may change without notice:

- Modules not explicitly exported in `__all__`
- Functions/classes prefixed with `_`
- Implementation details in submodules

## Deprecation Policy

When we need to make breaking changes:

1. **Deprecation Warning** (Minor version)
   - Add deprecation warning to the old API
   - Document the new API and migration path
   - Keep both APIs working for at least one minor version

2. **Breaking Change** (Major version)
   - Remove deprecated API in next major version
   - Update migration guide
   - Increment major version number

### Example Deprecation

```python
import warnings

def old_function():
    warnings.warn(
        "old_function() is deprecated, use new_function() instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```text

- **v2.3.0** (2025-11) - Episode summarization, public API refinement
- **v2.2.0** (2025-11) - Metadata generation
- **v2.1.0** (2025-11) - Automatic speaker detection
- **v2.0.0** (2025-11) - Modular architecture, service API

### v1.x.x (Legacy)

- **v1.0.0** (2025-11) - Initial release

## Compatibility Promise

### What We Guarantee

✅ **Backward compatibility within major versions:**

- Minor and patch releases are backward compatible
- Existing code continues to work
- Deprecation warnings before removal

✅ **Clear migration paths:**

- Breaking changes documented in migration guide
- Deprecation period of at least one minor version

### What We Don't Guarantee

❌ **Internal API stability:**

- Internal modules may change
- Private functions/classes (prefixed with `_`)
- Implementation details

❌ **CLI output format:**

- CLI output may change for better UX
- Use service API for programmatic parsing

❌ **File formats:**

- Metadata JSON/YAML structure may evolve
- Transcript formatting may improve

## Checking Compatibility

To check if your code is compatible with a version:

```python
import podcast_scraper

# Check API version

required_version = "2.3.0"
if podcast_scraper.__api_version__ < required_version:
    raise RuntimeError(f"Requires API version {required_version}+")
```python

from packaging import version

min_version = version.parse("2.3.0")
current_version = version.parse(podcast_scraper.__version__)

if current_version < min_version:
    raise RuntimeError(f"Requires version {min_version}+, got {current_version}")

```text
- Async/await support for I/O operations
- Restructured configuration model
- Different return types for better structured data

These are subject to change and will be announced well in advance.

## Migration Guides

When breaking changes occur, we provide detailed migration guides:

- [API Migration Guide](MIGRATION_GUIDE.md) - Version-specific migrations
- Release notes - Changes per version

## Questions?

If you have questions about API stability or versioning:

1. Check [API Boundaries](BOUNDARIES.md) for public vs internal API
2. Review [API Migration Guide](MIGRATION_GUIDE.md) for version changes
3. Open an issue on GitHub

## Related

- [API Boundaries](BOUNDARIES.md) - What's public vs internal
- [API Migration Guide](MIGRATION_GUIDE.md) - Upgrading between versions
- [Release Notes](../releases/) - Changes per version
- [Semantic Versioning](https://semver.org/) - Versioning standard
````
