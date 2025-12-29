# API Documentation

## Purpose

This directory contains comprehensive API documentation for `podcast_scraper`, including
programmatic interfaces, configuration options, data models, and migration guides.

## API Documentation Index

### Core APIs

| Document                                                         | Description                                                              |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [Core API](CORE.md)                                              | Primary public API (`run_pipeline`, `Config`, package information)       |
| [Service API](SERVICE.md)                                        | Non-interactive service interface for daemons and process management     |
| [CLI Interface](CLI.md)                                          | Command-line interface documentation                                     |
| [Configuration API](CONFIGURATION.md)                            | Configuration model, environment variables, and file formats               |
| [Data Models](MODELS.md)                                         | Core data structures (Episode, RssFeed, TranscriptionJob)                 |

### API Reference & Guides

| Document                                                         | Description                                                              |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [API Reference](REFERENCE.md)                                    | Complete API reference documentation                                     |
| [API Boundaries](BOUNDARIES.md)                                  | Public vs. private API boundaries and stability guarantees               |
| [API Versioning](VERSIONING.md)                                  | API versioning strategy and compatibility policies                       |
| [API Migration Guide](MIGRATION_GUIDE.md)                        | Migration guides for API changes and breaking changes                    |

## Quick Start

**For programmatic usage:**

1. Start with [Core API](CORE.md) - Main entry point and functions
2. Review [Configuration API](CONFIGURATION.md) - Setup and configuration options
3. See [Data Models](MODELS.md) - Understand data structures

**For command-line usage:**

1. See [CLI Interface](CLI.md) - Command-line options and examples

**For service/daemon usage:**

1. See [Service API](SERVICE.md) - Non-interactive service interface

**For API stability and migration:**

1. See [API Boundaries](BOUNDARIES.md) - What's public vs. private
2. See [API Versioning](VERSIONING.md) - Versioning strategy
3. See [API Migration Guide](MIGRATION_GUIDE.md) - Breaking changes

## Quick Links

- **[Architecture](../ARCHITECTURE.md)** - System design and module responsibilities
- **[Development Guide](../guides/DEVELOPMENT_GUIDE.md)** - Development practices and guidelines
- **[Testing Guide](../guides/TESTING_GUIDE.md)** - Testing strategies and examples
