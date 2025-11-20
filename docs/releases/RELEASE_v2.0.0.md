## Version 2.0.0

**Release Date:** November 13, 2025  
**Type:** Major Release  
**Last Updated:** November 18, 2025

### 🎉 Major Release: Refactored Architecture & Comprehensive Documentation

Version 2.0.0 represents a significant milestone with a complete codebase refactoring, comprehensive documentation, and new features.

### 🏗️ Major Architectural Changes

#### Codebase Refactoring into Modular Structure (#3)

The entire codebase has been refactored from a single-file implementation into a well-organized, modular architecture:

- **`cli.py`**: Command-line interface and argument parsing
- **`config.py`**: Configuration models and validation (Pydantic-based)
- **`workflow.py`**: High-level pipeline orchestration
- **`rss_parser.py`**: RSS feed parsing and episode extraction
- **`downloader.py`**: Resilient HTTP download utilities
- **`episode_processor.py`**: Episode-level processing logic
- **`filesystem.py`**: Filesystem utilities and path management
- **`whisper_integration.py`**: Whisper transcription integration
- **`progress.py`**: Pluggable progress reporting interface
- **`models.py`**: Shared data models (dataclasses)

This refactoring improves:

- **Maintainability**: Clear separation of concerns
- **Testability**: Isolated modules with focused responsibilities
- **Extensibility**: Easy to add new features without touching core logic
- **API Stability**: Clean public API surface (`Config`, `run_pipeline`, `load_config_file`)

#### Comprehensive Documentation (#9)

Added extensive documentation infrastructure:

- **Architecture Documentation** (`docs/ARCHITECTURE.md`): Complete system architecture overview
- **Product Requirements Documents (PRDs)**:
  - PRD-001: Transcript Acquisition Pipeline
  - PRD-002: Whisper Fallback Transcription
  - PRD-003: User Interfaces & Configuration
- **Request for Comments (RFCs)**: 10+ RFCs documenting design decisions
  - RFC-001 through RFC-010 covering all major features
- **Testing Strategy** (`docs/TESTING_STRATEGY.md`): Comprehensive testing approach
- **API Documentation**: Migration guides and API comparisons
- **MkDocs Site**: Live documentation at <https://chipi.github.io/podcast_scraper/>

### ✨ New Features

#### RFC-010: Automatic Speaker Name Detection (#11)

- **Named Entity Recognition (NER)**: Automatically extracts host and guest names from episode metadata using spaCy
- **Language-Aware Processing**: Single `language` configuration drives both Whisper model selection and NER
- **Smart Model Selection**: Automatically prefers English-only Whisper models (`.en` variants) for better performance
- **Host/Guest Distinction**: Intelligently identifies recurring hosts vs. episode-specific guests
- **Caching Support**: Optional host detection caching across episodes for performance
- **Graceful Fallback**: Works seamlessly when spaCy is unavailable

### 🔧 Improvements

- **MIT License**: Added proper license file and documentation
- **CI/CD Enhancements**: Improved workflows and tooling alignment (#10)
- **Code Quality**: Various code review improvements (#8)
- **Security**: CodeQL analysis workflow configuration
- **Type Safety**: Full type hints and Pydantic validation

### 📚 Documentation Highlights

- Complete architecture documentation
- 10+ RFCs covering design decisions
- Testing strategy and requirements
- API migration guides (v1 → v2)
- Product requirements documents
- Live MkDocs documentation site

### 🔄 Migration Notes

For users upgrading from v1.0.0:

- The public API remains stable (`Config`, `run_pipeline`, `load_config_file`)
- Internal module structure has changed but is transparent to users
- New optional features (speaker detection) are opt-in via `--auto-speakers`
- All existing functionality preserved and enhanced

**Full Changelog**: <https://github.com/chipi/podcast_scraper/compare/v1.0.0...v2.0.0>
