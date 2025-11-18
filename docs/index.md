# Podcast Scraper Documentation

Welcome! This site captures the knowledge required to understand, operate, and extend the `podcast_scraper` project.

## Quick Links

- **[API Reference](api/API_REFERENCE.md)** - Complete public API documentation with examples
- **[Architecture](ARCHITECTURE.md)** - High-level system design and module responsibilities
- **[API Boundaries](api/API_BOUNDARIES.md)** - API design principles and boundaries

## What you will find here

- **API Reference** — Complete documentation of all public functions, classes, and their parameters
- **Architecture overview** — High-level system design in `ARCHITECTURE.md`
- **Product Requirements (PRDs)** — Intent and functional expectations for major capabilities (transcript acquisition, Whisper fallback, user interfaces)
- **Requests for Comment (RFCs)** — Technical specifications for the modules that implement each capability
- **Guides** — API migration references and API boundaries documentation

## Getting Started

1. Review the [Architecture](ARCHITECTURE.md) to understand system boundaries and major components.
2. Read the PRDs to learn the *why* behind each feature area.
3. Dive into the RFCs for implementation details when building or modifying functionality.
4. Consult the Guides when migrating from earlier versions or comparing API changes.

## Local Development

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Visit [http://localhost:8000](http://localhost:8000) to preview the site. The page updates automatically as you edit Markdown files.

## Deployment

This repository includes a GitHub Actions workflow that builds the site with MkDocs and publishes it to GitHub Pages on every push to `main`.
