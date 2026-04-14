# Release History

All releases of the Podcast Scraper project.

Before tagging a release, follow the [Release checklist](../guides/DEVELOPMENT_GUIDE.md) in the Development Guide (search for "Release checklist"). In particular, run **`make release-docs-prep`** so architecture diagrams and release notes are up to date; the docs site uses committed diagrams only, and CI fails if they are stale.

## Latest Release

- **[v2.6.0](RELEASE_v2.6.0.md)** — GI/KG viewer (Library, Digest, Dashboard, Graph, Search), semantic search, multi-feed corpus, Run Compare Performance + frozen profiles (April 2026)
- **[v2.5.0](RELEASE_v2.5.0.md)** — LLM Provider Expansion & Production Hardening (released February 2026)
- **[v2.4.0](RELEASE_v2.4.0.md)** — Provider ecosystem & production readiness

## All Releases

| Version | Highlights |
| --------- | ------------ |
| [v2.6.0](RELEASE_v2.6.0.md) | Viewer v2 (Library, Digest, Dashboard, Graph, Search), FAISS semantic search, multi-feed corpus, Run Compare Performance tab, RFC-064 frozen profiles, live monitor |
| [v2.5.0](RELEASE_v2.5.0.md) | LLM provider expansion (6 cloud + Ollama local), production hardening, LLM metrics, MPS exclusive mode |
| [v2.4.0](RELEASE_v2.4.0.md) | Multi-provider ecosystem (8 providers), production defaults, cache management |
| [v2.3.2](RELEASE_v2.3.2.md) | Security tests, thread-safety fixes |
| [v2.3.1](RELEASE_v2.3.1.md) | Security fixes, code quality improvements |
| [v2.3.0](RELEASE_v2.3.0.md) | Episode summarization, public API, cleaned transcripts |
| [v2.2.0](RELEASE_v2.2.0.md) | Metadata generation, code quality improvements |
| [v2.1.0](RELEASE_v2.1.0.md) | Automatic speaker detection using NER |
| [v2.0.1](RELEASE_v2.0.1.md) | Bug fixes and stability improvements |
| [v2.0.0](RELEASE_v2.0.0.md) | Modular architecture, public API foundation |
| [v1.0.0](RELEASE_v1.0.0.md) | Initial release |

## Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0) — Breaking changes
- **Minor** (0.X.0) — New features, backwards compatible
- **Patch** (0.0.X) — Bug fixes, backwards compatible
