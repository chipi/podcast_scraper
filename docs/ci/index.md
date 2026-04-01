# CI/CD Documentation

This section contains comprehensive documentation for the Podcast Scraper CI/CD pipeline,
including workflow details, local development guides, and metrics dashboards.

## Quick Navigation

| Document | Description |
| --------- | ------------- |
| [Overview](OVERVIEW.md) | Architecture, optimization strategies, and high-level concepts |
| [Workflows](WORKFLOWS.md) | Detailed documentation for all GitHub Actions workflows |
| [Resource Usage](RESOURCE_USAGE.md) | Resource usage strategy and limits |
| [Snyk Setup](SNYK_SETUP.md) | Security scanning setup and configuration |
| [Local Development](LOCAL_DEVELOPMENT.md) | Pre-commit hooks, local CI validation, and debugging |
| [Test dashboard](METRICS.md) | GitHub Pages unified dashboard (CI + Nightly); see also [Code quality trends](CODE_QUALITY_TRENDS.md) |

## Workflows Summary

The CI/CD pipeline consists of **six main workflows**:

| Workflow | File | Purpose | Trigger |
| ---------- | ------ | --------- | --------- |
| **Python Application** | `python-app.yml` | Main CI pipeline with testing, linting, and builds | Push/PR to `main` (only when Python/config files change) |
| **Documentation Deploy** | `docs.yml` | Build and deploy MkDocs documentation to GitHub Pages | Push to `main`, PR with doc changes, manual |
| **CodeQL Security** | `codeql.yml` | Security vulnerability scanning | Push/PR to `main` (only when code/workflow files change), scheduled weekly |
| **Docker Build & Test** | `docker.yml` | Build and test Docker images | Push to `main` (all), PRs (Dockerfile/.dockerignore only) |
| **Snyk Security Scan** | `snyk.yml` | Dependency and Docker image vulnerability scanning | Push/PR to `main`, scheduled weekly (Mondays), manual |
| **Nightly Comprehensive** | `nightly.yml` | Full test suite with comprehensive metrics collection | Scheduled daily (2 AM UTC), manual |

## Key Features

- ✅ **Path-based optimization** - Workflows only run when relevant files change
- ✅ **Parallel execution** - Jobs run simultaneously for faster feedback
- ✅ **Two-tier testing** - Fast critical path tests on PRs, full suite on main
- ✅ **Comprehensive security** - CodeQL, Snyk, Dependabot, and bandit scanning
- ✅ **Unified metrics** - Single dashboard for CI and Nightly metrics
- ✅ **Local validation** - `make ci` runs full CI suite locally

## Getting Started

1. **New to CI/CD?** Start with [Overview](OVERVIEW.md) to understand the architecture
2. **Setting up locally?** See [Local Development](LOCAL_DEVELOPMENT.md) for pre-commit hooks and validation
3. **Understanding workflows?** Check [Workflows](WORKFLOWS.md) for detailed documentation
4. **Viewing metrics?** See [Test dashboard](METRICS.md) and [Code quality trends](CODE_QUALITY_TRENDS.md)

## Related Documentation

- [Testing Strategy](../TESTING_STRATEGY.md) - Overall testing approach
- [Development Guide](../guides/DEVELOPMENT_GUIDE.md) - Development environment setup
- [Git Worktree Guide](../guides/GIT_WORKTREE_GUIDE.md) - Development workflow with worktrees
