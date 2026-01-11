# ADR-015: Secure Credential Injection

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-013](../rfc/RFC-013-openai-provider-implementation.md)

## Context & Problem Statement

Integrating Cloud APIs (OpenAI, etc.) requires API keys. Hardcoding these or putting them in tracked configuration files is a major security risk.

## Decision

We mandate **Environment-Based Credential Injection**:

1. Secrets are never stored in `Config` files or source code.
2. The application uses `python-dotenv` to load a local `.env` file (which is strictly gitignored).
3. In CI/CD or production, keys are injected via standard system environment variables (`OPENAI_API_KEY`).

## Rationale

- **Security**: Prevents accidental leaks of paid API keys.
- **Portability**: Standardizes how keys are provided across Local, Docker, and GitHub Actions.
- **Developer Experience**: `.env` files provide a simple way to manage keys locally without polluting the shell profile.

## Alternatives Considered

1. **CLI Flag only**: Rejected as it's insecure (keys show up in process lists) and tedious for frequent runs.

## Consequences

- **Positive**: Secure by default; follows industry best practices.
- **Negative**: Requires users to perform a one-time setup of a `.env` file.

## References

- [RFC-013: OpenAI Provider Implementation](../rfc/RFC-013-openai-provider-implementation.md)
