# ADR-041: Mandatory Pre-Release Validation

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-038](../rfc/RFC-038-continuous-review-tooling.md)

## Context & Problem Statement

Releasing a version with a broken docs build, a missing entry in the changelog, or a version mismatch in `pyproject.toml` is a common but preventable mistake. Manual checklists are often skipped or forgotten.

## Decision

We enforce **Mandatory Pre-Release Validation**.

- All official releases MUST pass the automated `scripts/pre_release_check.py` script.
- The check is accessible via `make pre-release` and validates:
  - All tests passing.
  - Zero linting/type-check errors.
  - Successful `mkdocs` build.
  - Changelog contains the new version.
  - Version consistency across all project files.

## Rationale

- **Quality Guarantee**: Ensures that every public release meets the project's high standards.
- **Trust**: Users and contributors can trust that a tagged version is fully validated and documented.
- **Automation**: Removes the mental burden of remembering 10+ manual checks before hitting "publish."

## Alternatives Considered

1. **Manual Checklists**: Rejected as they are prone to human error and omission.

## Consequences

- **Positive**: Highly professional release process; zero "broken release" incidents.
- **Negative**: Adds a final "gate" that must be cleared before merging a release branch.

## References

- [RFC-038: Continuous Review Tooling Implementation](../rfc/RFC-038-continuous-review-tooling.md)
- [DEVELOPMENT_GUIDE.md](../guides/DEVELOPMENT_GUIDE.md)
