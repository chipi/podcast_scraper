#!/bin/bash
#
# Script to close resolved issues with appropriate comments linking to PRs
#
# Usage: ./close_resolved_issues.sh
#
# This script closes issues that have been resolved by merged PRs

set -e

echo "Closing resolved issues with PR references..."
echo ""

# Issue #53 - Resolved by PR #54
echo "Closing issue #53 (Fixed by PR #54)..."
gh issue close 53 --comment "Resolved by PR #54 which fixed the RSS parsing deprecation warnings that were causing CI test failures.

See: https://github.com/chipi/podcast_scraper/pull/54"

# Issue #48 - Resolved by PR #55
echo "Closing issue #48 (Fixed by PR #55)..."
gh issue close 48 --comment "Resolved by PR #55 which created a comprehensive CONTRIBUTING.md document covering all coding guidelines including:
- Code style guidelines (Black, isort, naming conventions, type hints)
- Development workflow (Git workflow, commit messages, module boundaries)
- Testing requirements (unit tests, integration tests, mocking patterns)
- Documentation standards (when to create PRDs/RFCs)
- CI/CD integration
- Architecture principles

See: https://github.com/chipi/podcast_scraper/pull/55"

# Issue #44 - Resolved by PR #55
echo "Closing issue #44 (Fixed by PR #55)..."
gh issue close 44 --comment "Resolved by PR #55 which added comprehensive contributing guidelines that both human and AI contributors can follow. The CONTRIBUTING.md provides:
- Clear decision trees for when to create PRDs/RFCs vs direct implementation
- Coding patterns and best practices
- Module boundary guidelines
- Testing and error handling patterns
- Configuration patterns

These guidelines enable AI coding assistants to understand project conventions and make appropriate decisions.

See: https://github.com/chipi/podcast_scraper/pull/55"

# Issue #38 - Resolved by PR #33
echo "Closing issue #38 (Fixed by PR #33)..."
gh issue close 38 --comment "Resolved by PR #33 which implemented comprehensive E2E library API tests in test_integration.py:
- Tests for \`podcast_scraper.run_pipeline()\` direct usage
- Tests for \`podcast_scraper.load_config_file()\`
- Validates no CLI dependencies leak into core pipeline
- Tests error handling
- Demonstrates config loading from Python objects and files

See: https://github.com/chipi/podcast_scraper/pull/33"

# Issue #35 - Resolved by PR #52
echo "Closing issue #35 (Fixed by PR #52)..."
gh issue close 35 --comment "Resolved by PR #52 which implemented the metrics.py module with:
- Structured logging capabilities
- Run summary reporting
- Metrics collection during pipeline execution

This provides consolidated pipeline statistics and outcomes at the end of each run.

See: https://github.com/chipi/podcast_scraper/pull/52"

# Issue #17 - Resolved by PR #52
echo "Closing issue #17 (Fixed by PR #52)..."
gh issue close 17 --comment "Resolved by PR #52 which implemented the summarizer.py module for generating episode summaries and key takeaways from transcripts:
- Support for multiple transformer models (BART, PEGASUS, LED variants)
- Integration with metadata generation pipeline (from PR #33)
- Automated key takeaway extraction

See: https://github.com/chipi/podcast_scraper/pull/52"

echo ""
echo "✅ All resolved issues have been closed!"
echo ""
echo "Note: Issues #39 and #19 remain open as they are not fully resolved:"
echo "  - #39: Public API documentation not yet created"
echo "  - #19: Multiple progress indicators issue not functionally fixed"
