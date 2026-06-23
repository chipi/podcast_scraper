"""Runtime environment detection — narrow pytest-only signals.

Single source of truth for "are we under pytest right now?" The detection
was historically duplicated in ``config.py`` and
``evaluation/autoresearch_track_a.py`` (both gating .env-file loading
during pytest runs for hermeticity). Each duplicate had its OWN buggy
``"unittest" in sys.modules`` check that false-positived in production
(numpy lazy-imports ``numpy.testing`` → unittest gets pulled in for
every prod code path that imports numpy). See commit ``ce029849`` for
the narrowing fix and ``docs/wip/POST_RFC097_DEV_PROD_REMOVAL.md`` for
the full chapter.

This module is the consolidated home. Both call sites import from here.

What this is NOT: a general "test vs prod" mode selector. The operator's
direction (2026-06-23) is that profiles, not runtime heuristics, are the
source of truth for "what defaults apply here". This helper exists ONLY
for the narrow case of "don't read the operator's ``.env`` file into the
pytest process" — a hermeticity guard, not a default-flipping mechanism.
"""

from __future__ import annotations

import os
import sys


def is_pytest_run() -> bool:
    """True iff we're under pytest right now.

    Used to gate `.env` / `.env.autoresearch` file loading so pytest tests
    stay hermetic with respect to shell-exported secrets in the operator's
    .env. Returns True only on EXPLICIT pytest / TESTING signals — never
    on ``"unittest" in sys.modules`` (the false-positive that bit us for
    5 months; see commit ce029849).
    """
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return True
    if os.environ.get("TESTING", "").lower() in ("1", "true", "yes"):
        return True
    return False
