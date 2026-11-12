"""A missing optional PACKAGE degrades speaker detection; anything else still fails the run.

THE CONTRADICTION THIS RESOLVES
Building the speaker detector already tolerates a missing spaCy — the pipeline logs *"Failed to
initialize spaCy (speaker detection will be unavailable)"* and carries on. One layer down,
``analyze_patterns`` -> ``_initialize_spacy`` -> ``import spacy`` raised ``ModuleNotFoundError``
and killed the whole run. The code announced a degrade it did not perform: the run died anyway,
several stages later, with a stack trace instead of the warning it had already printed.

That was never a decision anyone made. The FFmpeg call (#26) WAS deliberate — missing ffmpeg is
FATAL, because preprocessing determines whether a transcript is correct at all. Speaker detection
is different in kind: it is an enhancement over episode metadata, the pipeline already has a
no-detector path (``auto_speakers=False`` returns immediately), and the code already claimed to
take it.

MEASURED 2026-08-17: 33 of the 55 e2e tests that fail without the ML stack pass with
``auto_speakers=False`` and need no ML whatsoever — CLI, config-file, error-recovery, HTTP and
concurrency tests, dragged in by one default. The other 22 genuinely need torch/transformers.

DEGRADE LOUDLY, NOT SILENTLY — the part carried over from the FFmpeg decision. The stage records a
``degraded`` ledger row with its own reason slug, so an episode whose speakers were never detected
stays distinguishable from one where detection ran and honestly found nobody. #1647 exists to keep
exactly that distinction, and a silent skip would destroy it as surely as a crash.

ONE MECHANISM, NOT FIVE: the missing-package test is
``utils.optional_deps.caused_by_missing_import`` — the same walker
``preload_ml_models_if_needed`` uses (95be1ec1), promoted out of that module rather than
reimplemented here. Two call sites deciding "is this a missing dependency?" by different rules is
how a codebase ends up needing archaeology.
"""

# mypy: disable-error-code="arg-type"
# Deliberate: _LedgerSpy is a duck-typed stand-in passed where Metrics is declared. Building a
# real Metrics would drag in the machinery this test isolates. Same convention as the other
# test files that pass doubles into production signatures.

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


def _missing_package_error() -> Exception:
    """The real shape: a provider error wrapping an ImportError via IMPLICIT chaining.

    ``_initialize_spacy`` re-raises bare inside ``except ImportError``, so the ImportError lands in
    ``__context__`` while ``__cause__`` stays None — the case a naive ``__cause__`` check misses,
    which is why the shared walker follows both links.
    """
    try:
        try:
            raise ModuleNotFoundError("No module named 'spacy'")
        except ModuleNotFoundError:
            raise RuntimeError("spaCy speaker detection unavailable")
    except RuntimeError as exc:
        return exc


def test_the_missing_package_check_is_the_shared_one() -> None:
    """Consistency: one walker, used by both the preload path and this one.

    ``preload_ml_models_if_needed`` and speaker detection must answer "is this a missing optional
    dependency?" identically, or the pipeline degrades in one place and dies in the other for the
    very same cause — which is exactly the state this commit found.
    """
    from podcast_scraper.utils.optional_deps import caused_by_missing_import
    from podcast_scraper.workflow.stages import setup

    assert setup._caused_by_missing_import is caused_by_missing_import
    assert caused_by_missing_import(_missing_package_error()) is True
    assert caused_by_missing_import(RuntimeError("quota")) is False
