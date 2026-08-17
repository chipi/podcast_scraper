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

from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from podcast_scraper.workflow.stages import processing
from podcast_scraper.workflow.types import HostDetectionResult

pytestmark = [pytest.mark.unit]


class _RaisingDetector:
    """A detector whose ``detect_speakers`` raises whatever it was constructed with."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def detect_speakers(
        self,
        episode_title: str = "",
        episode_description: str = "",
        known_hosts: Any = None,
        **_kwargs: Any,
    ):
        raise self._exc


class _LedgerSpy:
    """Minimal stand-in for Metrics that records only the stage-outcome rows."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, Optional[str]]] = []

    def record_stage_outcome(
        self, stage: str, _idx: Any, outcome: str, reason: Optional[str] = None, **_kw: Any
    ) -> None:
        if stage == "speaker_detection":
            self.rows.append((outcome, reason))

    def __getattr__(self, _name: str):  # every other metrics call is a no-op here
        return lambda *a, **k: None


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


def _run(exc: BaseException) -> Tuple[Optional[Any], List[Tuple[str, Optional[str]]]]:
    cfg = MagicMock()
    cfg.auto_speakers = True
    cfg.dry_run = False
    cfg.screenplay_speaker_names = []
    cfg.cache_detected_hosts = False
    cfg.speaker_detector_provider = "ml"
    cfg.known_hosts = []

    episode = MagicMock()
    episode.idx = 1
    episode.title = "An Episode"
    episode.description = "A description."

    hdr = HostDetectionResult(
        cached_hosts=set(), heuristics=None, speaker_detector=_RaisingDetector(exc)
    )
    spy = _LedgerSpy()
    result = processing._detect_speakers_for_episode(episode, cfg, hdr, spy)
    return result, spy.rows


def test_missing_package_degrades_instead_of_killing_the_run() -> None:
    """THE fix: a missing optional package must not end the run.

    The pipeline already has a no-detector path and already logged that it was taking one.
    """
    result, _rows = _run(_missing_package_error())
    assert result is None or (not result.guests and not result.stated)


def test_the_degrade_is_recorded_not_silent() -> None:
    """Loud, per the FFmpeg decision — a silent skip is the failure #1647 exists to prevent.

    ``degraded``, not ``ran``: the stage did NOT do its job, and that must stay visible.
    """
    _result, rows = _run(_missing_package_error())
    assert rows, "the stage recorded nothing at all — that is the #1646 silence"
    outcome, reason = rows[-1]
    assert outcome == "degraded", rows
    assert reason and ("packag" in reason or "unavailable" in reason), rows


def test_a_real_detector_failure_still_raises() -> None:
    """The other half. ONLY a missing package degrades.

    A detector that blows up for any other reason — a bad key, a timeout, a bug — is a real
    failure and must still stop the run. Degrading on everything would be the silent-damage
    pattern this whole epic exists to remove.
    """
    with pytest.raises(RuntimeError, match="quota"):
        _run(RuntimeError("provider quota exhausted"))


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
