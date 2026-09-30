"""The S0.6 lint is only as true as the spellings it can see.

A whole-branch review on 2026-09-30 quoted this lint's own output — "cfg.language is read only by
languages.py and 7 whitelisted file(s)" — as evidence that the single-reader property held. It
did not. The pattern matched only the dotted form, so **11 reads** written as
`getattr(self.cfg, "language", "en")` were invisible, and two of the files containing them
(grok, anthropic) were not whitelisted at all.

That is the failure mode worth testing: not "does the lint pass" — it passed the whole time —
but "can the lint SEE a read". So these assert on SHAPES, by feeding the detector lines directly.
A count would go stale on the next provider added; a shape will not, and a third spelling nobody
anticipated is exactly what this is here to make visible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check" / "lint_language_readers.py"


@pytest.fixture(scope="module")
def lint() -> Any:
    spec = importlib.util.spec_from_file_location("lint_language_readers_under_test", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheShapesItMustSee:
    @pytest.mark.parametrize(
        "line",
        [
            "    language = cfg.language",
            "    language = self.cfg.language",
            "    language = config.language",
            "    language = self._cfg.language",
            "    return cfg.language or None",
            # The half that was missing, and every variant of it that exists in src/.
            '    language = getattr(self.cfg, "language", "en") or None',
            '    language = getattr(cfg, "language", None)',
            "    language = getattr(self.cfg, 'language', 'en')",
            '    profile_default=getattr(config, "language", None),',
            '    x = getattr( self.cfg , "language" , "en" )',
        ],
    )
    def test_it_is_detected(self, lint: Any, line: str) -> None:
        assert lint._is_read(line), f"blind to: {line.strip()}"


class TestTheShapesItMustNOTSee:
    @pytest.mark.parametrize(
        "line",
        [
            # The whole point of the `(?!_)` guard: the override is a DIFFERENT field, and
            # matching it would make every legitimate override read a violation.
            "    o = cfg.language_override",
            "    s = cfg.language_source",
            '    o = getattr(cfg, "language_override", None)',
            "    from ..languages import transcription_language",
            "    language = transcription_language(cfg)",
            "    langs = cfg.languages",
            # A different object that merely has a `.language`.
            "    language = feed.language",
            "    language = episode.language",
            '    language = getattr(feed, "language", None)',
        ],
    )
    def test_it_is_not_detected(self, lint: Any, line: str) -> None:
        assert not lint._is_read(line), f"false positive on: {line.strip()}"


class TestTheRealTreeIsFullyAccountedFor:
    def test_every_file_that_reads_it_is_whitelisted(self, lint: Any) -> None:
        """The lint's actual guarantee, asserted independently of its exit code."""
        unlisted = [
            str(path.relative_to(lint.SRC))
            for path in sorted(lint.SRC.rglob("*.py"))
            if lint._reads(path) and str(path.relative_to(lint.SRC)) not in lint.WHITELIST
        ]
        assert unlisted == []

    def test_the_whitelist_has_no_stale_entries(self, lint: Any) -> None:
        """A stale entry is a standing permission for a read nobody makes, which is how a real
        violation later slips in under cover of an existing line."""
        reading = {
            str(path.relative_to(lint.SRC)) for path in lint.SRC.rglob("*.py") if lint._reads(path)
        }
        assert sorted(set(lint.WHITELIST) - reading) == []

    def test_every_entry_states_a_reason(self, lint: Any) -> None:
        """The whitelist's grain is file-plus-reason on purpose: adding a line has to be argued
        for, and an empty reason makes it an escape hatch instead."""
        for rel, reason in lint.WHITELIST.items():
            assert reason.strip(), f"{rel} has no stated reason"
            assert len(reason.split()) >= 4, f"{rel}: {reason!r} is not a reason"

    def test_the_providers_that_were_MISSING_are_present(self, lint: Any) -> None:
        """Pinned by name. These two were invisible to the pattern AND absent from the
        whitelist, so nothing anywhere recorded that they read the run-global."""
        assert "providers/grok/grok_provider.py" in lint.WHITELIST
        assert "providers/anthropic/anthropic_provider.py" in lint.WHITELIST


class TestProseIsNotARead:
    @pytest.mark.parametrize(
        "line",
        [
            "    # cfg.language is read only by languages.py",
            '    """Reads cfg.language as a fallback."""',
        ],
    )
    def test_comments_and_docstrings_are_skipped(self, lint: Any, line: str) -> None:
        """A false positive here is cheap and visible; a false negative is the whole problem."""
        assert lint._is_comment_or_doc(line)
