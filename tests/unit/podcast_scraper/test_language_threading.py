"""The resolved language reaches the provider, and nothing substitutes "en" (#2177 / slice S0.6).

The hazard is silent by construction. A non-English episode transcribed as English does not
raise — it produces a plausible transcript of the wrong words, which summary, GI, KG and search
all then trust. So these tests assert the absence of a substitution, which is harder to see than
a wrong value and is exactly why the lint exists alongside them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from podcast_scraper import config as config_mod
from podcast_scraper.languages import transcription_language

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]


def _cfg(*, language: str = "en", override: Optional[str] = None) -> Any:
    # #2283: the profile `language` no longer gives an episode a language; the feed's tag
    # does. A test's language goes there, and an unstated one is an English feed.
    cfg = config_mod.Config(rss="https://example.com/f.xml", feed_declared_language=language)
    if override is not None:
        cfg = cfg.model_copy(update={"language_override": override})
    return cfg


class TestTheOneReader:
    def test_it_applies_the_full_precedence(self) -> None:
        assert transcription_language(_cfg(language="en")) == "en"
        assert transcription_language(_cfg(language="es")) == "es"
        assert transcription_language(_cfg(language="en", override="es")) == "es"

    def test_it_normalizes(self) -> None:
        """A profile carrying ``en-US`` must reach the provider as ``en``, or ``is_english``
        fails at ``whisper_utils.py:50`` and a non-``.en`` model is chosen."""
        assert transcription_language(_cfg(language="en-US")) == "en"

    def test_it_never_returns_a_fabricated_default(self) -> None:
        """``None`` is an honest "nobody resolved a language" — let the engine decide.

        ``"en"`` would be an assertion we cannot make, and making it is what produced a
        plausible English transcript of a Spanish episode.
        """
        cfg = config_mod.Config(rss="https://example.com/f.xml")  # no feed tag, no override
        assert transcription_language(cfg) is None


class TestTheSubstitutionsAreGone:
    def test_ml_provider_does_not_substitute_en(self) -> None:
        """``effective_language = language if ... else (self.cfg.language or "en")`` was the
        mechanism. Asserted against the source because the alternative is constructing a real
        MLProvider, which needs torch."""
        src = (REPO / "src/podcast_scraper/providers/ml/ml_provider.py").read_text(encoding="utf-8")
        assert 'self.cfg.language or "en"' not in src, (
            "the 'en' substitution is back in ml_provider — a non-English episode will be "
            "transcribed as English by a chain whose local default is base.en"
        )

    def test_the_dgx_provider_reports_what_the_server_said(self) -> None:
        """``"language": language or "en"`` in the RESULT dict was a provenance lie: the request
        omits `language` when it is None, so the server auto-detects, and the client used to
        overwrite the answer with "en"."""
        src = (REPO / "src/podcast_scraper/providers/tailnet_dgx/whisper_provider.py").read_text(
            encoding="utf-8"
        )
        assert '"language": language or "en"' not in src
        assert '"language": language or self._last_detected_language' in src

    def test_the_dgx_provider_starts_with_an_honest_none(self) -> None:
        """A call that fails before reaching the transport must not raise AttributeError, and
        must not claim a language either."""
        from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
            TailnetDgxWhisperTranscriptionProvider,
        )

        cfg = _cfg()
        provider = TailnetDgxWhisperTranscriptionProvider(cfg)
        assert provider._last_detected_language is None


class TestNoStageCanShipTheWrongLanguage:
    def test_an_episode_resolving_to_es_never_reports_en(self) -> None:
        """The regression the slice is judged by, stated as the artifact contract.

        `_build_feed_metadata` is what writes the language into the episode artifact, so this is
        the value every downstream consumer reads.
        """
        from podcast_scraper.workflow.metadata_generation import _build_feed_metadata

        feed = MagicMock()
        feed.title = "Show"
        feed.authors = []
        feed.language = "es-ES"

        meta = _build_feed_metadata(
            feed, "https://example.com/f.xml", "p01", _cfg(language="en"), None, None, None
        )
        assert meta.language == "es", "the feed's own tag must win over the profile default"
        assert meta.language_raw == "es-ES"
        assert meta.language_source == "rss"

    def test_the_transcription_call_site_asks_the_resolver(self) -> None:
        """Asserted structurally: the call site must not read cfg.language directly. The lint
        enforces this repo-wide; this names the specific site the hazard lived at."""
        src = (REPO / "src/podcast_scraper/workflow/episode_processor.py").read_text(
            encoding="utf-8"
        )
        assert "language=transcription_language(cfg)," in src
        assert "language=cfg.language," not in src


class TestTheLintItself:
    """A lint that cannot fail is not a lint."""

    def test_it_passes_on_the_current_tree(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check/lint_language_readers.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "read only by languages.py" in result.stdout

    def test_it_fires_on_an_unwhitelisted_read(self, tmp_path: Path) -> None:
        """Inject the exact shape the lint exists to catch, into a file not on the whitelist."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "lint_language_readers", REPO / "scripts/check/lint_language_readers.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        offender = tmp_path / "some_new_stage.py"
        offender.write_text("def go(cfg):\n    return provider(language=cfg.language)\n")
        assert mod._reads(offender), "the detector did not see a plain cfg.language read"

        clean = tmp_path / "well_behaved.py"
        clean.write_text(
            "def go(cfg):\n    return provider(language=transcription_language(cfg))\n"
        )
        assert not mod._reads(clean), "the detector flagged the CORRECT form"

    def test_it_does_not_flag_prose_or_the_override(self, tmp_path: Path) -> None:
        """Comments documenting the hazard, and `cfg.language_override`, are not reads."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "lint_language_readers", REPO / "scripts/check/lint_language_readers.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        f = tmp_path / "prose.py"
        f.write_text(
            "# a comment mentioning cfg.language for documentation\n"
            "def go(cfg):\n"
            "    return cfg.language_override\n"
        )
        assert not mod._reads(f)

    def test_it_runs_inside_make_lint(self) -> None:
        """The precedent lint (`lint-search-v3`) is a standalone target that runs in NO workflow,
        so the guard it provides has never gated a PR. This one lives inside `lint`, which
        python-app.yml, nightly.yml and ci-fast all run."""
        makefile = (REPO / "Makefile").read_text(encoding="utf-8")
        lint_body = makefile.split("\nlint:\n", 1)[1].split("\n\n", 1)[0]
        assert (
            "lint_language_readers.py" in lint_body
        ), "the lint is not inside the `lint` target, so CI does not run it"
