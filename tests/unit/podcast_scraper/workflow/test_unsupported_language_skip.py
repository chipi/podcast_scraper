"""An episode in a language we do not ingest is refused, visibly (#2179 / slice S0.8).

The skip and the ability to SEE it are one change. Shipping the refusal without the log line, the
incident and the ledger row would add a new silent failure to the phase whose entire purpose is
removing them — the episode would simply never appear, with nothing saying why.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from podcast_scraper import config as config_mod
from podcast_scraper.workflow.episode_processor import _unsupported_language_skip_reason

pytestmark = pytest.mark.unit


def _cfg(
    *, language: str = "en", override: Optional[str] = None, feed: Optional[str] = None
) -> Any:
    cfg = config_mod.Config(rss="https://example.com/f.xml", language=language)
    update: dict[str, Any] = {}
    if override is not None:
        update["language_override"] = override
    if feed is not None:
        # The channel tag as `run_pipeline` records it. Reachable only since #2172 wired the
        # pipeline's own `RssFeed`; before that this source could not be produced at all, which
        # is why every test here used to exercise only two of the three.
        update["feed_declared_language"] = feed
    return cfg.model_copy(update=update) if update else cfg


class TestWhatIsRefused:
    def test_a_language_the_registry_does_not_enable_is_refused(self) -> None:
        reason = _unsupported_language_skip_reason(_cfg(language="es"))
        assert reason is not None
        assert "'es'" in reason
        assert "not enabled in config/languages.yaml" in reason

    def test_english_proceeds(self) -> None:
        assert _unsupported_language_skip_reason(_cfg(language="en")) is None

    def test_a_regional_english_tag_proceeds(self) -> None:
        """``en-US`` normalizes to ``en`` in Config, so this must not read as a foreign language.

        Before S0.2 it became ``"en-us"``, which is not in the registry — so this exact gate would
        have refused every episode of an English corpus configured that way.
        """
        assert _unsupported_language_skip_reason(_cfg(language="en-US")) is None

    @pytest.mark.parametrize("code", ["de", "ja", "ar", "ru", "ca"])
    def test_every_described_but_disabled_language_is_refused(self, code: str) -> None:
        """Present-in-the-registry but ``enabled: false`` must refuse, not proceed. Being
        described is not being ingested."""
        assert _unsupported_language_skip_reason(_cfg(language=code)) is not None

    def test_an_unknown_language_is_refused(self) -> None:
        """Absent from the registry entirely — never a lenient default."""
        assert _unsupported_language_skip_reason(_cfg(language="xx")) is not None


class TestTheOverrideIsTheRemedy:
    def test_an_override_rescues_a_mis_tagged_feed(self) -> None:
        """THE ORDERING THIS SLICE DEPENDS ON.

        Publisher tags are routinely wrong. An English show tagged ``de`` would otherwise stop
        ingesting on every run — including relabels and rederives — with no remedy at all. The
        override is that remedy, which is why S0.8 must not land before S0.2.
        """
        assert _unsupported_language_skip_reason(_cfg(language="de")) is not None
        assert _unsupported_language_skip_reason(_cfg(language="de", override="en")) is None

    def test_an_override_can_also_refuse(self) -> None:
        """It is a correction, not a bypass: pointing a feed at a disabled language still skips."""
        assert _unsupported_language_skip_reason(_cfg(language="en", override="es")) is not None


class TestTheReasonIsUsable:
    def test_it_names_the_language_and_where_it_came_from(self) -> None:
        """One sentence carries the log line, the incident and the ledger row, so an operator
        reading any one of the three learns the same thing.

        All THREE sources, because the source used to be derived from `language_override` alone:
        a language that came from the feed's declared tag was reported as "the profile default",
        pointing an operator at the wrong thing to change.
        """
        from_profile = _unsupported_language_skip_reason(_cfg(language="es"))
        assert from_profile is not None and "the profile default" in from_profile

        from_override = _unsupported_language_skip_reason(_cfg(language="en", override="es"))
        assert from_override is not None and "the per-feed override" in from_override

        from_feed = _unsupported_language_skip_reason(_cfg(language="en", feed="es"))
        assert from_feed is not None and "the feed's declared <language> tag" in from_feed

    def test_the_remedy_is_PER_SOURCE(self) -> None:
        """A single remedy sentence was wrong in two of the three cases: it advised setting a
        per-feed override when the override was already the cause, and blamed a wrong publisher
        tag when no tag was involved at all. An operator acts on this sentence."""
        override = _unsupported_language_skip_reason(_cfg(language="en", override="es")) or ""
        assert "correct the per-feed `language:` override that set it" in override

        feed = _unsupported_language_skip_reason(_cfg(language="en", feed="es")) or ""
        assert "if the publisher's tag is wrong" in feed

        profile = _unsupported_language_skip_reason(_cfg(language="es")) or ""
        assert "point this feed at a profile" in profile
        assert "publisher" not in profile, "no publisher tag was involved"

    def test_it_says_what_to_do_about_it(self) -> None:
        reason = _unsupported_language_skip_reason(_cfg(language="es"))
        assert reason is not None
        assert "Enable it there" in reason

    def test_a_tag_that_is_not_a_usable_LANGUAGE_is_reported_as_such(self) -> None:
        """`profile_default` covers two corpus states — no tag, and a tag resolution could not
        read. Saying "the feed declared no <language>" for the second is a false statement about
        the feed, and it hides the one fact that explains the skip."""
        junk = _unsupported_language_skip_reason(_cfg(language="es", feed="?")) or ""
        assert "declared '?'" in junk
        assert "not a usable language tag" in junk

        silent = _unsupported_language_skip_reason(_cfg(language="es")) or ""
        assert "declared no <language>" in silent


class TestTheRefusalIsCountable:
    """Structurally guarded by test_refusals_are_countable; asserted here by behaviour."""

    def test_the_skip_writes_a_ledger_row_and_an_incident(self, monkeypatch) -> None:
        from podcast_scraper.workflow import episode_processor as ep

        recorded: list[dict[str, Any]] = []
        incidents: list[dict[str, Any]] = []

        monkeypatch.setattr(
            ep,
            "_record_unresolved_transcript",
            lambda job, cfg, pm, stage, *, error_type="", detail=None: recorded.append(
                {"stage": stage, "error_type": error_type, "detail": detail}
            ),
        )
        monkeypatch.setattr(
            ep,
            "_append_transcription_incident",
            lambda cfg, job, *, category="", message="", exception_type="": incidents.append(
                {"category": category, "message": message, "exception_type": exception_type}
            ),
        )
        monkeypatch.setattr(ep, "_bind_episode_correlation", lambda job, cfg: None)

        job = MagicMock()
        job.idx = 1
        ok, path, downloaded = ep.transcribe_media_to_text(
            job, _cfg(language="es"), None, None, "/tmp/out", None, None
        )

        assert (ok, path, downloaded) == (False, None, 0), "a refusal, not a silent success"
        assert len(recorded) == 1
        assert recorded[0]["error_type"] == "UnsupportedLanguage"
        assert recorded[0]["stage"] == "transcription"
        assert "'es'" in (recorded[0]["detail"] or "")
        assert len(incidents) == 1
        assert incidents[0]["exception_type"] == "UnsupportedLanguage"
        assert incidents[0]["category"] == "policy"

    def test_the_gate_runs_before_any_provider_is_touched(self, monkeypatch) -> None:
        """Refused before a provider is called, so a disabled language costs nothing.

        ``transcription_provider=None`` would raise on the ordinary path; returning the refusal
        cleanly is what proves the gate came first.
        """
        from podcast_scraper.workflow import episode_processor as ep

        monkeypatch.setattr(ep, "_record_unresolved_transcript", lambda *a, **k: None)
        monkeypatch.setattr(ep, "_append_transcription_incident", lambda *a, **k: None)
        monkeypatch.setattr(ep, "_bind_episode_correlation", lambda job, cfg: None)

        job = MagicMock()
        job.idx = 1
        result = ep.transcribe_media_to_text(
            job, _cfg(language="ja"), None, None, "/tmp/out", None, None
        )
        assert result == (False, None, 0)

    def test_it_is_refused_even_under_dry_run(self, monkeypatch) -> None:
        """A dry run should report the skip it WOULD make, not a transcription it would never
        attempt — so the gate sits ahead of the dry-run guard."""
        from podcast_scraper.workflow import episode_processor as ep

        monkeypatch.setattr(ep, "_record_unresolved_transcript", lambda *a, **k: None)
        monkeypatch.setattr(ep, "_append_transcription_incident", lambda *a, **k: None)
        monkeypatch.setattr(ep, "_bind_episode_correlation", lambda job, cfg: None)

        cfg = _cfg(language="es").model_copy(update={"dry_run": True})
        job = MagicMock()
        job.idx = 1
        assert ep.transcribe_media_to_text(job, cfg, None, None, "/tmp/out", None, None) == (
            False,
            None,
            0,
        )
