"""The fixtures carry language, and the generators agree with the pipeline (#2185 / slice S0.9).

Two failures this closes, both found by measurement rather than review:

* ``viewer-validation-corpus/v3`` had NO ``feed.language`` on any of its 40 episodes, so the S0.4
  audit reported the whole corpus as "not enabled in the registry". The RSS parser in
  ``build_synthetic_validation_corpus.py`` had been extracting the tag all along — it was simply
  never written into the metadata.
* ``app-validation-corpus/v3`` stored the RAW tag ``"en-us"``, so the fixture corpus disagreed
  with the pipeline about what the language IS.

And one gap: no RSS fixture declared a non-English language, which made S0.1a's own stated
acceptance (``es-ES`` -> ``es``) unmeetable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from podcast_scraper.languages import is_language_enabled, normalize_language_tag
from podcast_scraper.rss.parser import _channel_language

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
RSS_DIR = REPO / "tests" / "fixtures" / "rss"


class TestTheRssFixtures:
    def test_every_fixture_declares_a_language(self) -> None:
        """A feed with no tag is a real state the pipeline handles, but a FIXTURE with no tag is
        just an untested surface — the tag is what these files exist to exercise."""
        missing = [
            p.name
            for p in sorted(RSS_DIR.glob("*.xml"))
            if not re.search(rb"<language>", p.read_bytes(), re.I)
        ]
        assert not missing, f"RSS fixtures with no <language>: {missing}"

    def test_a_non_english_fixture_exists(self) -> None:
        """S0.1a (#2172) states its acceptance as a feed declaring es-ES. Nothing could exercise
        that until this fixture existed."""
        non_english = {
            p.name: raw
            for p in sorted(RSS_DIR.glob("*.xml"))
            if (raw := _channel_language(p.read_bytes())) and normalize_language_tag(raw) != "en"
        }
        assert non_english, "no RSS fixture declares a non-English language"
        assert "p10_spanish.xml" in non_english

    def test_the_spanish_fixture_meets_s0_1a_s_stated_acceptance(self) -> None:
        raw = _channel_language((RSS_DIR / "p10_spanish.xml").read_bytes())
        assert raw == "es-ES", "the raw tag must survive verbatim for the operator"
        assert normalize_language_tag(raw) == "es"

    def test_the_regional_subtag_makes_normalization_load_bearing(self) -> None:
        """A bare ``es`` would pass a normalizer that only lowercases — which is exactly the bug
        S0.2 fixed in ``Config._normalize_language``."""
        raw = _channel_language((RSS_DIR / "p10_spanish.xml").read_bytes())
        assert raw is not None
        assert raw.lower() != normalize_language_tag(raw), (
            "the fixture's tag must differ from its own lowercase form, or it cannot detect a "
            "lowercase-only normalizer"
        )

    def test_spanish_is_ENABLED_now_so_this_fixture_is_ingested(self) -> None:
        """Inverted 2026-09-30. This test used to assert `es` was disabled — "the language is
        parsed and recorded; it is NOT ingested" — which was the right assertion while Spanish
        was only a metadata fixture.

        `es` is enabled now, so this feed is a feed the pipeline processes, and the thing worth
        pinning is the opposite: that S0.8 no longer refuses it. The refusal path is still
        covered, by a language that IS still disabled.
        """
        assert is_language_enabled("es") is True
        assert is_language_enabled("de") is False, (
            "the S0.8 refusal path needs at least one described-but-disabled language to be "
            "testable at all"
        )


class TestTheGeneratorsWriteIt:
    """Asserted against the generator SOURCE, because regenerating a corpus is a separate,
    expensive step — the committed corpora are checked in the next class."""

    def test_the_viewer_generator_writes_the_feed_language_block(self) -> None:
        src = (REPO / "scripts" / "build_synthetic_validation_corpus.py").read_text(
            encoding="utf-8"
        )
        for field in ('"language":', '"language_raw":', '"language_source":'):
            assert field in src, f"{field} missing from the viewer generator"
        assert "normalize_language_tag" in src, "the generator must normalize as the pipeline does"

    def test_the_app_generator_normalizes_rather_than_storing_the_raw_tag(self) -> None:
        src = (REPO / "scripts" / "build_app_validation_corpus.py").read_text(encoding="utf-8")
        assert 'normalize_language_tag(feed_meta.get("language"))' in src
        assert '"language_raw": feed_meta.get("language") or None' in src

    def test_both_generators_record_the_source(self) -> None:
        """Without ``language_source`` the S0.4 audit cannot tell a measured corpus from a
        defaulted one — which is the failure that audit exists to prevent."""
        for name in ("build_synthetic_validation_corpus.py", "build_app_validation_corpus.py"):
            src = (REPO / "scripts" / name).read_text(encoding="utf-8")
            assert '"language_source": "rss" if feed_meta.get("language") else None' in src, name


class TestTheCommittedCorporaStillNeedRegenerating:
    """Honest record of what has NOT happened yet.

    The generator changes do not retroactively alter committed artifacts, so the corpora still
    carry their old shape. These tests document the current state rather than asserting the
    desired one — and they will FAIL when the corpora are regenerated, which is the signal to
    delete them and tighten the assertions above.
    """

    def test_the_viewer_corpus_still_has_no_language(self) -> None:
        import json

        root = REPO / "tests/fixtures/viewer-validation-corpus/v3"
        metas = sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        assert metas
        have = [
            m.name
            for m in metas
            if "language" in (json.loads(m.read_text(encoding="utf-8")).get("feed") or {})
        ]
        assert not have, (
            "the viewer corpus now carries a feed language — regenerated? Then delete this test "
            "and assert the language is present instead."
        )

    def test_the_app_corpus_still_carries_the_raw_tag(self) -> None:
        import json

        root = REPO / "tests/fixtures/app-validation-corpus/v3"
        metas = sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        assert metas
        stored = {
            (json.loads(m.read_text(encoding="utf-8")).get("feed") or {}).get("language")
            for m in metas
        }
        assert stored == {"en-us"}, (
            f"the app corpus's stored language changed to {stored} — regenerated? Then delete "
            "this test; S0.5's contract test asserts the served value either way."
        )
