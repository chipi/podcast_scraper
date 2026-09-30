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

from podcast_scraper.languages import (
    is_language_enabled,
    language_registry,
    normalize_language_tag,
)
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
        # DERIVED, not named. This asserted `de` was disabled — true when written, false the
        # moment German was enabled alongside it/fr/pt, and the failure then read as "the
        # refusal path is untestable" when the property it guards was perfectly intact. What
        # the test actually needs is that SOME described language is still disabled; which one
        # is not its business.
        described = set(language_registry())
        still_disabled = sorted(c for c in described if not is_language_enabled(c))
        assert still_disabled, (
            "the S0.8 refusal path needs at least one described-but-disabled language to be "
            "testable at all — every described language is now enabled"
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

    def test_the_app_corpus_IS_regenerated_and_normalized(self) -> None:
        """REGENERATED 2026-09-30, so this is the tightened assertion the class docstring asked
        for — it used to assert the corpus still stored the raw `en-us`.

        Two things changed in the rebuild, and only one of them was the point:

        * `p10` arrived, so the corpus carries `es` as well as English — the first non-English
          episode in it.
        * every ENGLISH episode went `en-us` -> `en`. That was pending, not new: the generator
          gained `normalize_language_tag` during Phase 0 and committed artifacts are not
          retroactively altered, so the corpus had been carrying a pre-normalisation shape.
          Rebuilding it for p10 is what finally applied it.

        The normalised form is the one the whole system reasons about — `whisper_utils` checks
        `language.lower() in ("en", "english")`, so `en-us` reads as NOT English (D-21), which is
        the exact bug normalisation exists to prevent.
        """
        import json

        root = REPO / "tests/fixtures/app-validation-corpus/v3"
        metas = sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        assert metas
        stored = {
            (json.loads(m.read_text(encoding="utf-8")).get("feed") or {}).get("language")
            for m in metas
        }
        # Five non-English shows since 2026-10-01 (es, it, fr, de, pt), each a counterpart of
        # p01 carrying the same conversation in its own language. Derived from the registry
        # rather than listed, so enabling a sixth is a fixture change and not a test edit.
        assert "en" in stored, f"the app corpus lost its English shows: {stored}"
        assert stored >= {
            "en",
            "es",
            "it",
            "fr",
            "de",
            "pt",
        }, f"the app corpus's stored languages are {stored}"
        # `stored` comes from `.get("language")`, so its members are Optional as far as the type
        # checker is concerned; narrow before comparing or sorting.
        codes = sorted(str(c) for c in stored if c)
        assert all(is_language_enabled(c) for c in codes), (
            "the corpus stores a language that is not enabled: "
            f"{[c for c in codes if not is_language_enabled(c)]}"
        )
        assert not any(
            s and "-" in s for s in stored
        ), f"a raw regional subtag survived normalisation: {stored}"

    def test_the_app_corpus_keeps_the_RAW_tag_beside_the_normalized_one(self) -> None:
        """`language_raw` is what makes the corpus auditable: it distinguishes a feed that
        declared `es-ES` from one that declared `es`, and normalisation would otherwise destroy
        that. §3.1's whole point is being able to tell a measured corpus from a defaulted one."""
        import json

        root = REPO / "tests/fixtures/app-validation-corpus/v3"
        raws = {
            (json.loads(m.read_text(encoding="utf-8")).get("feed") or {}).get("language_raw")
            for m in sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        }
        assert "es-ES" in raws, f"p10's raw tag is missing: {sorted(r for r in raws if r)}"
        assert "en-us" in raws, f"the English raw tags are missing: {sorted(r for r in raws if r)}"
