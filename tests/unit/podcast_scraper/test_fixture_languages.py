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

    BOTH CORPORA NOW CARRY LANGUAGE (2026-10-01), so this class no longer records a gap in the
    data — it records the one that is left, in the GENERATOR. See the viewer test below.
    """

    def test_the_viewer_corpus_carries_the_declared_language(self) -> None:
        """Tightened as the class docstring asked, rather than deleted.

        The viewer corpus had NO `feed.language` on any of its 40 episodes, so the S0.4 audit
        reported the whole corpus as "not enabled in the registry" — while the app corpus, which
        was regenerated, looked fine. #2185.

        WRITTEN SURGICALLY, NOT BY REGENERATING, and that is the finding worth keeping: re-running
        `build_synthetic_validation_corpus.py` is not a safe way to apply a field to this corpus.
        Its `base_date` is `datetime.utcnow()`, so every run shifts all 40 publish dates and the
        graph nodes derived from them — 131 of 332 files differ on a no-op rerun, almost none of it
        about language — and it would DELETE 18 git-tracked `.app/users/*` files (profiles,
        preferences, graph events) that the viewer e2e reads and the generator does not produce.
        Making the generator deterministic is the other half of #2185.
        """
        import json

        root = REPO / "tests/fixtures/viewer-validation-corpus/v3"
        metas = sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        assert metas
        missing = [
            m.name
            for m in metas
            if not (json.loads(m.read_text(encoding="utf-8")).get("feed") or {}).get("language")
        ]
        assert not missing, (
            f"{len(missing)} viewer-corpus episode(s) still carry no feed language, so the S0.4 "
            f"audit reports them as not enabled: {missing[:5]}"
        )

    def test_the_viewer_corpus_records_the_RAW_tag_and_its_source(self) -> None:
        """`en-us` normalized to `en` with `language_source: rss` — the same shape the app corpus
        carries, so one audit reads both corpora identically."""
        import json

        root = REPO / "tests/fixtures/viewer-validation-corpus/v3"
        feeds = [
            json.loads(m.read_text(encoding="utf-8")).get("feed") or {}
            for m in sorted(root.glob("feeds/*/**/metadata/*.metadata.json"))
        ]
        assert feeds
        assert {f.get("language") for f in feeds} == {"en"}
        assert {f.get("language_raw") for f in feeds} == {"en-us"}
        assert {f.get("language_source") for f in feeds} == {"rss"}

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


class TestANonEnglishCorpusEpisodeCarriesItsEnglishRender:
    """RFC-124's output must survive into the corpus, not just into a run's output directory.

    THE GAP THIS CLOSES. The committed app corpus had five non-English episodes and ZERO English
    renders, and every gate stayed green. `test_translation_artifacts.py` proves the stage writes
    `.en.txt` + `.en.segments.json` — but it runs the stage against a temp dir, so it says nothing
    about what the corpus ends up holding. The corpus side was never asked the question. The
    translation really did run (the committed summaries, GI and KG for p10-p14 are English,
    derived from it); the assembling script simply did not carry the English transcript across.

    Why it matters beyond tidiness: `.en.txt` IS the gate. RFC-124 makes the absence of that file
    mean "no complete translation exists", so a corpus missing it does not look half-built to the
    rest of the system — it looks like five untranslated episodes. The indexer then does the
    correct thing for that input and emits a single source-language layer, so RFC-124 6.2's
    two-layer path (Goal 6, "findable in the language it was spoken in") gets no corpus coverage
    at all while appearing to be exercised by the only corpus we base tests on.
    """

    @staticmethod
    def _tagged_source_for(meta_path: Path, doc: dict, language: str) -> Path:
        """Where the SOURCE body lives once translation has swapped it (D-44).

        INVERTED FROM WHAT THIS USED TO BE. It derived `<base>.en.txt` — the translation beside a
        canonical source. English is now the canonical file, so the artifact that proves a
        translation happened is the source at `<base>.<lang>.txt`, which only the atomic swap
        creates. Derived from `content.transcript_file_path` rather than guessing at the episode id,
        because the suffix is only well-defined from a canonical path.
        """
        rel = (doc.get("content") or {}).get("transcript_file_path")
        assert rel, f"{meta_path.name} has no content.transcript_file_path"
        source = meta_path.parent.parent / str(rel)
        assert source.name.endswith(".txt"), f"unexpected transcript suffix: {source.name}"
        lang = language.strip().lower().split("-")[0]
        return source.with_name(source.name[: -len(".txt")] + f".{lang}.txt")

    def _non_english_episodes(self) -> list[tuple[Path, dict]]:
        import json

        root = REPO / "tests/fixtures/app-validation-corpus/v3"
        out = []
        for m in sorted(root.glob("feeds/*/**/metadata/*.metadata.json")):
            doc = json.loads(m.read_text(encoding="utf-8"))
            lang = (doc.get("episode") or {}).get("language") or (doc.get("feed") or {}).get(
                "language"
            )
            if lang and str(lang) != "en":
                out.append((m, doc))
        return out

    def test_the_corpus_has_non_english_episodes_to_check(self) -> None:
        """Guard the guard: if the corpus loses its non-English shows, the assertion below would
        pass vacuously and the regression would be invisible again."""
        eps = self._non_english_episodes()
        assert len(eps) >= 5, f"expected the five non-English shows, found {len(eps)}"

    def test_every_non_english_episode_HAS_been_swapped(self) -> None:
        """The tagged source beside the canonical body — the artifact only the swap creates.

        Its absence means the canonical `<base>.txt` still holds the SOURCE language, which under
        D-44 makes the episode unusable: every generic reader opens that path believing it holds the
        analysis language.
        """
        missing = []
        for meta_path, doc in self._non_english_episodes():
            language = (doc.get("episode") or {}).get("language") or (doc.get("feed") or {}).get(
                "language"
            )
            tagged = self._tagged_source_for(meta_path, doc, str(language))
            if not tagged.exists():
                missing.append(str(tagged.relative_to(REPO)))
        assert not missing, (
            "non-English corpus episodes whose swap never happened, so their canonical body is "
            "still the source language and the two-layer index path has no corpus coverage:\n  "
            + "\n  ".join(missing)
        )

    def test_the_english_render_is_actually_english(self) -> None:
        """A present-but-source-language file would pass the existence check and still leave the
        corpus wrong, which is the failure mode that made the original gap so quiet."""
        bad = []
        for meta_path, doc in self._non_english_episodes():
            # The CANONICAL body is the English one after the swap — which is the whole point: a
            # generic reader opens this path without naming a language.
            rel = (doc.get("content") or {}).get("transcript_file_path")
            english = meta_path.parent.parent / str(rel)
            if not english.exists():
                continue  # the test above owns that failure
            head = english.read_text(encoding="utf-8")[:400].lower()
            if not re.search(r"\b(the|and|is|that|with|for)\b", head):
                bad.append(str(english.relative_to(REPO)))
        assert not bad, f"an .en.txt that does not read as English: {bad}"
