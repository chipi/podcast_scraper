"""No host-name source may seed a composite person (#1652).

A first fix for this bug patched the LLM-provider path only, shipped green, and changed
nothing: the acceptance re-run produced the identical
``person:erik-torenberg-ben-horowitz-travis-kalanick``. The fix was correct; it was on the
wrong path. *The a16z Show* publishes its cast in a single episode-level
``<itunes:author>Erik Torenberg, Ben Horowitz, Travis Kalanick</itunes:author>``, and the
episode-authors fallback — a different function, with its own org-only filter — let it
through whole. Fifteen tests written against the provider hypothesis all passed.

So these tests are organised by SOURCE, not by function. Four independent paths can seed
``known_hosts``:

    1. the deterministic feed parse          (hosts.detect_hosts_from_feed)
    2. the LLM provider                      (processing._sanitize_detected_hosts)
    3. episode-level <itunes:author> tags    (processing._fallback_to_episode_authors)
    4. config known_hosts                    (detect_feed_hosts_and_patterns, two sites)

Every one is asserted against the same real composite, and ``TestNoPathCanBypassTheGate``
fails if a fifth path is added that does not go through the shared normaliser. A
per-function test suite is what let the first fix look finished.

Why a composite is worse than no host: the roster compares per name, so it can never match a
diarized voice — it silently disables the known-hosts anchor *and* mints a Person node for a
human who does not exist, which cross-episode queries then happily join on.
"""

# mypy: disable-error-code="arg-type,list-item"
# Deliberate in this file: lightweight duck-typed doubles passed where the production type is
# declared; same, inside a list argument.
# Constructing the real types would pull in the machinery these tests isolate. The
# annotations on the helpers here are what make mypy check these bodies at all — most
# older test files are unannotated and therefore unchecked.

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: N817  # Element TYPE only; parsing uses defusedxml
from pathlib import Path
from typing import Any, cast, List, Set

import pytest

# defusedxml for the PARSE, matching rss/parser.py and stages/test_scraping.py. Bandit B314
# blacklists stdlib ElementTree parsing regardless of whether the input is trusted, and this
# repo answers that with the safe parser rather than a per-line suppression. ``ET`` stays for
# the ``Element`` type annotation, which B314 does not flag.
from defusedxml.ElementTree import fromstring as safe_fromstring

from podcast_scraper.speaker_detectors.hosts import (
    detect_hosts_from_feed,
    normalize_host_names,
)
from podcast_scraper.workflow.stages import processing

pytestmark = [pytest.mark.unit]

# The exact string observed in the acceptance run's a16z episode.
COMPOSITE = "Erik Torenberg, Ben Horowitz, Travis Kalanick"
EXPECTED = {"Erik Torenberg", "Ben Horowitz", "Travis Kalanick"}


def _item(author: str) -> ET.Element:
    # defusedxml has no stubs, so this is Any to mypy; the runtime type is a real Element.
    return cast(
        ET.Element,
        safe_fromstring(
            '<item xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">'
            f"<title>Ep</title><itunes:author>{author}</itunes:author></item>"
        ),
    )


class _Episode:
    def __init__(self, author: str) -> None:
        self.item = _item(author)
        self.title = "Ep"


class _Cfg:
    def __init__(self, known_hosts: Any = None) -> None:
        self.auto_speakers = True
        self.known_hosts = known_hosts or []


class TestEverySeedingPathSplitsTheComposite:
    """One assertion per source. Each is run against the real a16z string."""

    def test_path_1_deterministic_feed_parse(self) -> None:
        hosts = detect_hosts_from_feed("The a16z Show", "", [COMPOSITE])
        assert hosts == EXPECTED

    def test_path_2_llm_provider(self) -> None:
        assert processing._sanitize_detected_hosts({COMPOSITE}) == EXPECTED

    def test_path_3_episode_level_author_tags(self) -> None:
        """THE path that actually fired on the acceptance run, and the one the first fix
        missed. It previously applied ``is_network_or_org_author`` and nothing else — and that
        predicate returns False for a three-person string, since it is neither a mononym nor
        org-marked, so the composite passed the only check there was."""
        out = processing._fallback_to_episode_authors(_Cfg(), [_Episode(COMPOSITE)])
        assert out == EXPECTED

    def test_path_4_config_known_hosts(self) -> None:
        """Operator-supplied is not exempt: a composite in a show config is exactly as
        unmatchable against a diarized voice, and mints the same fake Person."""
        assert normalize_host_names([COMPOSITE]) == EXPECTED


class TestTheCompositeNeverSurvives:
    """The negative form of the above — stated separately because "produces three names" and
    "never produces the fused one" fail differently when a split half-works."""

    @pytest.mark.parametrize(
        "produce",
        [
            pytest.param(lambda: detect_hosts_from_feed("S", "", [COMPOSITE]), id="feed_parse"),
            pytest.param(lambda: processing._sanitize_detected_hosts({COMPOSITE}), id="provider"),
            pytest.param(
                lambda: processing._fallback_to_episode_authors(_Cfg(), [_Episode(COMPOSITE)]),
                id="episode_authors",
            ),
            pytest.param(lambda: normalize_host_names([COMPOSITE]), id="config_known_hosts"),
        ],
    )
    def test_no_path_emits_the_fused_string(self, produce: Any) -> None:
        assert COMPOSITE not in produce()

    def test_the_slug_that_reached_the_corpus_is_unreachable(self) -> None:
        """``person:erik-torenberg-ben-horowitz-travis-kalanick`` is what a composite becomes
        downstream. No emitted name may contain more than one person's worth of tokens."""
        for name in processing._fallback_to_episode_authors(_Cfg(), [_Episode(COMPOSITE)]):
            assert len(name.split()) <= 3, f"{name!r} looks like more than one person"


class TestNoPathCanBypassTheGate:
    """Structural. The per-source tests above only cover the paths I know about; this fails
    when a NEW seeding path is added without routing through the shared normaliser — the
    exact way the first fix ended up incomplete."""

    def _source(self, mod: Any) -> str:
        return Path(mod.__file__).read_text(encoding="utf-8")

    def test_the_shared_gate_exists_and_is_importable(self) -> None:
        assert callable(normalize_host_names)

    def test_processing_routes_every_path_through_it(self) -> None:
        src = self._source(processing)
        assert src.count("normalize_host_names(") >= 4, (
            "a host-name source in processing.py does not go through normalize_host_names — "
            "check _sanitize_detected_hosts, _fallback_to_episode_authors, and BOTH "
            "cfg.known_hosts sites"
        )

    def test_no_path_re_wraps_known_hosts_raw(self) -> None:
        """``set(cfg.known_hosts)`` is the un-normalised shortcut both config sites used."""
        src = self._source(processing)
        assert "set(cfg.known_hosts)" not in src

    def test_episode_authors_no_longer_hand_rolls_its_filter(self) -> None:
        """It used to add authors one at a time behind a bare org check. If that shape comes
        back, the split is bypassed again."""
        src = self._source(processing)
        assert "episode_authors.add(author)" not in src


class TestTheConservativeContractHolds:
    """An over-eager split INVENTS a person, which is worse than the bug being fixed. These
    pin the safe direction (#876): degrade to "no host", never to a fake one."""

    def test_a_single_name_is_untouched(self) -> None:
        assert normalize_host_names(["Patrick O'Shaughnessy"]) == {"Patrick O'Shaughnessy"}

    def test_a_name_suffix_is_not_a_second_person(self) -> None:
        assert normalize_host_names(["Martin Luther King, Jr."]) == {"Martin Luther King, Jr."}

    def test_organisations_are_dropped_not_split_into_people(self) -> None:
        assert normalize_host_names(["Colossus | Investing & Business Podcasts"]) == set()

    def test_a_mixed_cast_keeps_people_and_drops_the_publisher(self) -> None:
        """Latent Space ships "Brandon Anderson, RJ Honicky, and Latent.Space" — two hosts and
        the show itself in one tag."""
        assert normalize_host_names(["Brandon Anderson, RJ Honicky, and Latent.Space"]) == {
            "Brandon Anderson",
            "RJ Honicky",
        }

    def test_an_email_form_is_reduced_to_the_name(self) -> None:
        """The feed-author path stripped ``<addr>``; the other three did not, so one person
        arrived under two spellings depending on which path won."""
        assert normalize_host_names(["Jane Roe <jane@example.com>"]) == {"Jane Roe"}

    @pytest.mark.parametrize("bad", [None, "", "   ", "  ,  ,  "])
    def test_empty_and_junk_yield_nothing_rather_than_raising(self, bad: Any) -> None:
        """This runs inside feed processing; raising here would fail the whole show."""
        assert normalize_host_names([bad]) == set()

    def test_the_word_and_inside_a_name_is_not_a_separator(self) -> None:
        assert normalize_host_names(["Alexander Anderson"]) == {"Alexander Anderson"}


class TestTheLogSaysWhichPathActuallyFired:
    """Not cosmetic. The a16z log said "DETECTED HOSTS (from RSS author tags)" while the names
    had really come from episode-level authors — because the old branch tested only whether
    ``feed.authors`` was non-empty, which is true even when those tags were stripped as
    publisher metadata. That line is what sent the first investigation to the wrong path."""

    def _log_source(
        self, cached: Set[str], feed_authors: List[str], episode_authors: Set[str], cfg: _Cfg
    ) -> str:
        class _Feed:
            authors = feed_authors

        records: List[str] = []

        class _Logger:
            @staticmethod
            def info(msg: str, *args: Any) -> None:
                records.append(msg % args if args else msg)

            @staticmethod
            def debug(*_a: Any, **_k: Any) -> None:
                return None

        original = processing.logger
        processing.logger = _Logger()  # type: ignore[assignment]
        try:
            processing._log_detected_hosts(cached, _Feed(), episode_authors, cfg)
        finally:
            processing.logger = original
        return records[-1] if records else ""

    def test_episode_authors_are_not_labelled_rss_author_tags(self) -> None:
        """The regression case: an org-authored feed (author tags present but stripped) whose
        hosts came from episode-level authors."""
        line = self._log_source(EXPECTED, ["a16z"], EXPECTED, _Cfg())
        assert "episode-level authors" in line
        assert "RSS author tags" not in line

    def test_real_author_tag_hosts_are_still_labelled_correctly(self) -> None:
        line = self._log_source({"Jane Roe"}, ["Jane Roe"], set(), _Cfg())
        assert "RSS author tags" in line

    def test_config_known_hosts_are_labelled_even_when_they_needed_normalising(self) -> None:
        """``cached_hosts`` holds the normalised names, so comparing against the RAW config
        value would never match and would mislabel this as an author-tag hit."""
        line = self._log_source(EXPECTED, [], set(), _Cfg(known_hosts=[COMPOSITE]))
        assert "config known_hosts" in line

    def test_a_missing_cfg_does_not_raise(self) -> None:
        """Callers pass ``cfg=None`` in places; a logging helper must never be the thing that
        fails a feed."""
        assert "RSS author tags" in self._log_source({"Jane Roe"}, ["Jane Roe"], set(), None)

    def test_an_explicit_source_beats_inference(self) -> None:
        """The caller knows which branch fired; inference only guesses. Where they disagree,
        the caller wins — otherwise threading the source through changes nothing."""

        class _Feed:
            authors = ["a16z"]

        records: List[str] = []

        class _Logger:
            @staticmethod
            def info(msg: str, *args: Any) -> None:
                records.append(msg % args if args else msg)

            @staticmethod
            def debug(*_a: Any, **_k: Any) -> None:
                return None

        original = processing.logger
        processing.logger = _Logger()  # type: ignore[assignment]
        try:
            processing._log_detected_hosts(
                EXPECTED, _Feed(), set(), _Cfg(), source="episode-level authors"
            )
        finally:
            processing.logger = original
        assert "episode-level authors" in records[-1]

    def test_the_real_caller_threads_the_source_instead_of_inferring(self) -> None:
        """Structural: if detect_feed_hosts_and_patterns stops passing ``source=``, every case
        silently reverts to the guess that caused the misdiagnosis."""
        src = Path(processing.__file__).read_text(encoding="utf-8")
        assert "source=host_source" in src
        assert 'host_source = "episode-level authors"' in src
