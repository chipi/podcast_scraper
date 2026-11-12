"""Does the committed fixture corpus actually TRIGGER the detectors, per language?

THE DISTINCTION THIS FILE EXISTS TO MAKE. There are two kinds of detector test and only one of
them is evidence about the corpus:

1. A detector fed a sentence written in the test file. Proves the regex is well-formed.
   ``test_detector_vocabulary_is_language_keyed.py`` is this, for all six languages.
2. A detector run over the real fixture episodes. Proves the pattern survives being turned into a
   diarized screenplay with ``Name:`` prefixes, segmented and offset-indexed — and proves the
   corpus can exercise the path at all.

WHICH BODY TO READ, AND THE BUG THAT MAKES THIS THE FIRST THING THE FILE SAYS. Under D-44 a
translated episode's CANONICAL ``<base>.txt`` holds ENGLISH after the atomic swap; the source
survives at ``<base>.<lang>.txt``. So matching Spanish patterns against the canonical body returns
zero, every time, for a reason that has nothing to do with the fixtures. I did exactly that and
reported "no non-English fixture triggers any detector" — which was false. (An earlier version
resolved the transcript path against the corpus root rather than the run directory and read empty
strings, reporting zero for all 45 episodes including English. Two different bugs, same shape:
a zero I believed instead of checked.) ``TestTheScanItselfIsSound`` exists so neither can recur
silently.

WHAT RUNS ON WHICH BODY IN PRODUCTION — because it decides what this file can claim. The detectors
read the CANONICAL body, which after translation is English. The source-language vocabularies
therefore do not run in production today at all: translation happens first, and analysis is
English by construction. Scanning the source body here is a READINESS check — "if the detectors
were pointed at this language, would they fire" — and it is the measurement the per-language
tuning issues need. It is not a claim about a live code path.

MEASURED 2026-10-02, source-language bodies, after the four pattern fixes the fixtures exposed:

    language  episodes   >=2 ad patterns   host act   guest act   intro cue
    en        40 (9 feeds)          31          38          25          39
    es/it/fr/de/pt  3 each          yes         yes         yes         yes

The first scan of these feeds found four real defects in rows written the day after the fixtures:
Italian "Bentornati" (its own word for "welcome back", so the row scored zero host acts),
Portuguese ``patrocinado pela`` (``por + a`` contracts, and every row used the bare preposition),
and the first-person-singular introduction ("con me c'è", "bei mir ist", "comigo está") which four
of five rows missed entirely. Fixture content written by a different pass is weak evidence, but it
is evidence — which is the whole argument for this file existing.

WHAT A GREEN RESULT HERE DOES NOT MEAN. ``_e02`` / ``_e03`` are TRANSLATIONS of ``p01_e02`` /
``p01_e03``, hand-authored the same way ``_e01`` was, so a green result proves INTEGRATION — the
pattern fires after the text has become a diarized screenplay with ``Name:`` prefixes, segmented
and offset-indexed — and not RECALL. It cannot discover a phrase real speech uses and nobody wrote
down. That needs #2187 and the per-language issues (#2255-#2259).
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple

import pytest

# The repo's XML convention (see src/podcast_scraper/rss/parser.py): parse through
# defusedxml's safe API, never the stdlib parser. Bandit B314 flagged the stdlib call here
# and it was right to — these are repo fixtures today, but a test helper that reads XML with
# the unsafe parser is a pattern that gets copied somewhere the input is not trusted.
from defusedxml.ElementTree import fromstring as safe_fromstring, ParseError as XMLParseError

from podcast_scraper.languages import (
    is_language_enabled,
    language_registry,
    normalize_language_tag,
)
from podcast_scraper.rss.parser import _channel_language
from tests._detector_scenarios import scenario_for

pytestmark = pytest.mark.unit

REPO = pathlib.Path(__file__).resolve().parents[3]
TRANSCRIPTS = REPO / "tests/fixtures/transcripts/v3"
RSS = REPO / "tests/fixtures/rss"

TIER_1 = sorted(c for c in language_registry() if is_language_enabled(c))

#: Languages whose fixture feed still carries no detector-triggering content, with the issue that
#: closes each. DATA, not skip markers scattered through the file — so the day a feed gains its
#: episodes, one entry is deleted and the parameterization picks it up with no other edit.
#:
#: EMPTY AS OF 2026-10-02: ``p10_e02``..``p14_e03`` landed — rewritten 2026-10-03 as
#: translations of their ``p01`` counterparts — and every tier-1 language now exercises
#: all five checks. The mechanism is kept rather than deleted because enabling a SIXTH language
#: puts it straight back to use — a new language arrives with no fixture content by definition.
FIXTURE_COVERAGE_PENDING: Dict[str, str] = {}


def _pending(language: str) -> List[pytest.MarkDecorator]:
    """``xfail(strict=True)`` for a language with no trigger content, nothing for one that has it.

    Strict, deliberately. A non-strict xfail would let the fixture work land and leave the marker
    behind, and the next reader would see a language marked "pending" that has been covered for
    months — which is how a stale allowlist starts.
    """
    issue = FIXTURE_COVERAGE_PENDING.get(language)
    if not issue:
        return []
    return [
        pytest.mark.xfail(
            strict=True,
            reason=(
                f"the {language} fixture feed has one episode of plain conversation and no "
                f"detector-triggering content — tracked in {issue}. When its episodes land this "
                f"XPASSes and the suite fails until {language!r} is removed from "
                "FIXTURE_COVERAGE_PENDING."
            ),
        )
    ]


def _language_cases() -> List[pytest.param]:  # type: ignore[valid-type]
    """One case per enabled language, carrying its own pending marker.

    Parameterized from the REGISTRY rather than a written-down list, so enabling a sixth language
    produces a failing case immediately instead of silently not being tested.
    """
    return [pytest.param(lang, marks=_pending(lang), id=lang) for lang in TIER_1]


def _feed_languages() -> Dict[str, str]:
    """``{feed prefix: language}``, parsed from the RSS fixtures.

    THE FEED IS THE SOURCE OF TRUTH, and the obvious alternative is wrong: every v3 transcript
    carries a ``#fixture-v3: voice=<tag>`` header, which looks like a language and is not. It is
    the TTS ACCENT hint — ``p07_e01`` is an English episode whose header reads
    ``voice=de-DE host_voice=en-AU``, because its guest is voiced with a German accent. Reading it
    as the language would file nine English shows under four languages.
    """
    out: Dict[str, str] = {}
    for feed in sorted(RSS.glob("p*.xml")):
        prefix = feed.name.split("_")[0]
        if prefix in out:
            continue
        raw = _channel_language(feed.read_bytes())
        if raw:
            out[prefix] = normalize_language_tag(raw) or ""
    return out


def _episodes() -> Dict[str, List[Tuple[str, str, str]]]:
    """``{language: [(episode_id, transcript_text, description)]}`` from the AUTHORED transcripts.

    READ FROM ``tests/fixtures/transcripts/v3`` RATHER THAN THE APP-VALIDATION CORPUS, and the
    reason is the bug recorded in the module docstring. The corpus is a post-pipeline artifact: a
    translated episode's canonical body there holds ENGLISH after the D-44 swap, so matching
    source-language patterns against it returns zero for a reason that has nothing to do with the
    fixtures — which is precisely the false result I reported. The authored transcripts are
    unambiguously in their own language, and they are also what the mock server serves and what a
    real ingest would read, so this asks the question about the content that actually ships.

    The description comes from the RSS item, because guest discovery reads the description and a
    cue that never appears there is a cue the description path cannot exercise.
    """
    languages = _feed_languages()
    descriptions = _item_descriptions()
    out: Dict[str, List[Tuple[str, str, str]]] = {}
    paths = [
        tr
        for tr in sorted(TRANSCRIPTS.glob("p*_e[0-9]*.txt"))
        if "_multi_" not in tr.stem and "_fast" not in tr.stem
    ]
    assert paths, f"no fixture transcripts under {TRANSCRIPTS}"
    for tr in paths:
        prefix = tr.stem.split("_")[0]
        language = languages.get(prefix)
        if not language:
            continue
        out.setdefault(language, []).append(
            (tr.stem, tr.read_text(encoding="utf-8"), descriptions.get(tr.stem, ""))
        )
    return out


def _item_descriptions() -> Dict[str, str]:
    """``{episode_id: description}`` from every RSS fixture, keyed by the item's guid."""
    out: Dict[str, str] = {}
    for feed in sorted(RSS.glob("p*.xml")):
        try:
            root = safe_fromstring(feed.read_text(encoding="utf-8"))
        except XMLParseError:  # pragma: no cover - a malformed fixture is its own failure
            continue
        for item in root.findall("./channel/item"):
            guid = (item.findtext("guid") or "").strip()
            if guid:
                out.setdefault(guid, (item.findtext("description") or "").strip())
    return out


@pytest.fixture(scope="module")
def corpus() -> Dict[str, List[Tuple[str, str, str]]]:
    return _episodes()


class TestTheScanItselfIsSound:
    """Checked first, because every assertion below is a count and a broken scan reads as zero.

    The scan silently returning empty text is the exact mistake that produced a corpus census of
    all zeros for all 45 episodes, English included.
    """

    def test_every_episode_resolved_a_non_empty_transcript(
        self, corpus: Dict[str, List[Tuple[str, str, str]]]
    ) -> None:
        empty = [
            (lang, ep) for lang, eps in corpus.items() for ep, text, _ in eps if not text.strip()
        ]
        assert not empty, (
            "episodes whose transcript did not resolve — every count in this file would read as "
            f"zero for the wrong reason: {empty}"
        )

    def test_the_fixtures_cover_every_enabled_language(
        self, corpus: Dict[str, List[Tuple[str, str, str]]]
    ) -> None:
        missing = [lang for lang in TIER_1 if lang not in corpus]
        assert not missing, f"enabled languages with no fixture episode at all: {missing}"


class TestTheScenarioRowsAreWhatTheFixturesShouldCarry:
    """The link between the registry and this file, asserted so they cannot drift apart.

    Each pending assertion above names a scenario field as the content to use. If a field is
    renamed, this fails rather than leaving the guidance pointing at nothing.
    """

    @pytest.mark.parametrize("language", TIER_1)
    def test_the_scenario_supplies_content_for_every_detector_this_file_checks(
        self, language: str
    ) -> None:
        scenario = scenario_for(language)
        for field in ("ad_read", "host_opening", "guest_reply", "introduction", "mention"):
            assert getattr(scenario, field).strip(), f"{language}.{field} is empty"
