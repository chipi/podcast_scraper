"""Detector vocabularies are keyed by language, and say so when they cannot read a language.

THE CATEGORY THIS CLOSES. Three separate detectors decided things about an episode by matching
English phrases, with nothing in the code recording that the phrases were English:

====================================================  ==============================  ===========
constant                                              decides                         English in
====================================================  ==============================  ===========
``gi.filters._AD_PATTERNS``                           where the ads are               "sponsored
                                                                                      by", "dot
                                                                                      com slash"
``speaker_detectors.constants.INTERVIEW_*_PATTERNS``   is this an interview, who is    "joined by",
                                                      the guest, who is merely        "conversation
                                                      mentioned                       with"
``speaker_detectors.hosts._{HOST,GUEST}_SPEECH_ACTS``  which voice is the host         "welcome
                                                                                      back to"
====================================================  ==============================  ===========

Read against a Spanish body, each one matches nothing — and zero hits is indistinguishable from
"this episode has no ads / no guest / no host", which is a confident wrong answer rather than a
missing one. That is the failure these tests exist to make impossible to reintroduce.

WHY THIS IS LATENT AND NOT A LIVE BUG, which is also why the English rows must stay byte-identical:
under D-44 the canonical transcript body is always the ANALYSIS language, so every caller reads
English by construction today. The map buys the shape, not a behaviour change — and
``TestTheEnglishRowsAreWhatShipped`` is the test that keeps the second half of that sentence true.

WHAT IS *NOT* COVERED, stated plainly: no non-English phrase in any of these rows has ever been
matched against real non-English podcast audio or a real non-English feed description. They are
translations of the English categories, authored 2026-10-02. The fixtures below are sentences
written alongside the patterns, so they prove the rows are WELL-FORMED and INTERNALLY CONSISTENT —
they cannot prove recall or precision on real speech. The first non-English ASR run (#2187) is the
measurement; until it happens, treat the non-English rows as a starting vocabulary.
"""

from __future__ import annotations

import pathlib
import re
import unicodedata
from typing import Dict, List

import pytest

from podcast_scraper.gi.filters import (
    _AD_HITS_THRESHOLD,
    _AD_PATTERNS,
    AD_PATTERNS_BY_LANGUAGE,
    ad_patterns_for,
    AD_PATTERNS_LANGUAGES,
)
from podcast_scraper.languages import is_language_enabled, language_registry, TARGET_LANGUAGE
from podcast_scraper.speaker_detectors.constants import (
    interview_cue_patterns_for,
    INTERVIEW_INDICATOR_PATTERNS,
    INTERVIEW_INDICATOR_PATTERNS_BY_LANGUAGE,
    INTERVIEW_TRAILING_GAPPED_PATTERNS,
    INTERVIEW_TRAILING_GAPPED_PATTERNS_BY_LANGUAGE,
    INTERVIEW_TRAILING_PATTERNS,
    INTERVIEW_TRAILING_PATTERNS_BY_LANGUAGE,
    MENTIONED_ONLY_PATTERNS,
    MENTIONED_ONLY_PATTERNS_BY_LANGUAGE,
    SPEAKER_CUE_LANGUAGES,
)
from podcast_scraper.speaker_detectors.guests import _CUE_MAX_GAP
from podcast_scraper.speaker_detectors.hosts import (
    _clean_stated_name,
    _GUEST_SPEECH_ACTS,
    _GUEST_SPEECH_ACTS_BY_LANGUAGE,
    _HOST_SPEECH_ACTS,
    _HOST_SPEECH_ACTS_BY_LANGUAGE,
    _LEADING_JUNK_BY_LANGUAGE,
    _NAMING_VOCABULARY_MAPS,
    _POSSESSIVE_PREFIX_BY_LANGUAGE,
    naming_vocabulary_for,
    NAMING_VOCABULARY_LANGUAGES,
)
from tests._detector_scenarios import scenario_for, SCENARIO_LANGUAGES

pytestmark = pytest.mark.unit

#: The languages the arc committed to carrying content in (D-29 tier 1). Derived from the registry
#: rather than written down, so enabling a sixth language fails these tests instead of quietly
#: shipping a language no detector can read.
TIER_1 = sorted(c for c in language_registry() if is_language_enabled(c))

#: The four cue maps, so a test can assert over all of them without naming each one twice.
CUE_MAPS: Dict[str, Dict[str, List[str]]] = {
    "leading": INTERVIEW_INDICATOR_PATTERNS_BY_LANGUAGE,
    "trailing": INTERVIEW_TRAILING_PATTERNS_BY_LANGUAGE,
    "trailing_gapped": INTERVIEW_TRAILING_GAPPED_PATTERNS_BY_LANGUAGE,
    "mentioned_only": MENTIONED_ONLY_PATTERNS_BY_LANGUAGE,
}


def _strip_accents(text: str) -> str:
    """ASR drops diacritics routinely. A row that only matches WITH them stops working on audio."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


class TestEveryTier1LanguageHasVocabulary:
    """An enabled language with no detector vocabulary is content nothing can read."""

    def test_tier_1_is_the_five_plus_english(self) -> None:
        """Pins the premise the rest of the class rests on, so a registry change is visible here
        rather than as a confusing failure three tests down."""
        assert TIER_1 == ["de", "en", "es", "fr", "it", "pt"], (
            f"the enabled set moved to {TIER_1} — every map below needs the new language, and "
            "that is the point of deriving this list instead of writing it down"
        )

    @pytest.mark.parametrize("language", TIER_1)
    def test_it_has_ad_vocabulary(self, language: str) -> None:
        assert language in AD_PATTERNS_LANGUAGES
        assert ad_patterns_for(language), f"{language} is enabled with no ad-cue vocabulary"

    @pytest.mark.parametrize("language", TIER_1)
    def test_it_has_all_four_cue_lists(self, language: str) -> None:
        assert language in SPEAKER_CUE_LANGUAGES
        cues = interview_cue_patterns_for(language)
        assert cues is not None
        for field in ("leading", "trailing", "trailing_gapped", "mentioned_only"):
            assert getattr(cues, field), f"{language}.{field} is empty"

    @pytest.mark.parametrize("language", TIER_1)
    def test_it_has_host_and_guest_speech_acts(self, language: str) -> None:
        assert _HOST_SPEECH_ACTS_BY_LANGUAGE.get(language), f"{language} has no host speech acts"
        assert _GUEST_SPEECH_ACTS_BY_LANGUAGE.get(language), f"{language} has no guest speech acts"

    @pytest.mark.parametrize("language", TIER_1)
    def test_it_has_a_name_cleanup_rule(self, language: str) -> None:
        assert language in _LEADING_JUNK_BY_LANGUAGE
        # ``None`` is a legitimate value here — see the module's comment on why five of the six
        # languages have no prefix-possessive construction to strip. PRESENCE of the key is what
        # is asserted, so a language cannot be added without someone deciding.
        assert language in _POSSESSIVE_PREFIX_BY_LANGUAGE

    @pytest.mark.parametrize("language", TIER_1)
    def test_it_has_the_whole_naming_vocabulary(self, language: str) -> None:
        """The NAMING vocabulary main added in #2269, which arrived as flat English sets.

        WHY IT MATTERS FOR A NON-ENGLISH FEED EVEN THOUGH THE TRANSCRIPT IS TRANSLATED: these
        read the feed and episode DESCRIPTION, and nothing translates feed metadata (the same
        reason S2.14 exists). A Spanish show's description reaches `_LEADING_ROLE_WORDS` and
        `_PRESENTS` in Spanish, so an English-only row there is not a cosmetic gap — it is the
        host-naming lever silently not firing.
        """
        assert language in NAMING_VOCABULARY_LANGUAGES, (
            f"{language} is enabled but its naming vocabulary is incomplete; "
            "NAMING_VOCABULARY_LANGUAGES is an intersection, so some map is missing its row"
        )
        for name in _NAMING_VOCABULARY_MAPS:
            row = naming_vocabulary_for(name, language)
            # `possessive_prefix` is legitimately `None` for the four languages that postpose the
            # employer; every other map must have something.
            if name == "possessive_prefix":
                assert language in _POSSESSIVE_PREFIX_BY_LANGUAGE
            else:
                assert row, f"{language}.{name} is empty"

    def test_the_naming_accessor_does_not_fall_back_to_english(self) -> None:
        """Same contract as the ad and cue accessors: a language we cannot read returns NOTHING.

        An English fallback would make "no vocabulary for this language" look identical to "this
        language's vocabulary found nothing", and only the second is a measurement.
        """
        for name in _NAMING_VOCABULARY_MAPS:
            assert naming_vocabulary_for(name, "ja") is None, f"{name} fell back for 'ja'"
            assert naming_vocabulary_for(name, "") is None, f"{name} fell back for ''"
        assert naming_vocabulary_for("not_a_map", "en") is None


class TestNamingVocabularyIsNotAccidentallyEnglish:
    """A translated row that is still the English words is the failure this catches.

    Copying the English set into five rows would satisfy every presence check above while
    changing nothing, so the rows are compared for DISTINCTNESS from English on the maps where a
    translation must differ. `trailing_job_tokens` is deliberately excluded: the C-suite
    abbreviations are borrowed unchanged, so overlap there is correct.
    """

    WORD_MAPS = [
        "leading_role_words",
        "show_tail_words",
        "job_title_tokens",
        "number_words",
        "role_or_filler_tokens",
        "org_tail_tokens",
        "place_tail_tokens",
    ]

    @pytest.mark.parametrize("name", WORD_MAPS)
    @pytest.mark.parametrize("language", [lang for lang in TIER_1 if lang != "en"])
    def test_the_row_is_not_just_the_english_row(self, name: str, language: str) -> None:
        english = naming_vocabulary_for(name, "en")
        row = naming_vocabulary_for(name, language)
        assert row != english, f"{language}.{name} is a copy of the English row, not a translation"
        # "podcast" is a loanword in all five, so SOME overlap is expected; a row that is a
        # SUBSET of English has not been translated at all.
        assert not set(row) <= set(
            english
        ), f"{language}.{name} is a subset of English — it has been trimmed, not translated"


class TestTheRecallHalfNeverShipsWithoutTheGuard:
    """Cues without a mentioned-only guard is how a person an episode is merely ABOUT becomes a
    diarized voice's name (#876). The four lists have to move together."""

    def test_the_four_cue_maps_cover_exactly_the_same_languages(self) -> None:
        key_sets = {name: set(m) for name, m in CUE_MAPS.items()}
        assert len(set(map(frozenset, key_sets.values()))) == 1, (
            "the cue maps disagree about which languages they cover, so some language has recall "
            f"cues without its precision guard or vice versa: {key_sets}"
        )

    def test_the_advertised_language_set_is_not_narrower_than_the_data(self) -> None:
        """``SPEAKER_CUE_LANGUAGES`` is an intersection, so a half-added language fails CLOSED —
        it is simply not advertised. That is the safe direction, but it is also silent, so this
        test is what makes the half-add loud."""
        assert SPEAKER_CUE_LANGUAGES == set(INTERVIEW_INDICATOR_PATTERNS_BY_LANGUAGE)


class TestAnUnreadableLanguageSaysSoInsteadOfPretending:
    """The contract that distinguishes this from the flat lists: no English fallback.

    An English fallback returns zero hits on Spanish, which every caller downstream reads as
    "clean". Empty / ``None`` is the same zero with a label on it.
    """

    def test_ad_patterns_are_empty_for_an_unknown_language(self) -> None:
        assert ad_patterns_for("ja") == ()
        assert ad_patterns_for("ja") != _AD_PATTERNS, "an English fallback is the bug, not the fix"

    def test_cue_patterns_are_none_for_an_unknown_language(self) -> None:
        assert interview_cue_patterns_for("ja") is None

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_an_absent_language_is_not_silently_english(self, empty: object) -> None:
        assert ad_patterns_for(empty) == ()  # type: ignore[arg-type]
        assert interview_cue_patterns_for(empty) is None  # type: ignore[arg-type]

    def test_a_regional_tag_resolves_to_its_base_language(self) -> None:
        """``es-ES`` off an RSS feed must find the Spanish row, not fall off the end of the map."""
        assert ad_patterns_for("es-ES") == ad_patterns_for("es")
        assert ad_patterns_for("  PT-br  ") == ad_patterns_for("pt")
        assert interview_cue_patterns_for("de-AT") == interview_cue_patterns_for("de")


class TestTheEnglishRowsAreWhatShipped:
    """D-44 means every caller reads English today, so the restructure must be a no-op for it.

    Asserted by IDENTITY against the map's own ``en`` row rather than by re-listing the patterns:
    a copy of the list in the test would pass while the resolved name pointed somewhere else,
    which is the only failure mode worth catching here.
    """

    def test_ad_patterns_resolve_to_the_english_row(self) -> None:
        assert _AD_PATTERNS is AD_PATTERNS_BY_LANGUAGE[TARGET_LANGUAGE]

    def test_the_cue_lists_resolve_to_the_english_rows(self) -> None:
        assert INTERVIEW_INDICATOR_PATTERNS is INTERVIEW_INDICATOR_PATTERNS_BY_LANGUAGE["en"]
        assert INTERVIEW_TRAILING_PATTERNS is INTERVIEW_TRAILING_PATTERNS_BY_LANGUAGE["en"]
        assert (
            INTERVIEW_TRAILING_GAPPED_PATTERNS
            is INTERVIEW_TRAILING_GAPPED_PATTERNS_BY_LANGUAGE["en"]
        )
        assert MENTIONED_ONLY_PATTERNS is MENTIONED_ONLY_PATTERNS_BY_LANGUAGE["en"]

    def test_the_speech_acts_resolve_to_the_english_rows(self) -> None:
        assert _HOST_SPEECH_ACTS is _HOST_SPEECH_ACTS_BY_LANGUAGE[TARGET_LANGUAGE]
        assert _GUEST_SPEECH_ACTS is _GUEST_SPEECH_ACTS_BY_LANGUAGE[TARGET_LANGUAGE]

    def test_the_measured_english_ad_patterns_are_all_still_there(self) -> None:
        """The English row is the only MEASURED one (11/100 episodes of a real corpus at the
        >= 2-distinct threshold), so losing one of its patterns in a reshuffle is a real
        regression. Spot-checked on the phrases the measurement named."""
        patterns = {p.pattern for p in _AD_PATTERNS}
        for shipped in (
            r"\bbrought to you by\b",
            r"\bsponsored by\b",
            r"\b\w+\s+dot\s+com\b",
            r"\bdot\s+com\s+slash\b",
            r"\b\w+\.(?:com|ai|io|co)\s+slash\b",
            r"\b(?:promo|use)\s+code\s+\w+",
        ):
            assert shipped in patterns, f"the measured English row lost {shipped}"


class TestEveryPatternInEveryRowCompiles:
    """A typo in a row for a language nothing exercises yet would otherwise surface as an import
    error on the day that language is enabled."""

    def test_every_cue_pattern_compiles(self) -> None:
        broken = []
        for name, mapping in CUE_MAPS.items():
            for language, patterns in mapping.items():
                for pattern in patterns:
                    try:
                        re.compile(pattern)
                    except re.error as exc:  # pragma: no cover - the assert carries the report
                        broken.append(f"{name}[{language}] {pattern!r}: {exc}")
        assert not broken, "uncompilable cue patterns:\n  " + "\n  ".join(broken)

    def test_every_cue_pattern_still_compiles_with_a_name_glued_on(self) -> None:
        """How these are ACTUALLY used: ``pattern + gap + name`` / ``name + pattern``. A pattern
        that compiles alone but breaks when concatenated (a dangling group, a trailing ``\\``)
        would fail only at match time, deep inside the detector."""
        gap = r"[\s,'\-\w]{0,40}?"
        name = re.escape("brian chesky")
        broken = []
        for language in sorted(SPEAKER_CUE_LANGUAGES):
            cues = interview_cue_patterns_for(language)
            assert cues is not None
            for pattern in cues.leading + cues.mentioned_only:
                try:
                    re.compile(pattern + gap + name)
                except re.error as exc:  # pragma: no cover
                    broken.append(f"leading[{language}] {pattern!r}: {exc}")
            for pattern in cues.trailing + cues.trailing_gapped:
                try:
                    re.compile(name + pattern)
                except re.error as exc:  # pragma: no cover
                    broken.append(f"trailing[{language}] {pattern!r}: {exc}")
        assert not broken, "patterns that break when glued to a name:\n  " + "\n  ".join(broken)


class TestTheAdRowsAreWellFormed:
    """Each row reaches the threshold on its sponsor read and stays under it on editorial prose.

    NOT A MEASUREMENT of recall on real audio — see the caveat in ``tests/_detector_scenarios``.
    What it proves is that no row is internally broken: the three categories line up, and the
    threshold is reachable at all, which a row of patterns that never co-occur would fail.
    """

    @pytest.mark.parametrize("language", TIER_1)
    def test_the_sponsor_read_reaches_the_distinct_hit_threshold(self, language: str) -> None:
        scenario = scenario_for(language)
        hits = sum(1 for p in ad_patterns_for(language) if p.search(scenario.ad_read))
        assert hits >= _AD_HITS_THRESHOLD, (
            f"{language} scored {hits} distinct patterns on its own sponsor read, below the "
            f"{_AD_HITS_THRESHOLD} the filter requires — the row cannot cut an ad at all"
        )

    @pytest.mark.parametrize("language", TIER_1)
    def test_it_still_reaches_the_threshold_with_the_accents_stripped(self, language: str) -> None:
        """ASR output drops diacritics. A row that needs them works in the fixture and not on the
        DGX, which is the worst place for the difference to show up."""
        stripped = _strip_accents(scenario_for(language).ad_read)
        hits = sum(1 for p in ad_patterns_for(language) if p.search(stripped))
        assert hits >= _AD_HITS_THRESHOLD, (
            f"{language} scored {hits} on the accent-stripped read — some pattern requires a "
            "diacritic that real ASR will not produce"
        )

    @pytest.mark.parametrize("language", TIER_1)
    def test_ordinary_content_stays_under_the_threshold(self, language: str) -> None:
        scenario = scenario_for(language)
        fired = [p.pattern for p in ad_patterns_for(language) if p.search(scenario.clean_content)]
        assert (
            len(fired) < _AD_HITS_THRESHOLD
        ), f"{language} would cut ordinary editorial content as an ad: {fired}"


#: The bounded gap the detector allows between a cue and the name it introduces. Mirrored from
#: ``speaker_detectors.guests._CUE_MAX_GAP`` rather than invented: a test using a looser gap would
#: pass on a cue the detector itself cannot reach.
_GAP = r"[\s,'\-\w]{0," + str(_CUE_MAX_GAP) + r"}?"


class TestTheCueRowsReachAName:
    """Both directions, per language — the recall half and the precision half."""

    @pytest.mark.parametrize("language", TIER_1)
    def test_a_leading_cue_reaches_the_introduced_name(self, language: str) -> None:
        scenario = scenario_for(language)
        cues = interview_cue_patterns_for(language)
        assert cues is not None
        text = scenario.introduction.lower()
        name = re.escape(scenario.guest_name)
        fired = [p for p in cues.leading if re.search(p + _GAP + name, text)]
        assert fired, (
            f"{language}: no leading cue reaches the guest in {text!r} — the guest is invisible "
            "and their voice cluster is free for a merely-mentioned name to claim"
        )

    @pytest.mark.parametrize("language", TIER_1)
    def test_the_mentioned_only_guard_fires_on_a_mention(self, language: str) -> None:
        scenario = scenario_for(language)
        cues = interview_cue_patterns_for(language)
        assert cues is not None
        text = scenario.mention.lower()
        name = re.escape(scenario.mentioned_name)
        fired = [p for p in cues.mentioned_only if re.search(p + _GAP + name, text)]
        assert fired, (
            f"{language}: the mentioned-only guard misses {text!r}, so a person the episode is "
            "merely about is a candidate guest"
        )


def _role(language: str, text: str) -> str:
    """The same host-before-guest precedence ``roles_from_conversation`` applies."""
    if any(p.search(text) for p in _HOST_SPEECH_ACTS_BY_LANGUAGE[language]):
        return "host"
    if any(p.search(text) for p in _GUEST_SPEECH_ACTS_BY_LANGUAGE[language]):
        return "guest"
    return "unknown"


class TestTheSpeechActRowsSeparateHostFromGuest:
    """The role is PERFORMED, not measured by talk time — which is the whole reason these lists
    exist, and why confusing the two is the dangerous failure (#1169)."""

    @pytest.mark.parametrize("language", TIER_1)
    @pytest.mark.parametrize("accents", [True, False], ids=["accented", "stripped"])
    def test_a_host_opening_reads_as_host(self, language: str, accents: bool) -> None:
        text = scenario_for(language).host_opening
        assert _role(language, text if accents else _strip_accents(text)) == "host"

    @pytest.mark.parametrize("language", TIER_1)
    @pytest.mark.parametrize("accents", [True, False], ids=["accented", "stripped"])
    def test_a_guest_reply_reads_as_guest(self, language: str, accents: bool) -> None:
        text = scenario_for(language).guest_reply
        assert _role(language, text if accents else _strip_accents(text)) == "guest"

    @pytest.mark.parametrize("language", TIER_1)
    def test_a_guest_reply_performs_no_host_act(self, language: str) -> None:
        """The dangerous direction. A guest matching a host act crowns the guest — the #1169
        failure on The Daily, where the dominant voice became the host."""
        reply = scenario_for(language).guest_reply
        fired = [p.pattern for p in _HOST_SPEECH_ACTS_BY_LANGUAGE[language] if p.search(reply)]
        assert not fired, f"{language}: a guest's reply performs a host act: {fired}"


class TestNameCleanupDoesNotEatNameParticles:
    """The one place a careless translation would do real damage.

    English can strip a leading "The"/"From" because English names do not begin with them.
    Spanish, Italian, French, German and Portuguese names begin with exactly the words a
    symmetric translation would have added to the junk list.
    """

    def test_english_cleanup_is_unchanged(self) -> None:
        assert _clean_stated_name("Bloomberg's Joe Weisenthal") == "Joe Weisenthal"
        assert _clean_stated_name("At Planet Money") == "Planet Money"
        assert _clean_stated_name("Patrick O'Shaughnessy") == "Patrick O'Shaughnessy"

    @pytest.mark.parametrize("language", TIER_1)
    def test_every_declared_name_particle_survives(self, language: str) -> None:
        scenario = scenario_for(language)
        eaten = [n for n in scenario.name_particles if _clean_stated_name(n, language) != n]
        assert not eaten, (
            f"the {language} junk list ate a name particle — {eaten} are people, not "
            "prepositional phrases"
        )

    @pytest.mark.parametrize("language", TIER_1)
    def test_real_junk_is_still_stripped(self, language: str) -> None:
        """The other half: keeping particles must not cost the cleanup its actual job."""
        scenario = scenario_for(language)
        wrong = [
            (raw, _clean_stated_name(raw, language), expected)
            for raw, expected in scenario.leading_junk
            if _clean_stated_name(raw, language) != expected
        ]
        assert not wrong, f"{language} junk stripping is wrong: {wrong}"

    @pytest.mark.parametrize("language", TIER_1)
    def test_a_language_with_particles_declares_some(self, language: str) -> None:
        """Guards the guard. An empty ``name_particles`` row would make the test above pass
        vacuously, so the five languages that HAVE particles must declare at least one. English
        legitimately has none, which is the asymmetry the whole class is about.
        """
        scenario = scenario_for(language)
        if language == TARGET_LANGUAGE:
            assert scenario.name_particles == (), (
                "English names do not begin with particles — declaring some here would assert a "
                "property English does not have"
            )
        else:
            assert scenario.name_particles, (
                f"{language} declares no name particles, so "
                "`test_every_declared_name_particle_survives` passes without checking anything"
            )

    def test_german_genitive_is_recorded_as_uncovered_not_guessed(self) -> None:
        """German fronts the genitive ("Bloombergs Joe Weisenthal") with no apostrophe, so the
        English pattern has no anchor and ``^\\w+s\\s+`` would turn "Hans Zimmer" into "Zimmer".
        ``None`` is the honest value, and this test is what stops someone filling it in."""
        assert _POSSESSIVE_PREFIX_BY_LANGUAGE["de"] is None
        assert _clean_stated_name("Hans Zimmer", "de") == "Hans Zimmer"

    @pytest.mark.parametrize("language", ["es", "it", "fr", "pt"])
    def test_the_postposing_languages_have_no_prefix_possessive(self, language: str) -> None:
        """These four put the employer after the name ("Joe Weisenthal de Bloomberg"), so there is
        nothing in front to strip and a pattern here could only damage a name."""
        assert _POSSESSIVE_PREFIX_BY_LANGUAGE[language] is None


class TestTheScenarioRegistryCoversWhatIsEnabled:
    """The fail-closed link between ``config/languages.yaml`` and the test data.

    Without this, enabling a sixth language yields tests that quietly do not run for it — which
    looks exactly like passing, and is the state the registry replaces.
    """

    @pytest.mark.parametrize("language", TIER_1)
    def test_an_enabled_language_has_a_scenario(self, language: str) -> None:
        assert language in SCENARIO_LANGUAGES
        assert scenario_for(language).language == language

    def test_asking_for_an_unknown_language_raises_rather_than_defaulting(self) -> None:
        """No English fallback, matching the source-side lookups. A test silently re-running the
        English scenario would report coverage it does not have."""
        with pytest.raises(KeyError, match="no detector scenario"):
            scenario_for("ja")

    def test_a_regional_tag_resolves_to_its_base_language(self) -> None:
        assert scenario_for("es-ES") is scenario_for("es")
        assert scenario_for("  PT-br  ") is scenario_for("pt")


class TestTheMapsFollowTheExistingPrecedent:
    def test_the_query_language_markers_are_shaped_the_same_way(self) -> None:
        """``search.query_language._MARKERS`` was language-keyed before any of this, for the same
        reason. Asserted so the three maps cannot drift into three different conventions."""
        from podcast_scraper.search.query_language import _MARKERS

        described = set(language_registry())
        assert set(_MARKERS) <= described
        keyed: Dict[str, List[str]] = {
            "ad": sorted(AD_PATTERNS_BY_LANGUAGE),
            **{name: sorted(mapping) for name, mapping in CUE_MAPS.items()},
            "host_speech_acts": sorted(_HOST_SPEECH_ACTS_BY_LANGUAGE),
            "guest_speech_acts": sorted(_GUEST_SPEECH_ACTS_BY_LANGUAGE),
            "leading_junk": sorted(_LEADING_JUNK_BY_LANGUAGE),
            "possessive_prefix": sorted(_POSSESSIVE_PREFIX_BY_LANGUAGE),
        }
        undescribed = {
            name: sorted(set(langs) - described)
            for name, langs in keyed.items()
            if set(langs) - described
        }
        assert not undescribed, (
            "a detector map carries a key the language registry does not describe, so it could "
            f"never be reached: {undescribed}"
        )


class TestWhatIsNotMeasured:
    """T6: the gap, asserted against the SOURCE that records it, so it cannot quietly close.

    A test that greps its own file for a corpus path was the first attempt here and it was
    circular — the path appeared in its own assertion. What is actually checkable is that the
    provenance notes stay in the modules they describe: if someone measures the non-English rows,
    these fail and the notes get corrected in the same change.
    """

    @pytest.mark.parametrize(
        "module_path,marker",
        [
            ("src/podcast_scraper/gi/filters.py", "HONEST PROVENANCE OF THE NON-ENGLISH ROWS"),
            ("src/podcast_scraper/speaker_detectors/constants.py", "HONEST PROVENANCE"),
        ],
    )
    def test_the_provenance_note_still_stands_beside_the_vocabulary(
        self, module_path: str, marker: str
    ) -> None:
        """Only the English rows have a measurement behind them, and the maps say so in prose.

        Pinned here because the prose is the ONLY thing distinguishing a translated vocabulary
        from a validated one — strip the note and the next reader has no way to tell.
        """
        repo = pathlib.Path(__file__).resolve().parents[3]
        source = (repo / module_path).read_text(encoding="utf-8")
        assert marker in source, (
            f"{module_path} lost its provenance note. If the non-English rows have now been "
            "measured, replace the note with where — do not just delete it."
        )

    def test_the_asr_measurement_is_still_open(self) -> None:
        """The measurement these rows need is a non-English ASR run, which has never happened —
        every run including the 45-episode corpus build used VTT input. Recorded in the arc as
        V.6b / #2187; this asserts the arc still says so, so "we measured it" cannot be true
        here while the arc says it is open.
        """
        repo = pathlib.Path(__file__).resolve().parents[3]
        arc = (repo / "docs/architecture/MULTILINGUAL_ARC.md").read_text(encoding="utf-8")
        assert "#2187" in arc, (
            "the arc no longer references #2187 — if the non-English ASR measurement has landed, "
            "the non-English vocabulary rows can finally be validated against it, and the "
            "provenance notes above should be updated rather than left claiming otherwise"
        )
