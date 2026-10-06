"""The naming vocabulary is per-language everywhere, and English did not move.

`naming_vocabulary.py` was extracted from `hosts.py` by hand, and a hand-extraction loses rows.
It lost three during this one, each silently and each a different shape of loss:

  * `CUE_FIRST_BODY` dropped two of eight English alternatives — a whole introduction shape;
  * `NONPERSON_AUTHOR_WORDS` dropped nine of forty-five English markers, including the three
    subsets whose comments record the measurement that put them there;
  * `ARTICLE_BEFORE`'s German row GAINED `von`, which is both German's genitive and its by-agent
    marker, so the guard against "…Council of the Americas Online" ate the host of every German
    feed instead — four languages named their host, German named nobody.

None of the three failed a test, because no test read these maps. These do.
"""

from __future__ import annotations

import re

import pytest

from podcast_scraper.languages import TARGET_LANGUAGE
from podcast_scraper.speaker_detectors import hosts, naming_vocabulary

LANGUAGES = naming_vocabulary.TIER_1_LANGUAGES
NON_ENGLISH = tuple(lang for lang in LANGUAGES if lang != "en")


def _maps() -> dict[str, dict]:
    return {
        name: value
        for name, value in vars(naming_vocabulary).items()
        if name.isupper() and isinstance(value, dict)
    }


class TestEveryMapCoversEveryLanguage:
    def test_there_are_maps_to_check(self) -> None:
        # A guard on the guard: if the module is ever emptied or renamed, the tests below would
        # all pass vacuously by iterating nothing.
        assert len(_maps()) >= 30

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_it_has_a_row_for_the_language(self, language: str) -> None:
        missing = sorted(name for name, rows in _maps().items() if language not in rows)
        assert missing == [], f"{language} has no row in: {missing}"

    def test_no_row_is_empty(self) -> None:
        # An empty row is worse than a missing one: the fail-closed intersection in `hosts.py`
        # catches a MISSING language, and a row that is present but empty reads as "this language
        # says nothing here" instead.
        empty = sorted(
            f"{name}[{lang}]"
            for name, rows in _maps().items()
            for lang in LANGUAGES
            if not rows.get(lang)
        )
        assert empty == []

    def test_every_pattern_compiles(self) -> None:
        sub = {"lead": hosts._STATED_LEAD, "names": hosts._STATED_NAME, "show": r"[A-Z]\w*"}
        broken = []
        for name, rows in _maps().items():
            for lang, value in rows.items():
                items = value if isinstance(value, (tuple, list, frozenset, set)) else (value,)
                for item in items:
                    if not isinstance(item, str):
                        continue
                    try:
                        re.compile(item % sub if "%(" in item else item)
                    except re.error as exc:
                        broken.append(f"{name}[{lang}]: {exc}")
        assert broken == []


class TestEnglishDidNotMove:
    """The extraction must be invisible to English. These are the live values it has to equal."""

    def test_the_host_phrases_rebuild_byte_identically(self) -> None:
        built = [
            t % {"lead": hosts._STATED_LEAD, "names": hosts._STATED_NAMES}
            for t in naming_vocabulary.HOST_PHRASE_TEMPLATES["en"]
        ]
        assert built == [p.pattern for p in hosts._HOST_PHRASES]

    def test_the_cue_first_body_keeps_all_eight_alternatives(self) -> None:
        # The regression: a hand-copy that stopped at six. Counted rather than compared so the
        # failure message says "you dropped one", not "a long string differs somewhere".
        assert naming_vocabulary.CUE_FIRST_BODY["en"].count("|") >= 7
        for shape in ("chatting|talking|speaking|sitting", r"with\s+us\s+(?:today\s+)?(?:is|are)"):
            assert shape in naming_vocabulary.CUE_FIRST_BODY["en"]

    def test_the_author_markers_keep_every_measured_token(self) -> None:
        base = naming_vocabulary.NONPERSON_AUTHOR_BASE
        assert len(base) == 45
        # One token from each of the three measured subsets the comments record.
        for token in ("times", "centers?", "plus", "committees?", "librar(?:y|ies)"):
            assert token in base

    def test_the_stated_particle_is_unchanged(self) -> None:
        built = "(?:" + "|".join(naming_vocabulary.STATED_PARTICLES["en"]) + ")"
        assert built == hosts._STATED_PARTICLE

    @pytest.mark.parametrize(
        "name",
        [
            "CUE_FIRST_BODY",
            "CUE_FIRST_PAST_BODY",
            "NAME_FIRST_TAIL",
            "NAME_FIRST_REPORT_TAIL",
            "GREETED_TAIL",
        ],
    )
    def test_the_public_cue_constant_is_the_english_row(self, name: str) -> None:
        # These five are `hosts.py`'s public surface and were imported by name elsewhere
        # (`providers/ml/diarization/roster.py` built its match-form patterns from them; it now
        # reads the per-language maps instead). They stay as plain strings so an out-of-tree
        # caller still gets the analysis-language row rather than a dict it cannot index.
        value = getattr(hosts, name)
        assert isinstance(value, str)
        assert value == getattr(naming_vocabulary, name)[TARGET_LANGUAGE]


class TestNoRowIsSecretlyEnglish:
    #: A row that equals English is an untranslated placeholder — EXCEPT where the two languages
    #: genuinely use the same word, which has to be recorded here rather than tolerated silently,
    #: or this guard decays into "some rows may match English for some reason".
    SAME_ON_PURPOSE = (
        {
            # One shared cross-lingual list, for the reason recorded on the map itself.
            ("STATED_PARTICLES", lang)
            for lang in NON_ENGLISH
        }
        | {
            # Roman numerals and the borrowed "jr"/"sr" are written the same in all six.
            ("GENERATIONAL_SUFFIXES", lang)
            for lang in NON_ENGLISH
        }
        | {
            # German's preposition for "in a place" IS "in". Not a placeholder, a cognate.
            ("PLACE_PREPOSITION_IN", "de"),
        }
    )

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_the_rows_are_not_copies_of_english(self, language: str) -> None:
        copied = sorted(
            name
            for name, rows in _maps().items()
            if (name, language) not in self.SAME_ON_PURPOSE and rows.get(language) == rows.get("en")
        )
        assert copied == []

    def test_every_recorded_exception_is_still_real(self) -> None:
        # The allowlist above must not outlive its reason: once a row diverges from English, its
        # entry here is dead weight that would hide the next placeholder.
        maps = _maps()
        stale = sorted(
            f"{name}[{lang}]"
            for name, lang in self.SAME_ON_PURPOSE
            if maps[name].get(lang) != maps[name].get("en")
        )
        assert stale == []


class TestTheDefectsThatMotivatedTheRows:
    """Each of these FAILED before its map existed. They are the measurements, as tests."""

    @pytest.mark.parametrize(
        "title,description,language,expected",
        [
            (
                "Sesiones de Sendero",
                "Sesiones de Sendero es presentado por la anfitriona Lucía Herrera, "
                "periodista de ciclismo de montaña.",
                "es",
                "Lucía Herrera",
            ),
            (
                "Sessioni di Sentiero",
                "Sessioni di Sentiero è condotto da Chiara Ricci, giornalista di mountain bike.",
                "it",
                "Chiara Ricci",
            ),
            (
                "Sessions de Sentier",
                "Sessions de Sentier est animé par Élodie Chevalier, journaliste de VTT.",
                "fr",
                "Élodie Chevalier",
            ),
            (
                "Pfadgespräche",
                "Pfadgespräche wird moderiert von Lena Hofmann, Mountainbike-Journalistin.",
                "de",
                "Lena Hofmann",
            ),
            (
                "Sessões de Trilha",
                "Sessões de Trilha é apresentado pela anfitriã Inês Carvalho, "
                "jornalista de BTT.",
                "pt",
                "Inês Carvalho",
            ),
            (
                "Trail Sessions",
                "Trail Sessions is hosted by mountain bike journalist Casey Rowe.",
                "en",
                "Casey Rowe",
            ),
        ],
    )
    def test_a_feed_that_states_its_host_names_that_host(
        self, title: str, description: str, language: str, expected: str
    ) -> None:
        # The RECALL half. Before `HOST_PHRASE_TEMPLATES`, all five non-English feeds returned an
        # empty set from a description that states its host in the first sentence.
        assert hosts.hosts_from_feed_statement(title, description, language=language) == {expected}

    def test_german_is_not_blocked_by_its_own_by_agent_marker(self) -> None:
        # German named nobody while the other four worked, because `von` was in ARTICLE_BEFORE.
        assert "von" not in naming_vocabulary.ARTICLE_BEFORE["de"]
        assert "des" in naming_vocabulary.ARTICLE_BEFORE["de"]

    def test_the_english_article_guard_still_refuses_a_proper_noun_tail(self) -> None:
        # The other side of that fix: #2075, which is why ARTICLE_BEFORE exists at all.
        assert (
            hosts.hosts_from_feed_statement(
                "Latin America in Focus",
                "The Council of the Americas Online team brings you conversations.",
                language="en",
            )
            == set()
        )

    @pytest.mark.parametrize(
        "name,language",
        [
            ("Anfitrión Miguel", "es"),
            ("Anfitrión", "es"),
            ("Cumbre Tecnología", "es"),
            ("Conduttore Marco", "it"),
            ("Animateur Pierre", "fr"),
            ("Gastgeber Hans", "de"),
            ("Apresentador João", "pt"),
            ("Host Mike", "en"),
        ],
    )
    def test_a_role_word_plus_a_name_is_not_publishable_in_its_own_language(
        self, name: str, language: str
    ) -> None:
        # The PRECISION half. "Host Mike" was refused and "Anfitrión Miguel" was published, so a
        # Spanish feed minted a person called "Anfitrión Miguel" (§5.2's phantom person, reached
        # through the description, which is never translated).
        assert hosts.is_publishable_speaker_name(name, language=language) is False

    @pytest.mark.parametrize(
        "name,language",
        [
            ("Lucía Herrera", "es"),
            ("Chiara Ricci", "it"),
            ("Élodie Chevalier", "fr"),
            ("Lena Hofmann", "de"),
            ("Inês Carvalho", "pt"),
            ("Casey Rowe", "en"),
        ],
    )
    def test_a_real_name_survives_in_its_own_language(self, name: str, language: str) -> None:
        assert hosts.is_publishable_speaker_name(name, language=language) is True

    @pytest.mark.parametrize(
        "stated,language,expected",
        [
            ("Professor Hannah Fry", "en", "Hannah Fry"),
            ("Doctora Marta Solís Vega", "es", "Marta Solís Vega"),
            ("Dottor Luca Moretti Rossi", "it", "Luca Moretti Rossi"),
            ("Professeur Mathieu Lefèvre Dupont", "fr", "Mathieu Lefèvre Dupont"),
            ("Professorin Lena Hofmann Weber", "de", "Lena Hofmann Weber"),
            ("Doutora Inês Carvalho Lima", "pt", "Inês Carvalho Lima"),
        ],
    )
    def test_an_honorific_is_stripped_in_its_own_language(
        self, stated: str, language: str, expected: str
    ) -> None:
        # `_clean_stated_name`'s docstring used to say the honorific pass was English-only and
        # "the day a language needs them the fix is a row, not a rewrite". This is that day.
        assert hosts._clean_stated_name(stated, language=language) == expected

    def test_italian_carries_the_apocopated_titles(self) -> None:
        # "Dottor Rossi", never "Dottore Rossi" — the apocopated form is the ONLY one that ever
        # stands in front of a name, so a row with just the full forms strips nothing.
        for title in ("dottor", "professor", "monsignor"):
            assert title in naming_vocabulary.HONORIFIC_TITLES["it"]


class TestTheRegistryAdvertisesWhatExists:
    def test_every_language_is_still_complete(self) -> None:
        # The intersection fails CLOSED: a language added to some maps and not others drops out
        # of this set rather than being half-supported.
        assert hosts.NAMING_VOCABULARY_LANGUAGES == frozenset(LANGUAGES)

    def test_the_registry_covers_the_new_maps(self) -> None:
        for key in (
            "host_phrases",
            "host_self_intro",
            "cue_first_body",
            "greeted_tail",
            "honorific_titles",
            "nonperson_author_markers",
            "leading_article",
            "article_before",
        ):
            assert key in hosts._NAMING_VOCABULARY_MAPS

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_the_accessor_returns_the_language_row_not_english(self, language: str) -> None:
        row = hosts.naming_vocabulary_for("host_phrases", language)
        assert row is not None
        assert [p.pattern for p in row] != [p.pattern for p in hosts._HOST_PHRASES]

    def test_the_accessor_has_no_english_fallback(self) -> None:
        # `None` rather than the English row, so "cannot read this language" stays
        # distinguishable from "this language says nothing here".
        assert hosts.naming_vocabulary_for("host_phrases", "ja") is None

    def test_the_intersection_actually_bites(self) -> None:
        # A fail-closed set that never excludes anything is indistinguishable from a hardcoded
        # list. Drop one language from one map and it must disappear from the advertised set.
        probe = dict(hosts._NAMING_VOCABULARY_MAPS)
        probe["recap_markers"] = {
            lang: row for lang, row in probe["recap_markers"].items() if lang != "pt"
        }
        advertised = set.intersection(
            *(
                {lang for lang, row in m.items() if row or name == "possessive_prefix"}
                for name, m in probe.items()
            )
        )
        assert "pt" not in advertised
        assert advertised == set(LANGUAGES) - {"pt"}

    def test_the_cross_module_maps_are_registered(self) -> None:
        # `roster.py`, `resolution.py` and `gi/speakers.py` cannot be imported from `hosts.py`
        # without a cycle, so their maps are registered as plain data. Without these entries a
        # language could be advertised complete while the diarizer reads nothing for it.
        for key in (
            "self_intro_words",
            "this_is_intro",
            "recap_markers",
            "intro_affiliation_tokens",
            "sign_off_cues",
            "greeting_at_open",
            "non_person_label_tokens",
            "generational_suffixes",
        ):
            assert key in hosts._NAMING_VOCABULARY_MAPS


class TestDescriptorsAreNotNames:
    """main's `_DESCRIPTOR_SUFFIX` (11e1e425e) arrived English-only after this arc keyed the rest.

    It refuses "Pulitzer Prize-winning" as a name. Moved into `DESCRIPTOR_PATTERNS` so it is
    covered by the completeness checks above, and so a non-English description's equivalent is
    refused in its own language.
    """

    def test_the_english_row_is_mains_pattern_verbatim(self) -> None:
        assert (
            naming_vocabulary.DESCRIPTOR_PATTERNS["en"]
            == r"\w-(?:winning|nominated|based|born|selling|renowned|acclaimed)\b"
        )

    @pytest.mark.parametrize(
        "name,language",
        [
            ("Pulitzer Prize-winning", "en"),
            ("Emmy-winning Jane Doe", "en"),
            ("Periodista Galardonada", "es"),
            ("Lucía Herrera Galardonada", "es"),
            ("Chiara Ricci Premiata", "it"),
            ("Journaliste Primée", "fr"),
            ("Grammy-prämierte Lena", "de"),
            ("Preisgekrönte Autorin", "de"),
            ("Jornalista Premiada", "pt"),
            # The English row applies to every language: an English phrase in a Spanish feed.
            ("Emmy-winning Jane Doe", "es"),
        ],
    )
    def test_a_descriptor_is_refused(self, name: str, language: str) -> None:
        assert hosts.is_publishable_speaker_name(name, language=language) is False

    @pytest.mark.parametrize(
        "name,language",
        [
            ("Lucía Herrera", "es"),
            ("Chiara Ricci", "it"),
            ("Élodie Chevalier", "fr"),
            ("Lena Hofmann", "de"),
            ("Inês Carvalho", "pt"),
            ("René Dubois", "fr"),
            ("Renato Bellini", "it"),
            ("Jean-Luc Picard", "fr"),
        ],
    )
    def test_a_real_name_is_kept(self, name: str, language: str) -> None:
        assert hosts.is_publishable_speaker_name(name, language=language) is True


class TestTheTwoEnglishOnlyReadersAreLanguageAware:
    """`strip_role_prefix` and `resolution._mentions_of` were English-only readers.

    English keeps main's ASCII patterns (pinned byte-for-byte by test_english_patterns_are_mains);
    the other languages get accent-aware ones, chosen by the feed's language.
    """

    def test_a_french_show_prefix_with_an_accented_initial_is_stripped(self) -> None:
        assert hosts.strip_role_prefix("Écran's Marie Dupont", "fr") == "Marie Dupont"

    def test_on_an_english_feed_it_is_exactly_main(self) -> None:
        # main's ASCII possessive cannot see the accented initial, so nothing is stripped.
        assert hosts.strip_role_prefix("Écran's Marie Dupont", "en") == ("Écran's Marie Dupont")
        assert hosts.strip_role_prefix("Planet Money's Kenny Malone") == "Kenny Malone"
        assert hosts.strip_role_prefix("Your Host Luisa Leni", "en") == "Luisa Leni"

    def test_an_unsupported_language_behaves_as_english(self) -> None:
        assert hosts.strip_role_prefix("Écran's Marie Dupont", "ja") == ("Écran's Marie Dupont")

    @pytest.mark.parametrize("language", ["fr", "es", "de"])
    def test_an_accented_given_name_pairs_from_the_given_name(self, language: str) -> None:
        from podcast_scraper.speaker_detectors import resolution

        pairs = [
            (m.group(1), m.group(2))
            for m in resolution._spoken_full_name(language).finditer("et Élodie Chevalier dit")
        ]
        assert ("Élodie", "Chevalier") in pairs

    @pytest.mark.parametrize("language", ["en", None, "ja"])
    def test_english_pairs_exactly_as_main(self, language: object) -> None:
        from podcast_scraper.speaker_detectors import resolution

        scanner = resolution._spoken_full_name(language)  # type: ignore[arg-type]
        assert scanner is resolution._SPOKEN_FULL_NAME
