"""The shared name-similarity helpers (RFC-131): one implementation for every caller."""

from __future__ import annotations

import pytest

from podcast_scraper.identity import person_match as pm


@pytest.mark.parametrize(
    "a,b,d",
    [
        ("", "", 0),
        ("kitten", "sitting", 3),
        ("Galeotti", "galliotti", 2),  # case-insensitive
        ("ab", "ba", 2),  # a short pair is not shortcut to 1
        ("Moise", "Moiz", 2),
        ("", "abc", 3),
    ],
)
def test_edit_distance(a: str, b: str, d: int) -> None:
    assert pm.edit_distance(a, b) == d
    assert pm.edit_distance(b, a) == d


@pytest.mark.parametrize(
    "a,b,one",
    [
        ("Knutson", "Knudson", True),
        ("Velt", "Vellt", True),
        ("Velt", "velt", False),  # identical ignoring case
        ("Velt", "Vlet", False),  # a swap is two edits
        ("ab", "abcd", False),
    ],
)
def test_one_edit_apart_agrees_with_edit_distance(a: str, b: str, one: bool) -> None:
    assert pm.one_edit_apart(a, b) is one
    assert one == (pm.edit_distance(a, b) == 1)


@pytest.mark.parametrize(
    "word,code",
    [
        ("Robert", "R163"),
        ("Rupert", "R163"),
        ("Roose", "R200"),
        ("Russo", "R200"),
        ("Ashcraft", "A261"),  # H between two codes of the same digit does not separate them
        ("Tymczak", "T522"),
        ("Pfister", "P236"),
        ("O'Neill", "O540"),
        ("", ""),
        ("123", ""),
    ],
)
def test_soundex(word: str, code: str) -> None:
    assert pm.soundex(word) == code


def test_soundex_keeps_a_non_ascii_first_letter() -> None:
    # The roster's behaviour, which every caller now shares. The offline review tool in
    # speaker_detectors/name_canonicalization used to drop such a letter ("S362" here).
    assert pm.soundex("Østergaard") == "Ø236"


def test_the_old_call_sites_are_the_shared_helpers() -> None:
    from podcast_scraper.providers.ml.diarization import roster
    from podcast_scraper.speaker_detectors import name_canonicalization, resolution

    assert roster._edit_distance is pm.edit_distance
    assert roster._soundex is pm.soundex
    assert resolution._edit_distance is pm.edit_distance
    assert resolution._one_edit_apart is pm.one_edit_apart
    assert name_canonicalization._edit_distance is pm.edit_distance
    assert name_canonicalization._soundex is pm.soundex


@pytest.mark.parametrize(
    "a,b,mode,want",
    [
        ("Bernard Leong", "Bernard Leung", "cluster", True),
        ("Jensen Huang", "Jesse Zhang", "cluster", False),  # two tokens differ: two people
        ("Charles I", "Charles II", "cluster", False),  # regnal numbers
        ("Mike Selig", "Michael Selig", "cluster", True),  # nickname escape
        ("Erica Barris", "Erika Barris", "host_variant", True),
        ("Erica Barris", "Erica Barris", "host_variant", False),  # identical is not a variant
        ("Erica", "Erika", "host_variant", False),  # one-word names never collapse here
    ],
)
def test_same_person_dispatches_to_the_named_mode(a: str, b: str, mode: str, want: bool) -> None:
    assert pm.same_person(a, b, mode=mode) is want
    assert pm.same_person(b, a, mode=mode) is want


def test_language_is_accepted_and_does_not_change_todays_rules() -> None:
    # RFC-131, languages: the parameter exists now; the per-language vocabulary is wired in a
    # later, measured step. Until then every language gets today's behaviour.
    for language in (None, "en", "de", "es", "zh", "xx"):
        assert pm.same_person("Bernard Leong", "Bernard Leung", mode="cluster", language=language)
        assert pm.same_person(
            "Erica Barris", "Erika Barris", mode="host_variant", language=language
        )


def test_an_unknown_mode_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown same-person mode"):
        pm.same_person("a b", "a c", mode="fuzzy")


def test_the_old_rule_sites_are_the_modes() -> None:
    from podcast_scraper.kg import entity_clusters
    from podcast_scraper.search.corpus_graph import CorpusGraph

    assert entity_clusters._are_xep_variants is pm.cluster_variants
    assert CorpusGraph._same_person_variant("Erica Barris", "Erika Barris") is True


# ---- S5a: a hyphen read as a space, one middle initial (invented names) ------------------------


@pytest.mark.parametrize(
    "a,b,want",
    [
        # a hyphen against a space: the word counts differ, so D1's rule refused these
        ("Tamsin Orrell-Vanders", "Tamsin Orell Vanders", True),
        ("Corwen Halloway Brisk", "Corwin Halloway-Brisk", True),
        # D1's joins stay: compared as written first, so folding never splits a pair
        ("Yi-Ling Quorn", "Yilin Quorn", True),
        ("Tamsin Orrell-Vanders", "Tamsin Orell-Vanders", True),
        # the fold does not relax the one-drifting-word rule (#2056)
        ("Tamsin Orrell-Vanders", "Tamzin Orell Vanders", False),
        # one middle initial on one side, the rest identical
        ("Corwen T. Halloway", "Corwen Halloway", True),
        ("Corwen T Halloway", "Corwen Halloway", True),
        # ... but no fuzzy spelling or nickname on top of the dropped initial
        ("Rob E. Halloway", "Robert Halloway", False),
        ("Lizzi C. Halloway", "Lizzie Halloway", False),
        # two initials on one side, or an initial on both, are not dropped
        ("Corwen R.R. Halloway", "Corwen Halloway", False),
        ("Corwen H. W. Halloway", "Corwen W. Halloway", False),
        # the one-word rule is unchanged
        ("Corwen", "Corwin", False),
    ],
)
def test_cluster_person_hyphen_and_middle_initial(a: str, b: str, want: bool) -> None:
    assert pm.cluster_person_variants(a, b) is want
    assert pm.cluster_person_variants(b, a) is want


def test_cluster_person_only_adds_to_the_cluster_rule() -> None:
    """Every pair mode ``cluster`` joins, ``cluster_person`` still joins (multi-word names)."""
    pairs = [
        ("Bernard Leong", "Bernard Leung"),
        ("Mike Selig", "Michael Selig"),
        ("Dr. Elena Fischer", "Elena Fischer"),
        ("Data Bricks", "Databricks"),
    ]
    for a, b in pairs:
        assert pm.cluster_variants(a, b, "person")
        assert pm.cluster_person_variants(a, b)


# ---- S5b: the person rule reads the name's language (invented names) ---------------------------


@pytest.mark.parametrize(
    "a,b,language,want",
    [
        # None is today's rule, so every caller that passes no language is unchanged
        ("Korin Vael", "Korin Vaal", None, True),
        ("Korin Vael", "Korin Vaal", "en", True),
        ("Korin Vael", "Korin Vaal", "en-GB", True),
        # unknown (und) or a code with no vocabulary row: equal after normalisation only
        ("Korin Vael", "Korin Vaal", "und", False),
        ("Korin Vael", "Korin Vaal", "ar", False),
        ("Korin Vael", "KORIN VAEL", "und", True),
        ("Korin Vael", "KorinVael", "und", True),
        ("Tamsin Orrell-Vanders", "Tamsin Orrell Vanders", "und", True),
        ("Dr. Korin Vael", "Korin Vael", "und", False),
        ("Vael", "Korin Vael", "und", False),
        ("Corwen T. Halloway", "Corwen Halloway", "und", False),
        # de: the umlaut fold, and the row's titles
        ("Jörn Brüll", "Joern Bruell", "de", True),
        ("Jörn Brüll", "Joern Bruell", None, False),
        ("Herr Korin Vael", "Korin Vael", "de", True),
        ("Herr Korin Vael", "Korin Vael", None, False),
        ("Präsident Korin Vael", "Korin Vael", "de", True),
        ("Korin Vael", "Korin Vaal", "de", True),
        # no English nicknames outside English
        ("Mike Halloway", "Michael Halloway", "en", True),
        ("Mike Halloway", "Michael Halloway", "de", False),
        # family name first: the bare family name is the FIRST word
        ("Wen", "Wen Lirao", "zh", True),
        ("Lirao", "Wen Lirao", "zh", False),
        ("Lirao", "Wen Lirao", "en", True),
    ],
)
def test_cluster_person_by_language(a: str, b: str, language, want: bool) -> None:
    assert pm.cluster_person_variants(a, b, language) is want
    assert pm.cluster_person_variants(b, a, language) is want


def test_the_cluster_mode_ignores_language() -> None:
    """Orgs, the read-time map and the within-episode merge call mode cluster: unchanged."""
    for language in (None, "und", "de", "zh"):
        assert pm.cluster_variants("Korin Vael", "Korin Vaal", "person", language)
        assert pm.cluster_variants("Halloway Labs", "Haloway Labs", "org", language)


# ---- S5o: an organisation's differing words may be one edit apart, no more (invented names) ------


@pytest.mark.parametrize(
    "a,b,want",
    [
        ("Velmora", "Delmora", True),  # one letter: a transcription slip
        ("Velmora", "Belmira", False),  # two letters: names that rhyme
        ("Quorlex", "Quorlexa", True),
        ("West Quorland", "East Quorland", False),  # one word, two edits: two places
        ("Halloway Labs", "Haloway Labs", True),
        ("Data Bricks", "Databricks", True),  # spacing only
        # not joined today and not by this narrowing: the dedup direction is S5o's next part
        ("Quor-Tech", "Quor Tech", False),
        ("Quor Tech", "Quortec", False),
    ],
)
def test_org_differing_words_one_edit_apart(a: str, b: str, want: bool) -> None:
    assert pm.cluster_variants(a, b, "org") is want
    assert pm.cluster_variants(b, a, "org") is want


def test_the_org_limit_does_not_reach_people() -> None:
    assert pm.cluster_variants("Korin Belmira", "Korin Velmora", "person")
    assert pm.cluster_person_variants("Korin Belmira", "Korin Velmora")
