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
