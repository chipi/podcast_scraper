"""The metadata name readers read the feed's language as well as English (D22).

`split_author_names`, `names_the_show`, `looks_like_a_person_name` and `is_plausible_mononym` read
only the English rows of maps that carry one per language. Measured on 585 chart feeds (es, mx, fr,
de, br, it; eval repo `metadata_name_readers_probe_v1.py`, 2026-10-10): a German cast "A und B"
stayed one composite person, an accented one-word name was refused, a publisher with a Spanish or
French article read as a person. The shapes below are the probe's; the names are made up.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors import hosts


class TestSplitAuthorNames:
    @pytest.mark.parametrize(
        ("tag", "language", "expected"),
        [
            ("Anna Berg und Kai Lenz", "de", ["Anna Berg", "Kai Lenz"]),
            ("Anna Berg, Jule Weis und Kai Lenz", "de", ["Anna Berg", "Jule Weis", "Kai Lenz"]),
            ("Eva Ruiz y Óscar Mora", "es-ES", ["Eva Ruiz", "Óscar Mora"]),
            ("Nina Rossi e Luca Bruni", "it", ["Nina Rossi", "Luca Bruni"]),
            ("Hugo Blanc et Marc Petit", "fr", ["Hugo Blanc", "Marc Petit"]),
            # English stays a separator on a non-English feed: El Hilo's tag is in English.
            ("My Network and Big Podcasts", "es", ["My Network", "Big Podcasts"]),
        ],
    )
    def test_the_feed_conjunction_separates_names(self, tag, language, expected) -> None:
        assert hosts.split_author_names(tag, language) == expected

    def test_without_a_language_only_english_separates(self) -> None:
        assert hosts.split_author_names("Anna Berg und Kai Lenz") == ["Anna Berg und Kai Lenz"]

    def test_a_word_containing_the_conjunction_is_not_cut(self) -> None:
        assert hosts.split_author_names("Yolanda Reyes", "es") == ["Yolanda Reyes"]


class TestNamesTheShow:
    def test_the_feed_language_with_form_is_stripped_from_the_title(self) -> None:
        assert hosts.names_the_show("Edeltalk", "Edeltalk - mit Dominik & Kevin", "de")
        assert not hosts.names_the_show("Edeltalk", "Edeltalk - mit Dominik & Kevin")

    def test_the_feed_language_article_is_dropped(self) -> None:
        assert hosts.names_the_show("entrevista", "La Entrevista con Ana Ruiz", "es")

    def test_the_host_in_the_with_form_is_still_a_person(self) -> None:
        assert not hosts.names_the_show("Ana Ruiz", "La Entrevista con Ana Ruiz", "es")


class TestLooksLikeAPersonName:
    @pytest.mark.parametrize(
        ("name", "language"),
        [
            ("El Diario Nacional", "es"),
            ("Le Journal", "fr"),
            ("Il Quotidiano", "it"),
            ("Die Redaktion", "de"),
            ("O Jornal", "pt"),
            ("Anna Berg und Kai Lenz", "de"),
        ],
    )
    def test_an_article_or_conjunction_of_the_feed_language_refuses(self, name, language) -> None:
        assert not hosts.looks_like_a_person_name(name, language)

    @pytest.mark.parametrize(
        ("name", "language"),
        [
            ("Altair de Souza", "pt"),
            ("Gilson da Silva Pupo", "pt"),
            ("José Mario de la Garza", "es"),
            ("Ivonne de los Ríos", "es"),
            ("Sophie von der Tann", "de"),
            ("Anna Di Rocco", "it"),
            ("E. Castelar", "es"),
        ],
    )
    def test_a_name_particle_inside_the_name_is_kept(self, name, language) -> None:
        assert hosts.looks_like_a_person_name(name, language)

    def test_without_a_language_the_english_row_alone_decides(self) -> None:
        assert hosts.looks_like_a_person_name("El Diario Nacional")


class TestIsPlausibleMononym:
    @pytest.mark.parametrize(
        ("token", "language"), [("Zoé", "fr"), ("Müller", "de"), ("Mabê", "pt")]
    )
    def test_an_accented_name_is_a_name(self, token, language) -> None:
        assert hosts.is_plausible_mononym(token, language)

    def test_a_word_of_the_feed_language_is_not(self) -> None:
        assert not hosts.is_plausible_mononym("Elle", "fr")

    def test_english_keeps_its_pinned_ascii_check(self) -> None:
        assert not hosts.is_plausible_mononym("Zoé")


def test_honorific_titles_add_the_feed_language() -> None:
    assert "sra" in hosts.honorific_titles("es")
    assert "dr" in hosts.honorific_titles("es")
    assert "sra" not in hosts.honorific_titles()


class TestOrgMarkersGluedIntoOneToken:
    @pytest.mark.parametrize(
        ("name", "language"),
        [("RadioAgência Senado", "pt-BR"), ("iHeartPodcasts", "es"), ("esRadio", "es")],
    )
    def test_a_glued_marker_refuses_on_a_non_english_feed(self, name, language) -> None:
        assert hosts.has_org_markers(name, language)
        assert hosts.is_network_or_org_author(name, language)

    def test_a_camel_cased_surname_is_still_a_person(self) -> None:
        assert not hosts.has_org_markers("Ana McKenzie", "es")
        assert not hosts.has_org_markers("Luis DeLeón", "es")

    def test_english_feeds_keep_their_pinned_behaviour(self) -> None:
        assert not hosts.has_org_markers("OnePodcast", "en")


def test_senado_s_portuguese_marker_reaches_the_feed_host_detection() -> None:
    """D13 wired the language into the org check but not into this call: "Rádio Senado" was
    seated as the feed's host though `is_network_or_org_author("Rádio Senado", "pt")` is True."""
    found = hosts.detect_hosts_from_feed(
        "Jornal do Senado", None, ["Rádio Senado", "RadioAgência Senado"], language="pt-br"
    )
    assert found == set()
