"""The bleed rescue and the show-mononym refusal read the feed's language (D24).

`_rescued_from_bleed` read host acts in the feed's language (D18) but guest replies in English
only, so a Spanish cluster never showed a guest reply and the rescue's first safeguard always
passed. `_is_show_mononym` dropped only an English article from the show title. Both only refuse
more with the language added; English is unchanged.
"""

from __future__ import annotations

from podcast_scraper.providers.ml.diarization import roster


def _cluster_es() -> str:
    # A host act and, one turn later, the guest's reply bled into the same cluster.
    for act in roster._host_acts("es"):
        for text in ("Bienvenidos a El Programa.", "Esto es El Programa.", "Hola, bienvenidos."):
            if act.search(text):
                return f"{text} Gracias por invitarme, es un placer."
    raise AssertionError("no Spanish host act matched the sample openings")


def test_a_spanish_guest_reply_beside_the_host_act_blocks_the_rescue() -> None:
    text = _cluster_es()
    assert not roster._rescued_from_bleed(
        "S1",
        text,
        "Ana Ruiz",
        language="es",
        voice_intro={},
        voice_texts={"S1": text},
        talk_share={"S1": 0.3, "S2": 0.7},
    )


def test_english_keeps_its_reply_list() -> None:
    assert roster._guest_replies("en") == tuple(roster._GUEST_REPLY_WIDE)
    assert roster._guest_replies(None) == tuple(roster._GUEST_REPLY_WIDE)
    assert len(roster._guest_replies("es")) > len(roster._GUEST_REPLY_WIDE)


def test_the_feed_language_article_does_not_hide_the_show_name() -> None:
    assert roster._is_show_mononym("Partidazo", "El Partidazo de COPE", [], "es")
    assert not roster._is_show_mononym("Partidazo", "El Partidazo de COPE", [])


def test_a_stated_host_by_that_name_is_still_a_person() -> None:
    assert not roster._is_show_mononym(
        "Partidazo", "El Partidazo de COPE", ["Partidazo Ruiz"], "es"
    )


class TestATitleInAnyLanguageIsNotTheGivenName:
    def test_spanish_and_italian_titles_do_not_split_one_person(self) -> None:
        assert roster._same_person("Sra. Ana Ruiz", "Ana Ruiz")
        assert roster._same_person("Dott. Luca Bruni", "Luca Bruni")
        assert roster._core_name_tokens("Herr Kai Lenz") == ["Kai", "Lenz"]

    def test_a_given_name_that_is_a_title_elsewhere_stays(self) -> None:
        # Spanish "don" is an English given name; two tokens never lose it.
        assert roster._given_tokens("Don Lemon") == ["Don", "Lemon"]
        assert not roster._same_person("Don Lemon", "Jim Lemon")
        assert roster._strip_titles("Don Lemon") == ["don", "lemon"]

    def test_a_french_m_is_kept_because_it_is_an_english_initial(self) -> None:
        assert roster._core_name_tokens("M. Night Shyamalan") == ["M", "Night", "Shyamalan"]

    def test_an_english_military_title_is_a_title_too(self) -> None:
        # Replayed on the prod snapshot of 2026-10-09 (2,454 English episodes): no name changed.
        assert roster._core_name_tokens("General Mark Milley") == ["Mark", "Milley"]
        assert roster._core_name_tokens("General Milley") == ["General", "Milley"]


def test_a_greeting_is_read_in_the_feed_language() -> None:
    assert roster._greeted_names("Ana Ruiz, bienvenida al programa.", "es") == ["Ana Ruiz"]
    assert roster._greeted_names("Ana Ruiz, bienvenida al programa.") == []
    assert roster._greeted_names("Kara Swisher, welcome to the show.") == ["Kara Swisher"]
