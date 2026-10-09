"""A feed description is in the feed's language, and its guest cues have to be read in it.

Found on El Hilo (es, 2026-10-09, non-English measuring arc): the description says "esta semana
hablamos con Vanessa Torres, de Ambiente y Sociedad ... y con Brenda Guillén" — both are the
episode's guests — and corroboration rejected both as "named but never introduced as speaking".
The Spanish row already holds `habla(?:mos|n)?\\s+con\\s+`; the corroboration path read only the
English constants, on the premise that the description is always in the analysis language. D-44
translates the transcript, not the feed.
"""

from __future__ import annotations

from podcast_scraper.speaker_detectors.corroboration import corroborate_guests
from podcast_scraper.speaker_detectors.guests import _is_likely_actual_guest, is_introduced_guest

ES = (
    "Para entender por qué defender un territorio es cada vez más peligroso, esta semana hablamos "
    "con Lucía Herrera, de una organización ambiental, y con Marta Ríos, de otra."
)


def test_a_spanish_cue_introduces_a_spanish_guest() -> None:
    assert _is_likely_actual_guest("Lucía Herrera", "Episodio", ES, language="es")


def test_corroboration_keeps_the_guests_the_description_introduces() -> None:
    assert corroborate_guests(["Lucía Herrera"], "Episodio", ES, language="es") == ["Lucía Herrera"]


def test_without_a_language_the_english_rows_are_unchanged() -> None:
    assert not _is_likely_actual_guest("Lucía Herrera", "Episodio", ES)
    en = "This week we speak with Casey Rowe about chips."
    assert _is_likely_actual_guest("Casey Rowe", "Episode", en)


def test_a_language_without_cue_lists_corroborates_nobody() -> None:
    en = "This week we speak with Casey Rowe about chips."
    assert not _is_likely_actual_guest("Casey Rowe", "Episode", en, language="ja")


def test_the_spanish_mentioned_only_guard_still_refuses() -> None:
    # A person the episode is ABOUT is not a guest in any language.
    text = "Hablamos sobre Lucía Herrera y su legado."
    assert not _is_likely_actual_guest("Lucía Herrera", "Episodio", text, language="es")


def test_the_transcript_intro_reads_its_language_too() -> None:
    assert is_introduced_guest("Lucía Herrera", "hoy hablamos con Lucía Herrera.", language="es")


def test_a_coordinated_second_guest_is_introduced_in_spanish() -> None:
    # "hablamos con A, de X, y con B, de Y": the leading cue reaches only A. English has the same
    # shape covered by "(?:and|along) with". Measured on 714 real El Hilo / Radio Ambulante
    # descriptions (2026-10-09): 34 names newly introduced, 24 of them that episode's guests and
    # none a person merely mentioned; the other 10 were organisations, which the person check in
    # front of this cue refuses.
    text = (
        "Esta semana hablamos con Lucía Herrera, de una organización ambiental, en Colombia "
        "(el país más mortífero para defensores) y con Marta Ríos, de otra. También con Ana Gil."
    )
    assert _is_likely_actual_guest("Marta Ríos", "Episodio", text, language="es")
    assert _is_likely_actual_guest("Ana Gil", "Episodio", text, language="es")
