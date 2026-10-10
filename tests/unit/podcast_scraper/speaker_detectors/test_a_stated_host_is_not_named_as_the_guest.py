"""A name the feed states as a host, answered with the role "guest", is not believed (D27).

The model contradicts itself: the person it names hosts the show, the role it gives the voice says
the voice is visiting. Every case on record was the wrong name:

- El Hilo 2026-10-02 (eval relabels, 2026-10-10): "Silvia Viñas, guest" on the guest's voice
  (Vanessa Torres, 42% of the episode) in 3 of 12 runs; Silvia Viñas is not on the episode.
- The Rest Is History, "The Terror: The Fall of Robespierre" (prod snapshot 2026-10-09): "Tom
  Holland, guest" on the voice that says "the speech, Tom, that you regard..." -- Dominic Sandbrook.
- a16z "The Top 100 Consumer AI Apps" (same snapshot): "Elena Burger, guest" on a voice other than
  the one that introduces itself as Elena Burger.

The role goes with the name: on The Rest Is History "guest" sat on a host.
"""

from __future__ import annotations

import json

from podcast_scraper.speaker_detectors.resolution import resolve_voices_and_roles

HOST = "Bienvenidos a El Hilo. Soy Eliezer Budasoff. Hoy hablamos con Vanessa Torres."
GUEST = "Bueno, cuando empezamos a investigar la frontera, lo primero que vimos fue el costo."


def _llm(payload: dict):
    return lambda _prompt: json.dumps({"voices": payload})


def _resolve(payload: dict, report: dict | None = None):
    return resolve_voices_and_roles(
        stated_names=["Vanessa Torres", "Eliezer Budasoff", "Silvia Viñas"],
        voice_texts={"SPEAKER_07": HOST, "SPEAKER_06": GUEST},
        complete=_llm(payload),
        known_hosts=["Eliezer Budasoff", "Silvia Viñas"],
        episode_title="La última frontera",
        report=report,
        language="es",
    )


def test_a_stated_host_answered_as_the_guest_names_nobody() -> None:
    report: dict = {}
    out = _resolve(
        {
            "SPEAKER_07": {"name": "Eliezer Budasoff", "role": "host"},
            "SPEAKER_06": {"name": "Silvia Viñas", "role": "guest"},
        },
        report,
    )
    assert out["SPEAKER_07"].name == "Eliezer Budasoff"
    assert "SPEAKER_06" not in out
    outcome = {v["voice"]: v["outcome"] for v in report["verdicts"]}
    assert outcome["SPEAKER_06"] == "stated_host_as_guest"


def test_the_stated_guest_on_that_voice_is_still_accepted() -> None:
    out = _resolve({"SPEAKER_06": {"name": "Vanessa Torres", "role": "guest"}})
    assert out["SPEAKER_06"].name == "Vanessa Torres"
    assert out["SPEAKER_06"].role == "guest"


def test_a_stated_host_answered_as_a_host_is_accepted() -> None:
    out = _resolve({"SPEAKER_07": {"name": "Eliezer Budasoff", "role": "host"}})
    assert out["SPEAKER_07"].name == "Eliezer Budasoff"


def test_without_known_hosts_nothing_changes() -> None:
    out = resolve_voices_and_roles(
        stated_names=["Silvia Viñas"],
        voice_texts={"SPEAKER_06": GUEST},
        complete=_llm({"SPEAKER_06": {"name": "Silvia Viñas", "role": "guest"}}),
        episode_title="La última frontera",
        language="es",
    )
    assert out["SPEAKER_06"].name == "Silvia Viñas"


class TestASelfIntroductionIsReadInTheTextLanguage:
    """D28, found writing the tests above: `_introduces_itself_as` read English cues only, so a
    voice saying "Soy Eliezer Budasoff" was refused the name it gave itself, as if it were talking
    about him."""

    def test_soy_vouches_for_the_name(self) -> None:
        from podcast_scraper.speaker_detectors.resolution import refuted_by_third_person

        assert not refuted_by_third_person(HOST, "Eliezer Budasoff", "es")

    def test_a_voice_that_names_someone_else_is_still_refuted(self) -> None:
        from podcast_scraper.speaker_detectors.resolution import refuted_by_third_person

        assert refuted_by_third_person(HOST, "Vanessa Torres", "es")

    def test_english_keeps_its_own_cues_only(self) -> None:
        from podcast_scraper.speaker_detectors.resolution import refuted_by_third_person

        assert refuted_by_third_person(HOST, "Eliezer Budasoff", "en")
