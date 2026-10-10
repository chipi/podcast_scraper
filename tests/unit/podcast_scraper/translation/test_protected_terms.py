"""Names the episode's metadata states are protected from the translator (tier 2, 2026-10-10).

TranslateGemma rendered "La Boca" as "Mouth" and the podcast "Central" as "flagship". The names
come from the episode's own metadata, are swapped for placeholders before the request and put
back after; a reply that loses a placeholder means the unit is translated unprotected instead.
"""

from __future__ import annotations

from podcast_scraper.translation import protected_terms as pt


class TestCandidateTerms:
    def test_names_from_title_and_description_and_the_show(self) -> None:
        terms = pt.candidate_terms(
            "Desde Central, Superclásico 94: el crimen",
            "En 1994 dos hinchas de River fueron asesinados por la barra de Boca Juniors en "
            "Buenos Aires.",
            show_title="Radio Ambulante",
        )
        assert "Radio Ambulante" in terms
        assert "Boca Juniors" in terms and "Buenos Aires" in terms and "River" in terms
        assert "Central" in terms and "Superclásico 94" in terms

    def test_a_capitalised_word_opening_a_sentence_is_not_a_name(self) -> None:
        assert "Hoy" not in pt.candidate_terms("Hoy hablamos de agua.")

    def test_longest_first_so_a_longer_name_wins(self) -> None:
        terms = pt.candidate_terms("Hablamos de Boca y de Boca Juniors.")
        assert terms.index("Boca Juniors") < terms.index("Boca")


class TestProtectAndRestore:
    def test_round_trip(self) -> None:
        lines = ["Los chabones de Boca llegaron.", "Ganó en Buenos Aires con Boca."]
        sent, mapping = pt.protect(lines, ["Buenos Aires", "Boca"])
        assert sent == ["Los chabones de [1] llegaron.", "Ganó en [2] con [1]."]
        back = pt.restore(["The [1] guys arrived.", "It won in [2] with [1]."], sent, mapping)
        assert back == ["The Boca guys arrived.", "It won in Buenos Aires with Boca."]

    def test_lowercase_is_a_word_not_the_name(self) -> None:
        sent, mapping = pt.protect(["Abrió la boca."], ["Boca"])
        assert sent == ["Abrió la boca."] and mapping == {}

    def test_a_lost_or_extra_placeholder_refuses(self) -> None:
        sent, mapping = pt.protect(["Llegó a Boca."], ["Boca"])
        assert pt.restore(["He arrived at Mouth."], sent, mapping) is None
        assert pt.restore(["He arrived at [1] and [1]."], sent, mapping) is None


class TestWhatIsNotProtected:
    def test_german_gets_only_the_show_title(self) -> None:
        terms = pt.candidate_terms(
            "Organisierte Kriminalität in Europa", show_title="SWR Das Wissen", language="de"
        )
        assert terms == ["SWR Das Wissen"]

    def test_a_one_word_name_is_not_protected_where_a_sentence_starts(self) -> None:
        sent, _ = pt.protect(
            ["Apoya el programa. Gracias.", "Le dije: Apoya.", "Fue a Apoya."], ["Apoya"]
        )
        assert sent == ["Apoya el programa. Gracias.", "Le dije: Apoya.", "Fue a [1]."]

    def test_an_id_string_is_not_a_name(self) -> None:
        assert not pt.candidate_terms("Ver Y3JpZDovL25kci5kZS80ODc4IGNyaWQ6Ly9uZHIu aquí.")


class TestTheProviderProtectsAndFallsBack:
    @staticmethod
    def _run(replies: list) -> tuple:
        from podcast_scraper import config
        from podcast_scraper.providers.vllm.translate_provider import GemmaTranslateProvider
        from podcast_scraper.translation.units import TranslationUnit, UnitSentence

        p = GemmaTranslateProvider(
            config.Config(
                rss="https://e.com/f.xml",
                translate_api_base="http://translator.invalid:8005/v1",
                translate_model="google/translategemma-12b-it",
                translate_verify_served_model=False,
            )
        )
        queue, sent = list(replies), []

        def fake(text: str, **_kw: object) -> dict:
            sent.append(text)
            return {"text": queue.pop(0) if queue else None, "metadata": {"model": "stub"}}

        p.translate = fake  # type: ignore[method-assign]
        text = "Los chabones de Boca llegaron."
        unit = TranslationUnit(
            unit_id="u1",
            turn_id="t1",
            speaker_label="S",
            sentences=[UnitSentence(sent_id="t1.s0", text=text, char_start=0, char_end=len(text))],
            protected_terms=["Boca"],
        )
        return p.translate_unit(unit, source_language="es"), sent

    def test_an_intact_placeholder_brings_the_name_back(self) -> None:
        got, sent = self._run(["The [1] guys arrived."])
        assert sent == ["Los chabones de [1] llegaron."]
        assert got["sentences"][0]["en_text"] == "The Boca guys arrived."
        assert got["metadata"]["protected_terms"] == ["Boca"]

    def test_a_lost_placeholder_falls_back_to_the_plain_translation(self) -> None:
        got, sent = self._run(["The guys from Mouth arrived.", "The Boca guys arrived."])
        assert sent == ["Los chabones de [1] llegaron.", "Los chabones de Boca llegaron."]
        assert got["sentences"][0]["en_text"] == "The Boca guys arrived."
        assert "placeholder" in got["metadata"]["refusals"][0]


class TestTheShowTitleOnly:
    def test_a_multi_word_title_is_matched_whatever_its_case(self) -> None:
        sent, mapping = pt.protect(["Estamos de vuelta en el hilo."], ["El hilo"])
        assert sent == ["Estamos de vuelta en [1]."] and mapping == {"[1]": "El hilo"}

    def test_a_one_word_title_stays_case_sensitive(self) -> None:
        sent, _ = pt.protect(["Es la idea central del libro."], ["Central"])
        assert sent == ["Es la idea central del libro."]

    def test_with_no_texts_only_the_show_title_is_a_term(self) -> None:
        assert pt.candidate_terms(show_title="El hilo", language="es") == ["El hilo"]
