"""The previous turn's last sentences go ahead of a unit as numbered context (tier 2, 2026-10-10).

About a fifth of the judge's low units on the first non-English set were who-did-what errors -- a
man rendered "her", "he went" as "I used to go" -- because each turn is translated alone. With
`translation_context_sentences` > 0 the previous unit's last sentences are sent first, numbered
with the unit's own, and their translation is dropped. Off by default.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from podcast_scraper.translation.units import (
    attach_context,
    content_key,
    TranslationUnit,
    UnitSentence,
)


def _unit(uid: str, *texts: str, backchannel: bool = False) -> TranslationUnit:
    sentences = [
        UnitSentence(sent_id=f"{uid}.s{i}", text=t, char_start=0, char_end=len(t))
        for i, t in enumerate(texts)
    ]
    u = TranslationUnit(
        unit_id=uid, turn_id=uid, speaker_label="S", sentences=sentences, backchannel=backchannel
    )
    u.content_key = content_key("es", sentences)
    return u


class TestAttachContext:
    def test_each_unit_gets_the_previous_units_last_sentences(self) -> None:
        a = _unit("t1", "Diego subió al camión.", "Le tiraron una piedra.")
        b = _unit("t2", "Le pegó en la cabeza.")
        given = attach_context([a, b], 1, source_language="es", max_input_tokens=2048)
        assert given == 1
        assert a.context == []
        assert b.context == ["Le tiraron una piedra."]
        assert b.numbered_source == "1. Le tiraron una piedra.\n2. Le pegó en la cabeza."

    def test_context_is_part_of_the_memory_key_and_absent_context_keeps_the_old_key(self) -> None:
        a, b = _unit("t1", "Uno."), _unit("t2", "Dos.")
        before = b.content_key
        attach_context([a, b], 1, source_language="es", max_input_tokens=2048)
        assert b.content_key != before
        assert a.content_key == content_key("es", a.sentences)

    def test_no_unit_is_pushed_past_the_models_input_limit(self) -> None:
        a = _unit("t1", "palabra " * 200)
        b = _unit("t2", "Corto.")
        assert attach_context([a, b], 1, source_language="es", max_input_tokens=300) == 0
        assert b.context == []

    def test_a_backchannel_gets_none_and_zero_means_off(self) -> None:
        a, b = _unit("t1", "Sí."), _unit("t2", "Ajá.", backchannel=True)
        assert attach_context([a, b], 1, source_language="es", max_input_tokens=2048) == 0
        c = _unit("t3", "Otra.")
        assert attach_context([a, c], 0, source_language="es", max_input_tokens=2048) == 0


def _provider(responses: List[Optional[str]]) -> Any:
    from podcast_scraper import config
    from podcast_scraper.providers.vllm.translate_provider import GemmaTranslateProvider

    p = GemmaTranslateProvider(
        config.Config(
            rss="https://e.com/f.xml",
            translate_api_base="http://translator.invalid:8005/v1",
            translate_model="google/translategemma-12b-it",
            translate_verify_served_model=False,
        )
    )
    queue = list(responses)
    sent: List[str] = []

    def fake_translate(text: str, **_kw: Any) -> Dict[str, Any]:
        sent.append(text)
        return {"text": queue.pop(0) if queue else None, "metadata": {"model": "stub"}}

    p.translate = fake_translate  # type: ignore[method-assign]
    p.sent = sent  # type: ignore[attr-defined]
    return p


class TestTheProviderDropsTheContextLines:
    def test_a_single_sentence_with_context_is_sent_numbered_and_only_its_line_kept(self) -> None:
        u = _unit("t2", "Le pegó en la cabeza.")
        u.context = ["Diego subió al camión."]
        p = _provider(["1. Diego got on the truck.\n2. It hit him in the head."])
        got = p.translate_unit(u, source_language="es")
        assert p.sent == ["1. Diego subió al camión.\n2. Le pegó en la cabeza."]
        assert got["alignment"] == "sentence"
        assert got["sentences"] == [{"sent_id": "t2.s0", "en_text": "It hit him in the head."}]

    def test_a_reply_missing_a_line_is_a_mismatch_as_before(self) -> None:
        u = _unit("t2", "Le pegó en la cabeza.", "Se cayó.")
        u.context = ["Diego subió al camión."]
        p = _provider(
            ["1. It hit him.\n2. He fell.", "1. It hit him.\n2. He fell.", "It hit him. He fell."]
        )
        got = p.translate_unit(u, source_language="es")
        assert got["alignment"] == "unit"
        assert "3 sentences" in got["metadata"]["refusals"][0]
        # The whole-unit fallback sends the unit alone, without its context.
        assert p.sent[-1] == "Le pegó en la cabeza. Se cayó."

    def test_without_context_nothing_changes(self) -> None:
        u = _unit("t2", "Le pegó en la cabeza.")
        p = _provider(["It hit him in the head."])
        p.translate_unit(u, source_language="es")
        assert p.sent == ["Le pegó en la cabeza."]


def test_the_switch_is_off_by_default() -> None:
    from podcast_scraper.config import Config

    assert Config(rss="https://e.com/f.xml").translation_context_sentences == 0
