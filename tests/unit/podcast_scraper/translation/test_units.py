"""Unit packing: turn-bounded, deterministic, and never sending a label to the translator.

WHAT THESE GUARD. Packing is "the one decision v1 cannot cheaply reverse" (RFC-124 §5.1):
changing alignment granularity later re-translates every episode and invalidates every
provenance block. So the properties pinned here are the ones a later refactor must not drift.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.translation.units import (
    content_key,
    DEFAULT_OVERHEAD_TOKENS,
    pack_stats,
    pack_units,
    UnitSentence,
)

pytestmark = pytest.mark.unit


#: Built alongside the turns so offsets and text can never disagree — the packer slices the
#: screenplay, exactly as it does in the pipeline.
_SCREENPLAY: List[str] = [""]


def _turn(
    turn_id: str, texts: List[str], *, label: str = "Maya", backchannel: bool = False
) -> Dict[str, Any]:
    """A turn whose sentence spans index the shared screenplay being accumulated in _SCREENPLAY."""
    sentences = []
    for i, txt in enumerate(texts, start=1):
        start = len(_SCREENPLAY[0])
        _SCREENPLAY[0] += txt
        sentences.append(
            {"sent_id": f"{turn_id}.s{i:02d}", "char_start": start, "char_end": start + len(txt)}
        )
        _SCREENPLAY[0] += " "
    return {
        "turn_id": turn_id,
        "speaker_label": label,
        "backchannel": backchannel,
        "sentences": sentences,
    }


@pytest.fixture(autouse=True)
def _reset_screenplay() -> Any:
    _SCREENPLAY[0] = ""
    yield
    _SCREENPLAY[0] = ""


def _pack(turns: List[Dict[str, Any]], **kw: Any) -> Any:
    kw.setdefault("source_language", "es")
    kw.setdefault("max_input_tokens", 2048)
    kw.setdefault("screenplay_text", _SCREENPLAY[0])
    return pack_units(turns, **kw)


class TestTurnBoundaries:
    def test_a_unit_never_crosses_a_turn(self) -> None:
        """A turn is one speaker's uninterrupted run. Packing across a speaker change would let
        one person's words be rendered in the grammar of another's."""
        units = _pack(
            [
                _turn("t0000", ["Uno.", "Dos."], label="Maya"),
                _turn("t0001", ["Tres."], label="Liam"),
            ]
        )
        assert [u.turn_id for u in units] == ["t0000", "t0001"]
        for u in units:
            assert len({s.sent_id.split(".")[0] for s in u.sentences}) == 1

    def test_a_backchannel_turn_is_its_own_unit(self) -> None:
        """RFC-123 flags backchannels rather than merging them, precisely so they are not folded
        into a neighbour's context. Packing one with a neighbour would undo that on the way to
        the model."""
        units = _pack(
            [
                _turn("t0000", ["Una frase larga."], label="Maya"),
                _turn("t0001", ["Claro."], label="Liam", backchannel=True),
                _turn("t0002", ["Sigo."], label="Maya"),
            ]
        )
        assert [u.backchannel for u in units] == [False, True, False]
        assert units[1].source_text == "Claro."

    def test_unit_ids_are_turn_scoped_and_ordinal(self) -> None:
        units = _pack(
            [_turn("t0007", ["A.", "B."])],
            max_input_tokens=200,
            overhead_tokens=190,
            count_tokens=lambda _t: 9,  # 10 of budget, 9 per sentence -> one each
        )
        assert [u.unit_id for u in units] == ["t0007.u01", "t0007.u02"]

    def test_empty_and_blank_sentences_are_dropped(self) -> None:
        turn = _turn("t0000", ["Real.", "   ", ""])
        units = _pack([turn])
        assert len(units) == 1
        assert [s.text for s in units[0].sentences] == ["Real."]

    def test_a_turn_with_nothing_translatable_yields_no_unit(self) -> None:
        assert _pack([_turn("t0000", ["  ", ""])]) == []


class TestTheBudget:
    def test_sentences_are_packed_greedily_up_to_the_budget(self) -> None:
        # 10 tokens of budget, ~2.2 chars/token -> roughly 22 chars per unit.
        units = _pack(
            [_turn("t0000", ["aaaaaaaaaa.", "bbbbbbbbbb.", "cccccccccc."])],
            max_input_tokens=20,
            overhead_tokens=10,
        )
        assert len(units) >= 2, "a single unit would have exceeded the budget"
        assert sum(len(u.sentences) for u in units) == 3, "no sentence may be lost"

    def test_no_sentence_is_ever_lost_or_duplicated(self) -> None:
        """The invariant that matters most: units partition the turn's sentences."""
        texts = [f"Frase numero {i} con algo de texto." for i in range(20)]
        units = _pack([_turn("t0000", texts)], max_input_tokens=120, overhead_tokens=40)
        packed = [s.text for u in units for s in u.sentences]
        assert packed == texts

    def test_a_single_oversized_sentence_is_kept_whole_and_FLAGGED(self) -> None:
        """It cannot be split without breaking the alignment atom, so it is flagged instead.

        The provider then refuses it and the episode records a failed unit, which is visible.
        Sending it silently returns a translation of its first clause with
        `finish_reason: stop` — measured on the real service at 4,800 prompt tokens.
        """
        huge = "palabra " * 500
        units = _pack([_turn("t0000", [huge])], max_input_tokens=200, overhead_tokens=50)
        assert len(units) == 1
        assert units[0].oversized is True
        assert units[0].sentences[0].text == huge, "kept whole, not truncated"

    def test_an_oversized_sentence_does_not_swallow_its_neighbours(self) -> None:
        huge = "palabra " * 500
        units = _pack(
            [_turn("t0000", ["Corta.", huge, "Tambien corta."])],
            max_input_tokens=200,
            overhead_tokens=50,
        )
        assert [u.oversized for u in units] == [False, True, False]
        assert [len(u.sentences) for u in units] == [1, 1, 1]

    def test_an_exact_token_counter_is_used_when_given(self) -> None:
        """Normally the provider's /tokenize — the number the SERVER would use, which removes
        the estimate entirely."""
        calls: List[str] = []

        def counter(text: str) -> int:
            calls.append(text)
            return 5

        units = _pack(
            [_turn("t0000", ["A.", "B.", "C."])],
            max_input_tokens=20,
            overhead_tokens=10,
            count_tokens=counter,
        )
        assert calls, "the exact counter must be consulted"
        assert len(units) == 2, "10 budget / 5 per sentence = 2 per unit"

    def test_a_counter_returning_none_falls_back_to_the_estimate(self) -> None:
        units = _pack(
            [_turn("t0000", ["A.", "B."])],
            max_input_tokens=2048,
            count_tokens=lambda _t: None,
        )
        assert len(units) == 1

    def test_the_overhead_default_is_bigger_than_the_measured_template(self) -> None:
        """The prompt's fixed instruction measured 74 tokens; the default reserves more so a
        boundary unit cannot land one token over — that failure is deterministic and silent."""
        assert DEFAULT_OVERHEAD_TOKENS > 74


class TestTheModelNeverSeesALabel:
    def test_the_payload_excludes_the_speaker_label(self) -> None:
        """D-24: labels bypass the translator entirely. Sending one would rename the same person
        inconsistently between units."""
        units = _pack([_turn("t0000", ["Hola."], label="Maya Koster")])
        assert "Maya" not in units[0].source_text
        assert "Maya" not in units[0].numbered_source
        assert units[0].speaker_label == "Maya Koster", "carried for provenance, not for sending"

    def test_the_numbered_payload_is_one_line_per_sentence(self) -> None:
        units = _pack([_turn("t0000", ["Uno.", "Dos."])])
        assert units[0].numbered_source == "1. Uno.\n2. Dos."


class TestContentKey:
    def test_it_is_stable_across_renumbering_and_offsets(self) -> None:
        """RFC-124 §5.1b. A rename changes `Label:` prefix lengths and therefore every offset,
        but never the unit TEXT — so the key still hits and a naming repair costs a re-render
        instead of a re-translation. Naming repair is the most common repair in this corpus.
        """
        a = [UnitSentence("t0000.s01", "Hola.", 6, 11)]
        b = [UnitSentence("t0099.s07", "Hola.", 9999, 10004)]
        assert content_key("es", a) == content_key("es", b)

    def test_it_changes_with_the_text(self) -> None:
        a = [UnitSentence("x", "Hola.", 0, 5)]
        b = [UnitSentence("x", "Adios.", 0, 6)]
        assert content_key("es", a) != content_key("es", b)

    def test_it_changes_with_the_source_language(self) -> None:
        s = [UnitSentence("x", "Hola.", 0, 5)]
        assert content_key("es", s) != content_key("pt", s)

    def test_sentence_boundaries_are_part_of_the_key(self) -> None:
        """Two sentences and one concatenated sentence are different units even with identical
        joined text — they produce different requests and different alignment."""
        two = [UnitSentence("a", "Hola.", 0, 5), UnitSentence("b", "Adios.", 6, 12)]
        one = [UnitSentence("a", "Hola. Adios.", 0, 12)]
        assert content_key("es", two) != content_key("es", one)


class TestDeterminismAndStats:
    def test_packing_the_same_input_twice_gives_identical_units(self) -> None:
        """A unit's identity is its position, so any wobble would silently re-key provenance."""
        turns = [_turn("t0000", [f"Frase {i}." for i in range(12)])]
        a = _pack(turns, max_input_tokens=120, overhead_tokens=40)
        b = _pack(turns, max_input_tokens=120, overhead_tokens=40)
        assert [(u.unit_id, u.content_key) for u in a] == [(u.unit_id, u.content_key) for u in b]

    def test_stats_report_what_the_manifest_needs(self) -> None:
        units = _pack(
            [
                _turn("t0000", ["Uno.", "Dos."]),
                _turn("t0001", ["Claro."], backchannel=True),
            ]
        )
        st = pack_stats(units)
        assert st == {
            "units": 2,
            "sentences": 3,
            "backchannel_units": 1,
            "oversized_units": 0,
            "max_sentences_per_unit": 2,
        }

    def test_stats_survive_an_empty_episode(self) -> None:
        assert pack_stats([])["units"] == 0


class TestOverRealTurns:
    def test_it_packs_the_real_fixture_corpus_turns(self) -> None:
        """Driven through the REAL builder over a real English episode, so the packer is tested
        against the artifact shape the pipeline actually writes rather than hand-made dicts."""
        import json
        from pathlib import Path

        from podcast_scraper.providers.ml.diarization.formatting import (
            format_diarized_screenplay_with_offsets,
        )
        from podcast_scraper.providers.ml.diarization.turns import build_turns

        repo = Path(__file__).resolve().parents[4]
        seg = (
            repo
            / "tests/fixtures/app-validation-corpus/v3/feeds/p01/run_20260101-000000"
            / "transcripts/p01_e01.segments.json"
        )
        raw = json.loads(seg.read_text(encoding="utf-8"))
        text, offsets = format_diarized_screenplay_with_offsets(raw)
        turns = build_turns(offsets, screenplay_text=text)

        units = pack_units(
            [t.to_dict() for t in turns.turns],
            screenplay_text=text,
            source_language="en",
            max_input_tokens=2048,
        )
        assert units, "a real episode must produce units"
        # Every sentence of every turn survives packing, in order.
        expected = [
            text[s.char_start : s.char_end]
            for t in turns.turns
            for s in t.sentences
            if text[s.char_start : s.char_end].strip()
        ]
        got = [s.text for u in units for s in u.sentences]
        assert got == expected
        # And every unit's span is real text from the screenplay.
        for u in units:
            for s in u.sentences:
                assert text[s.char_start : s.char_end] == s.text
