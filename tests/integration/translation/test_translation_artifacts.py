"""A Spanish episode through the real translation stage with a stub translator (S2.4).

WHAT THIS PROVES that the unit tests cannot: that the ledger, the English render and the
COMPLETENESS GATE agree with each other on a real screenplay. The gate is the whole point —
RFC-124 §5.3 says an episode without a complete English set skips summary, GI and KG, and
because the resolver keys on file PRESENCE, "not writing the render" IS the gate. A test that
only checked the happy path would not notice the gate failing open.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.translation.artifacts import (
    english_artifacts_present,
    load_translation_json,
)
from podcast_scraper.workflow import translation_stage as ts

pytestmark = pytest.mark.integration

REL = "transcripts/p10_e01.txt"

_ES_TURNS = [
    ("Maya", "Bienvenidos de nuevo a Sesiones de Sendero. Hoy hablamos de senderos."),
    ("Liam", "Gracias, Maya. Con muchas ganas."),
    ("Maya", "Claro."),
    ("Liam", "El drenaje es la decision mas importante. Lo hemos visto en tres equipos."),
]


def _lay_down_spanish_episode(root: Path) -> str:
    """Write a source-language episode the way the pipeline does."""
    segments: List[Dict[str, Any]] = []
    clock = 0.0
    for label, text in _ES_TURNS:
        segments.append({"start": clock, "end": clock + 6.0, "text": text, "speaker_label": label})
        clock += 6.0
    text, _ = format_diarized_screenplay_with_offsets(segments)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    (root / REL).write_text(text, encoding="utf-8")
    with (root / "transcripts" / "p10_e01.segments.json").open("w", encoding="utf-8") as fh:
        json.dump(segments, fh, indent=0)
    return text


class _StubProvider:
    """Translates by prefixing, so alignment is verifiable without a model.

    `fail_units` names unit ids that should come back failed — which is how the gate gets
    exercised, rather than being asserted about.
    """

    MODEL_INPUT_TOKEN_LIMIT = 2048

    def __init__(self, fail_units: tuple = ()) -> None:
        self.fail_units = set(fail_units)
        self.seen: List[str] = []

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def translate_unit(self, unit: Any, *, source_language: str, target_language: str = "en"):
        self.seen.append(unit.unit_id)
        meta = {"prompt": {"name": "stub", "sha256": "0" * 64}, "attempts": 1}
        if unit.unit_id in self.fail_units:
            return {
                "sentences": [],
                "alignment": "failed",
                "metadata": {**meta, "error": "stubbed failure"},
            }
        return {
            "sentences": [
                {"sent_id": s.sent_id, "en_text": f"EN[{s.text}]"} for s in unit.sentences
            ],
            "alignment": "sentence",
            "metadata": meta,
        }


@pytest.fixture
def cfg() -> config.Config:
    return config.Config(
        rss="https://example.com/p10.xml",
        language="es",
        multilingual_ingest=True,
        translate_api_base="http://translator.invalid:8005/v1",
        translate_model="google/translategemma-12b-it",
        translate_verify_served_model=False,
        save_adfree_transcript=False,
    )


def _run(
    monkeypatch: pytest.MonkeyPatch, root: Path, cfg: config.Config, stub: _StubProvider
) -> Any:
    monkeypatch.setattr(
        "podcast_scraper.translation.factory.create_translation_provider", lambda _c: stub
    )
    return ts.run_translation_stage(
        cfg, transcript_relpath=REL, effective_output_dir=str(root), episode_id="p10_e01"
    )


class TestTheHappyPath:
    def test_a_complete_translation_writes_both_english_artifacts(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _lay_down_spanish_episode(tmp_path)
        stub = _StubProvider()
        got = _run(monkeypatch, tmp_path, cfg, stub)

        assert got.status == "translated"
        assert got.english_ready is True
        assert got.units_failed == 0
        assert english_artifacts_present(REL, str(tmp_path))
        assert "Bienvenidos" in src, "the source is untouched"

    def test_the_english_render_keeps_the_speaker_labels_verbatim(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D-24 / S2.6: the label never went through the translator, so it must appear on the
        English line exactly as it appears on the Spanish one."""
        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider())
        en = (tmp_path / "transcripts" / "p10_e01.en.txt").read_text(encoding="utf-8")
        assert en.startswith("Maya: ")
        assert "Liam: " in en
        assert "EN[" in en, "the stub's marker proves this is translated text"

    def test_one_cue_per_SENTENCE_not_per_unit(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RFC-124 §5.1. With one cue per unit, an ad boundary would drop up to ~45s of speech
        instead of one fragment, and a cue would be a paragraph rather than a subtitle."""
        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider())
        segs = json.loads(
            (tmp_path / "transcripts" / "p10_e01.en.segments.json").read_text(encoding="utf-8")
        )
        doc = load_translation_json(REL, str(tmp_path))
        assert doc is not None
        total_sentences = sum(len(u.sentences) for u in doc.units)
        assert len(segs) == total_sentences > len(doc.units)

    def test_every_cue_carries_unit_id_and_sent_id(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What `resolve_units_for_span` reads to put provenance on every claim (S2.11). The
        formatter drops unknown keys, so these surviving the render is not automatic."""
        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider())
        segs = json.loads(
            (tmp_path / "transcripts" / "p10_e01.en.segments.json").read_text(encoding="utf-8")
        )
        assert segs
        for s in segs:
            assert s["unit_id"] and s["sent_id"]

    def test_the_english_spans_index_the_english_text(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same identity S1.1 guarantees for the source, now for the English body."""
        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider())
        en = (tmp_path / "transcripts" / "p10_e01.en.txt").read_text(encoding="utf-8")
        segs = json.loads(
            (tmp_path / "transcripts" / "p10_e01.en.segments.json").read_text(encoding="utf-8")
        )
        for s in segs:
            assert en[s["char_start"] : s["char_end"]] == s["text"]

    def test_the_ledger_records_the_prompt_that_produced_it(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider())
        doc = load_translation_json(REL, str(tmp_path))
        assert doc is not None
        assert doc.prompt == {"name": "stub", "sha256": "0" * 64}
        assert doc.source_language == "es"
        assert doc.complete is True


class TestTheCompletenessGate:
    def test_ONE_failed_unit_withholds_the_whole_english_render(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The operator's objection, answered. A summary built from an episode with a hole in it
        reads perfectly coherent and is wrong — and the missing unit can be the pivot the whole
        episode turns on. So the render is all-or-nothing."""
        _lay_down_spanish_episode(tmp_path)
        got = _run(monkeypatch, tmp_path, cfg, _StubProvider(fail_units=("t0001.u01",)))

        assert got.status == "failed"
        assert got.english_ready is False
        assert got.units_failed == 1
        assert not english_artifacts_present(REL, str(tmp_path))

    def test_the_ledger_IS_still_written_so_nothing_successful_is_lost(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two artifacts, two meanings: `translation.json` says what was attempted and holds the
        resume state; `.en.txt` says the result is complete and safe to consume."""
        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider(fail_units=("t0001.u01",)))
        doc = load_translation_json(REL, str(tmp_path))
        assert doc is not None
        assert doc.complete is False
        assert len([u for u in doc.units if u.ok]) >= 2, "the successful units are preserved"
        assert doc.failed_units[0].error == "stubbed failure"

    def test_the_source_transcript_still_serves_when_english_is_withheld(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With `.en.txt` absent the resolver falls back to the canonical source — so the right
        thing happens by construction rather than by a flag someone must check."""
        from podcast_scraper.workflow.transcript_resolution import (
            resolve_text_path,
            TranscriptPurpose,
        )

        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider(fail_units=("t0000.u01",)))
        resolved = resolve_text_path(tmp_path, REL, purpose=TranscriptPurpose.TIMELINE)
        assert resolved == tmp_path / REL


class TestResume:
    def test_a_second_run_reuses_the_ledger_and_calls_the_model_for_nothing(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D-33's translation memory, content-keyed. This is what makes a naming repair on a
        translated show cost a re-render instead of a full re-translation."""
        _lay_down_spanish_episode(tmp_path)
        first = _StubProvider()
        _run(monkeypatch, tmp_path, cfg, first)
        assert first.seen, "the first run translates everything"

        second = _StubProvider()
        got = _run(monkeypatch, tmp_path, cfg, second)
        assert second.seen == [], "no unit should have been re-translated"
        assert got.english_ready is True

    def test_a_repair_run_retranslates_ONLY_the_failed_units(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider(fail_units=("t0003.u01",)))

        repair = _StubProvider()
        got = _run(monkeypatch, tmp_path, cfg, repair)
        assert repair.seen == ["t0003.u01"], "only the failure is re-requested"
        assert got.english_ready is True, "and the episode completes"


class TestTranslationReadsTheSourceNotItsOwnOutput:
    """The bug this class exists for was live for one commit, and it was the dangerous kind.

    `TranscriptPurpose.TIMELINE` prefers `.en.txt` by D-38. So once an episode had been
    translated, a second run that asked the RESOLVER what to read got the English body, packed
    units from it, found no matching content keys (English hashes differently from Spanish),
    and re-translated English into English — overwriting the ledger and paying for every unit.

    The rule: the resolver is for CONSUMERS choosing which rendering to read. A producer of a
    rendering must never ask it what to read. Translation's input is defined, not resolved.
    """

    def test_a_second_run_still_reads_the_SPANISH_body(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider())
        assert english_artifacts_present(REL, str(tmp_path)), "first run completed"

        doc_before = load_translation_json(REL, str(tmp_path))
        assert doc_before is not None
        keys_before = [u.content_key for u in doc_before.units]

        _run(monkeypatch, tmp_path, cfg, _StubProvider())
        doc_after = load_translation_json(REL, str(tmp_path))
        assert doc_after is not None
        assert [u.content_key for u in doc_after.units] == keys_before, (
            "the content keys must be identical across runs — different keys mean the second "
            "run hashed a different body, i.e. it read its own English output"
        )

    def test_the_english_text_is_never_double_translated(
        self, tmp_path: Path, cfg: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stub marks every translation with `EN[...]`, so a double pass would nest them."""
        _lay_down_spanish_episode(tmp_path)
        _run(monkeypatch, tmp_path, cfg, _StubProvider())
        _run(monkeypatch, tmp_path, cfg, _StubProvider())
        en = (tmp_path / "transcripts" / "p10_e01.en.txt").read_text(encoding="utf-8")
        assert "EN[EN[" not in en
