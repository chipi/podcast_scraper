"""The WHOLE Phase 2 chain on the real Spanish episode, with a stub translator.

WHAT THIS IS FOR. Every Phase 2 slice has its own tests, and the gate still needs "one real
end-to-end run" because none of them walks the chain: pack -> translate -> English artifacts ->
completeness gate -> naming from the English render -> re-render both bodies -> index both
layers. A failure anywhere in the seams between those would pass every existing test.

This walks it, on `tests/fixtures/translation/p10_e01.es.txt` — the actual episode every
measurement in the arc was taken on (V.6a's hazards, S2.10's 174 requests, ADR-157's
chars/token floor).

THE TRANSLATOR IS STUBBED, AND THAT IS THE POINT. It returns the English original of the same
episode, sentence for sentence. So when the real run points at `:8005`, the only new variable is
the MODEL — every other link in the chain has already been exercised on this exact input. A
stubbed rehearsal cannot tell us the translation is GOOD; it tells us that if the translation is
good, everything around it works.

WHAT IT DELIBERATELY DOES NOT COVER, so this is not mistaken for the gate:
- translation quality, latency, or throughput — those need the real model (S2.10, ADR-157);
- ASR and diarization — this starts from a transcript, which is Pass A. Pass B starts from audio
  and is blocked on a Spanish male voice (see tests/fixtures/translation/README.md);
- summary, GI and KG — they need the DGX, so the gate they pass through is asserted here, not
  their output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper import config

pytestmark = pytest.mark.integration

_ASSETS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "translation"
_ES = _ASSETS / "p10_e01.es.txt"
_EN_REFERENCE = _ASSETS / "p01_e01.en.reference.txt"

REL = "transcripts/01 - p10_e01.txt"


def _english_sentences() -> List[str]:
    """Sentences from the English original, in order — the stub's translation memory."""
    out: List[str] = []
    for line in _EN_REFERENCE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "[")) or line.lower().startswith(("host:", "guest:")):
            continue
        _label, _, body = line.partition(": ")
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip())
    return out


class _StubTranslator:
    """Hands back English sentences in order, one per source sentence.

    Order-based rather than content-matched: the Spanish is an abridged translation of the
    English (47 turns against 62), so a content match would fail on the turns that were merged.
    What matters for the chain is that every unit comes back with the RIGHT NUMBER of aligned
    sentences in plausible English — which is exactly what the real provider's numbered
    alignment contract guarantees.
    """

    MODEL_INPUT_TOKEN_LIMIT = 2048

    def __init__(self) -> None:
        self._pool = _english_sentences()
        self._at = 0
        self.units_seen = 0
        self.labels_seen: List[str] = []

    def initialize(self) -> None:  # pragma: no cover - the stub has nothing to set up
        pass

    def cleanup(self) -> None:  # pragma: no cover
        pass

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def translate_unit(self, unit: Any, **_kw: Any) -> Dict[str, Any]:
        self.units_seen += 1
        # D-24: a speaker label must NEVER reach the translator. Recorded so the test can
        # assert it rather than trust it.
        self.labels_seen.append(unit.speaker_label)
        sentences = []
        for s in unit.sentences:
            en = self._pool[self._at % len(self._pool)]
            self._at += 1
            sentences.append({"sent_id": s.sent_id, "en_text": en})
        return {
            "alignment": "sentence",
            "sentences": sentences,
            "metadata": {
                "model": "stub/rehearsal",
                "prompt": {"name": "stub", "sha256": "0" * 64},
                "attempts": 1,
            },
        }


def _lay_down_episode(root: Path) -> None:
    """The episode as diarization leaves it: SPEAKER_NN labels, text + segments, no English."""
    from podcast_scraper.providers.ml.diarization.formatting import (
        format_diarized_screenplay_with_offsets,
    )

    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(_ES.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        label, _, body = line.partition(": ")
        rows.append(
            {
                "id": i,
                "start": float(i * 14),
                "end": float((i + 1) * 14),
                "speaker": label,
                "speaker_label": label,
                "text": body,
            }
        )
    text, offsets = format_diarized_screenplay_with_offsets(rows)
    # Carry the frozen voice id onto the offset rows: the formatter emits `speaker_label` only,
    # and the naming stage keys the source re-render on `speaker`.
    by_label = {r["speaker_label"]: r["speaker"] for r in rows}
    for row in offsets:
        row["speaker"] = by_label.get(row.get("speaker_label"), row.get("speaker_label"))

    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    (root / REL).write_text(text, encoding="utf-8")
    seg = root / "transcripts" / "01 - p10_e01.segments.json"
    seg.write_text(json.dumps(offsets, indent=2, ensure_ascii=False), encoding="utf-8")


def _cfg(root: Path) -> config.Config:
    return config.Config(
        rss="https://example.com/p10_spanish.xml",
        output_dir=str(root),
        language="es",
        speaker_resolution_llm=False,
        translate_api_base="http://translator.invalid:8005/v1",
        translate_model="stub/rehearsal",
        translate_verify_served_model=False,
        save_adfree_transcript=False,
    )


@pytest.fixture
def episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Run the chain once; every test below reads the result."""
    from podcast_scraper.translation import factory as tfactory
    from podcast_scraper.workflow import translation_stage

    _lay_down_episode(tmp_path)
    stub = _StubTranslator()
    monkeypatch.setattr(tfactory, "create_translation_provider", lambda *a, **k: stub)
    monkeypatch.setattr(
        translation_stage, "create_translation_provider", lambda *a, **k: stub, raising=False
    )

    cfg = _cfg(tmp_path)
    translation = translation_stage.run_translation_stage(
        cfg,
        feed_language="es-ES",
        transcript_relpath=REL,
        effective_output_dir=str(tmp_path),
        episode_id="p10_e01",
        feed_id="p10",
        run_id="rehearsal",
        episode_title="Construyendo Senderos Que Duran",
    )
    return {
        "root": tmp_path,
        "cfg": cfg,
        "stub": stub,
        "translation": translation,
    }


class TestTranslationProducedTheEnglishSet:
    def test_the_stage_reports_translated(self, episode: Dict[str, Any]) -> None:
        got = episode["translation"]
        assert got.status == "translated", f"{got.status}: {got.reason}"

    def test_every_unit_succeeded(self, episode: Dict[str, Any]) -> None:
        got = episode["translation"]
        assert got.units > 0
        assert got.units_failed == 0

    def test_the_whole_english_set_is_on_disk(self, episode: Dict[str, Any]) -> None:
        """The §5.3 completeness gate keys on PRESENCE, so the set is the gate."""
        root: Path = episode["root"]
        for rel in (
            "transcripts/01 - p10_e01.txt",
            "transcripts/01 - p10_e01.segments.json",
            "transcripts/01 - p10_e01.adfree.txt",
            "transcripts/01 - p10_e01.translation.json",
        ):
            assert (root / rel).is_file(), f"missing {rel}"

    def test_the_analysis_gate_is_OPEN(self, episode: Dict[str, Any]) -> None:
        """What summary/GI/KG actually consult. A complete set must not block."""
        from podcast_scraper.workflow.translation_stage import analysis_blocked_reason

        assert (
            analysis_blocked_reason(
                episode["cfg"],
                transcript_relpath=REL,
                effective_output_dir=str(episode["root"]),
                feed_language="es-ES",
            )
            is None
        )


class TestTheLabelsNeverReachedTheTranslator:
    """D-24. A label in a unit payload would have the model renaming the same person
    inconsistently between units."""

    def test_no_unit_carried_a_label_in_its_payload(self, episode: Dict[str, Any]) -> None:
        from podcast_scraper.providers.ml.diarization.turns import build_turns
        from podcast_scraper.translation.units import pack_units

        root: Path = episode["root"]
        text = (root / REL).read_text(encoding="utf-8")
        segs = json.loads(
            (root / "transcripts" / "01 - p10_e01.segments.json").read_text(encoding="utf-8")
        )
        turns = build_turns(segs, screenplay_text=text)
        units = pack_units(
            [t.to_dict() for t in turns.turns],
            screenplay_text=text,
            source_language="es",
            max_input_tokens=2048,
            overhead_tokens=0,
            count_tokens=None,
        )
        for u in units:
            assert "SPEAKER_" not in u.source_text, u.source_text[:80]

    def test_the_english_render_still_carries_the_labels(self, episode: Dict[str, Any]) -> None:
        """Carried onto the line, not through the model — and at the moment translation
        finishes they are still the ANONYMOUS ids, which is the whole hinge of D-34.

        Read straight off the file. This used to need a snapshot taken BETWEEN translation and
        naming, because the naming stage re-rendered it — that stage was a reordering of the generic
        pipeline made for this feature and was reverted 2026-10-02, so nothing rewrites the English
        render and the property is observable at the end of the chain.
        """
        en = json.loads(
            (episode["root"] / "transcripts" / "01 - p10_e01.segments.json").read_text(
                encoding="utf-8"
            )
        )
        labels = {str(r.get("speaker_label")) for r in en}
        assert labels and all(lab.startswith("SPEAKER_") for lab in labels), sorted(labels)


# `TestNamingRanOnTheEnglishRender` lived here and is gone (2026-10-02): the naming stage it
# exercised was a reordering of the generic pipeline made for this feature, reverted on the
# operator's instruction. A translated episode therefore keeps anonymous voice ids, which the
# class above now asserts directly.


class TestAdsAreFoundOnTheEnglishRender:
    """The ordering S2.7 exists for: `_AD_PATTERNS` are English regexes, so the Spanish source
    yields nothing and the English render yields the sponsor reads. ADR-157 measured 2 against 0
    on this very episode."""

    @staticmethod
    def _hits(text: str) -> int:
        from podcast_scraper.gi.filters import _AD_PATTERNS

        return sum(1 for p in _AD_PATTERNS if p.search(text))

    def test_the_english_render_is_ad_detectable(self, episode: Dict[str, Any]) -> None:
        root: Path = episode["root"]
        en = (root / "transcripts" / "01 - p10_e01.txt").read_text(encoding="utf-8")
        assert self._hits(en) > 0, "no ad pattern matched the English render"

    def test_the_spanish_source_is_NOT(self, episode: Dict[str, Any]) -> None:
        """The control. Without it, "ads were found" proves nothing about the ordering.

        Reads the TAGGED source (D-44). `REL` is the canonical path, which after the swap holds the
        English render — reading it here would compare English against English and the control would
        pass for the wrong reason, which is worse than failing.
        """
        src = (episode["root"] / "transcripts" / "01 - p10_e01.es.txt").read_text(encoding="utf-8")
        assert self._hits(src) == 0, "an English ad pattern matched Spanish text"

    def test_the_source_has_no_adfree_base(self, episode: Dict[str, Any]) -> None:
        """S2.7: building one would assert ads were removed when the patterns could not see them.

        Under D-44 `<base>.adfree.txt` IS the English analysis base and must exist — this test is
        about the SOURCE, whose ad-free base would be `<base>.es.adfree.txt` and must not.
        """
        tr = episode["root"] / "transcripts"
        assert (tr / "01 - p10_e01.adfree.txt").is_file(), "the English analysis base is missing"
        assert not (tr / "01 - p10_e01.es.adfree.txt").exists()


class TestBothLayersReachTheIndex:
    """RFC-124 §6.2. Without the source layer, a query in the episode's own language matches
    nothing; without the English layer being labelled `en`, it loses its embedding."""

    @pytest.fixture
    def rows(self, episode: Dict[str, Any]) -> List[Dict[str, Any]]:
        from podcast_scraper.search.indexer import _collect_docs_for_episode

        root: Path = episode["root"]
        doc = {
            "episode": {"episode_id": "p10_e01", "title": "es", "language": "es"},
            "feed": {"feed_id": "p10", "title": "Sesiones de Sendero", "language": "es-ES"},
            "content": {"transcript_file_path": REL},
        }
        meta = root / "p10_e01.metadata.json"
        meta.write_text(json.dumps(doc), encoding="utf-8")
        collected = _collect_docs_for_episode(
            root,
            meta,
            doc,
            target_tokens=120,
            overlap_tokens=20,
            metadata_relative_path="p10_e01.metadata.json",
        )
        return [m for _i, _t, m in collected if m.get("doc_type") == "transcript"]

    def test_both_languages_are_present(self, rows: List[Dict[str, Any]]) -> None:
        assert {r.get("language") for r in rows} == {"en", "es"}

    def test_the_english_layer_is_the_analysis_one(self, rows: List[Dict[str, Any]]) -> None:
        from podcast_scraper.search.indexer import is_source_layer_chunk

        english = [r for r in rows if r.get("language") == "en"]
        assert english and not any(is_source_layer_chunk(r) for r in english)

    def test_the_source_layer_is_marked(self, rows: List[Dict[str, Any]]) -> None:
        from podcast_scraper.search.indexer import is_source_layer_chunk

        spanish = [r for r in rows if r.get("language") == "es"]
        assert spanish and all(is_source_layer_chunk(r) for r in spanish)

    def test_the_english_chunks_keep_their_EMBEDDING(self, rows: List[Dict[str, Any]]) -> None:
        """The defect this caught once: labelled `es`, they were routed to the vector-less tier
        and the episode lost semantic search entirely."""
        from podcast_scraper.search.backend import SegmentDocument
        from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend

        docs = [
            SegmentDocument(
                id=f"c{i}",
                text="t",
                show_id="p10",
                episode_id="p10_e01",
                start_time=0.0,
                end_time=1.0,
                language=r.get("language"),
                embedding=[0.1] * 384,
            )
            for i, r in enumerate(rows)
        ]
        english, non_english = LanceDBBackend._split_segments_by_language(docs)
        assert english, "no chunk reached the vector-bearing tier"
        assert non_english, "no chunk reached the keyword-only tier"


class TestTheLedgerIsHonest:
    def test_it_records_the_source_language_and_its_provenance(
        self, episode: Dict[str, Any]
    ) -> None:
        got = episode["translation"]
        assert got.source_language == "es"
        assert got.language_source == "rss", "the feed's tag, not the profile default"

    def test_the_translated_title_is_recorded(self, episode: Dict[str, Any]) -> None:
        """D-42: the EPISODE title goes through the translator so the roster reads it in
        English; the SHOW name does not."""
        doc = json.loads(
            (episode["root"] / "transcripts" / "01 - p10_e01.translation.json").read_text(
                encoding="utf-8"
            )
        )
        assert doc.get("title_en"), "no title_en in the ledger"

    def test_every_unit_carries_its_model_and_prompt(self, episode: Dict[str, Any]) -> None:
        """So a claim can never be attributed to a model that did not produce its text."""
        doc = json.loads(
            (episode["root"] / "transcripts" / "01 - p10_e01.translation.json").read_text(
                encoding="utf-8"
            )
        )
        units = doc["units"]
        assert units
        for u in units:
            assert u["model"] == "stub/rehearsal"
            assert u["prompt_sha256"] == "0" * 64


class TestTheAnalysisBaseFailingDoesNotTakeTheEpisODEWithIt:
    """The branch that had NO test, which is how a function that cannot run stayed green.

    `_withdraw_english_render` imported `english_artifact_relpaths` — deleted by D-44 — so it
    raised ImportError the instant it was called, which was whenever `write_analysis_base`
    returned None for a translated episode. The tests that had covered it were deleted in the
    same change (their note correctly says the atomic swap leaves no partial set to withdraw),
    and nothing replaced them, so the gate was green over a crashing path for a day.

    WHAT THIS ASSERTS IS THE CONTRACT, NOT THE OLD MECHANISM: the stage records the failure and
    leaves the disk alone. Deleting the canonical body — which is what the withdrawal did before
    the inversion, when the canonical body was the SOURCE — would now destroy the only copy of
    the English text and leave the episode with no body at all.
    """

    @pytest.fixture
    def failed_base(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
        from podcast_scraper.translation import artifacts as tartifacts, factory as tfactory
        from podcast_scraper.workflow import translation_stage

        _lay_down_episode(tmp_path)
        stub = _StubTranslator()
        monkeypatch.setattr(tfactory, "create_translation_provider", lambda *a, **k: stub)
        monkeypatch.setattr(
            translation_stage, "create_translation_provider", lambda *a, **k: stub, raising=False
        )
        # `write_analysis_base` returns None for an unreadable or empty English body — not for
        # "no ads found", which produces a valid identity base. Forced here rather than
        # constructed, because the states that produce it naturally (an empty segments list after
        # a successful swap) are themselves invariant violations.
        #
        # PATCHED ON THE DEFINING MODULE, not on `translation_stage`. The stage imports it inside
        # the function body, so the name is looked up on `translation.artifacts` at call time and
        # a module-attribute patch on the stage is simply ignored — which is how the first version
        # of this fixture "passed" three assertions while never taking the branch at all.
        assert hasattr(tartifacts, "write_analysis_base"), "the patch target moved"
        monkeypatch.setattr(tartifacts, "write_analysis_base", lambda *a, **k: None)

        translation = translation_stage.run_translation_stage(
            _cfg(tmp_path),
            feed_language="es-ES",
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path),
            episode_id="p10_e01",
            feed_id="p10",
            run_id="rehearsal-no-base",
            episode_title="Construyendo Senderos Que Duran",
        )
        return {"root": tmp_path, "translation": translation}

    def test_the_stage_does_not_raise(self, failed_base: Dict[str, Any]) -> None:
        """The whole point. Reaching this assertion at all is the regression test — the fixture
        raised ImportError before the dead withdrawal was removed."""
        assert failed_base["translation"] is not None

    def test_it_records_failed_rather_than_claiming_success(
        self, failed_base: Dict[str, Any]
    ) -> None:
        """`TranslationDocument.status` is derived from unit outcomes, so with zero failed units
        it reads `translated` — which is why the ledger needs the explicit flag. Without it the
        API reports success for an episode nothing could analyse."""
        assert failed_base["translation"].status == "failed"
        assert failed_base["translation"].english_ready is False

    def test_the_canonical_english_body_is_STILL_ON_DISK(self, failed_base: Dict[str, Any]) -> None:
        """The inversion's teeth. Before D-44 the canonical path held the SOURCE, so deleting the
        English files was recoverable. It now holds the TRANSLATION, and the source lives at its
        tagged name — so a withdrawal would delete the English text and leave the episode with a
        canonical path that does not exist."""
        body = failed_base["root"] / REL
        assert body.exists(), "the canonical body was deleted — that is the bug, inverted"
        assert body.read_text(encoding="utf-8").strip(), "the canonical body is empty"

    def test_the_tagged_source_is_still_on_disk_too(self, failed_base: Dict[str, Any]) -> None:
        """Both halves of the swap survive, so the episode is repairable by re-running the stage
        rather than by re-transcribing."""
        base = (failed_base["root"] / REL).with_suffix("")
        assert base.with_suffix(".es.txt").exists(), "the tagged Spanish source is gone"

    def test_the_ledger_says_the_set_is_not_consumable(self, failed_base: Dict[str, Any]) -> None:
        import json

        ledger = json.loads(
            (failed_base["root"] / "transcripts" / "p10_e01.translation.json").read_text(
                encoding="utf-8"
            )
            if (failed_base["root"] / "transcripts" / "p10_e01.translation.json").exists()
            else (
                (failed_base["root"] / REL)
                .with_suffix("")
                .with_suffix(".translation.json")
                .read_text(encoding="utf-8")
            )
        )
        assert ledger["english_withdrawn"] is True, (
            "the legacy-named flag is what drives `status` to failed; without it the ledger "
            "contradicts the disk"
        )
        assert ledger["units_failed"] == 0, (
            "no unit failed — this is not a model failure, and an operator re-running the "
            "translator would be chasing the wrong thing"
        )
