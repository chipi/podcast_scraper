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
    from podcast_scraper.workflow import naming_stage, translation_stage

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
    # Snapshot the English labels BEFORE naming re-renders them. This is the state D-34 depends
    # on — anonymous voice ids carried verbatim onto the English line (D-24) — and it exists only
    # BETWEEN the two stages, so a test reading the file afterwards sees names instead.
    en_before = json.loads(
        (tmp_path / "transcripts" / "01 - p10_e01.en.segments.json").read_text(encoding="utf-8")
    )

    naming = naming_stage.run_naming_stage(
        cfg,
        transcript_relpath=REL,
        effective_output_dir=str(tmp_path),
        # THE HOST COMES FROM THE FEED, not the transcript. The English text never says "I'm
        # Maya" — she is named by `<itunes:author>Maya Koster</itunes:author>`, which
        # `detect_hosts_from_feed` extracts and the seam passes as `feed_hosts`. Omitting it left
        # the host voice unnamed and made the rehearsal look like a naming defect; the real
        # pipeline supplies this channel, so the rehearsal must too.
        feed_hosts=["Maya Koster"],
        episode_title="Construyendo Senderos Que Duran",
    )
    return {
        "root": tmp_path,
        "cfg": cfg,
        "stub": stub,
        "translation": translation,
        "naming": naming,
        "en_before_naming": en_before,
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
            "transcripts/01 - p10_e01.en.txt",
            "transcripts/01 - p10_e01.en.segments.json",
            "transcripts/01 - p10_e01.en.adfree.txt",
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

        Read from the snapshot taken between the two stages: naming re-renders this file, so by
        the end of the chain it holds names and the property is no longer observable.
        """
        labels = {str(r.get("speaker_label")) for r in episode["en_before_naming"]}
        assert labels and all(lab.startswith("SPEAKER_") for lab in labels), sorted(labels)


class TestNamingRanOnTheEnglishRender:
    def test_it_named_voices(self, episode: Dict[str, Any]) -> None:
        got = episode["naming"]
        assert got.status == "named", f"{got.status}: {got.reason}"
        assert got.renamed, "no voice was named"

    def test_it_found_BOTH_real_people(self, episode: Dict[str, Any]) -> None:
        """The D-34 payoff, and it takes both channels.

        The GUEST comes from the ENGLISH text — "I'm joined by Liam Verbeek" — which the
        Spanish "me acompaña Liam Verbeek" cannot give an English cue regex. The HOST comes from
        the feed's author tag. Asserting only one of them would have hidden that the rehearsal
        was not supplying the host channel at all, which is exactly what happened first.
        """
        names = set(episode["naming"].renamed.values())
        assert any("Liam" in n for n in names), f"guest not named: {sorted(names)}"
        assert any("Maya" in n for n in names), f"host not named: {sorted(names)}"

    def test_both_bodies_were_re_rendered_named(self, episode: Dict[str, Any]) -> None:
        """Both the source and the English body carry the resolved names for every PERSON."""
        root: Path = episode["root"]
        named = set(episode["naming"].renamed.values())
        for rel, which in (
            (REL, "source"),
            ("transcripts/01 - p10_e01.en.txt", "english"),
        ):
            body = (root / rel).read_text(encoding="utf-8")
            for name in named:
                assert f"{name}:" in body, f"{which} is missing the label {name!r}"

    def test_the_AD_voice_stays_anonymous(self, episode: Dict[str, Any]) -> None:
        """And that is correct, not a gap.

        This episode has three diarized voices: two people and a sponsor read. The roster types
        the third as `commercial` and never gives it a name, because it is not a person — naming
        it would mint a phantom into the roster and then the KG. So one `SPEAKER_NN` legitimately
        survives the re-render, and an assertion that NO anonymous label remains is wrong. This
        test exists because that is exactly the assertion I wrote first.
        """
        import re as _re

        root: Path = episode["root"]
        src = (root / REL).read_text(encoding="utf-8")
        remaining = sorted(set(_re.findall(r"^(SPEAKER_\d+):", src, _re.MULTILINE)))
        assert len(remaining) == 1, f"expected only the ad voice to remain, got {remaining}"
        # It is the ad voice: its turn is the sponsor read.
        ad_line = next(line for line in src.splitlines() if line.startswith(f"{remaining[0]}:"))
        assert "Stripe" in ad_line or "patrocinio" in ad_line, ad_line[:100]

    def test_the_source_kept_its_SPANISH_text(self, episode: Dict[str, Any]) -> None:
        """Only the label column moves. If the Spanish text changed, this is not the same
        episode any more."""
        src = (episode["root"] / REL).read_text(encoding="utf-8")
        assert "Bienvenidos de nuevo" in src
        assert "construcción de senderos" in src

    def test_the_offsets_still_describe_their_bodies(self, episode: Dict[str, Any]) -> None:
        """The re-render rewrites both bodies and both sidecars. If they disagree, every GI
        quote in the episode points at the wrong characters."""
        root: Path = episode["root"]
        for body_rel, seg_rel in (
            (REL, "transcripts/01 - p10_e01.segments.json"),
            ("transcripts/01 - p10_e01.en.txt", "transcripts/01 - p10_e01.en.segments.json"),
        ):
            text = (root / body_rel).read_text(encoding="utf-8")
            for row in json.loads((root / seg_rel).read_text(encoding="utf-8")):
                cs, ce = int(row["char_start"]), int(row["char_end"])
                assert text[cs:ce] == row["text"], f"{seg_rel} disagrees with {body_rel}"


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
        en = (root / "transcripts" / "01 - p10_e01.en.txt").read_text(encoding="utf-8")
        assert self._hits(en) > 0, "no ad pattern matched the English render"

    def test_the_spanish_source_is_NOT(self, episode: Dict[str, Any]) -> None:
        """The control. Without it, "ads were found" proves nothing about the ordering."""
        src = (episode["root"] / REL).read_text(encoding="utf-8")
        assert self._hits(src) == 0, "an English ad pattern matched Spanish text"

    def test_the_source_has_no_adfree_base(self, episode: Dict[str, Any]) -> None:
        """S2.7: building one would assert ads were removed when the patterns could not see
        them. The analysis base for a translated episode is `.en.adfree.txt`."""
        assert not (episode["root"] / "transcripts" / "01 - p10_e01.adfree.txt").exists()


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
