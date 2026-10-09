"""`pipeline_stage=translate_only` — RFC-124 §5.2's retry, which did not exist.

The RFC names this mode in three places as the recovery for a `translation_pending` episode and
for a partial failure, and rests a rejected alternative on it ("a per-episode
`translation_pending` queue … `translate_only` reprocess covers retries"). Nothing implemented
it, so every one of those paths was unactionable and recovery meant deleting `.en.*` by hand.

It is deliberately thin, because the work already exists in order: discard the English render,
then take the relabel path, which loads the on-disk transcript and its frozen `SPEAKER_NN`
diarization — and for a non-English episode naming declines (the S2.14 English-only guard), so the
re-rendered source carries anonymous labels again, which is exactly what the translator must be
given. The seam then re-translates and cascades GI/KG. (Naming from the fresh English render was
D-34, reverted 2026-10-02 — #2234 — so a non-English episode stays anonymous.)

TWO OPERATIONS WEAR THIS NAME, and conflating them is the trap:

* a RETRY keeps the content-keyed memory, so the units that already succeeded cost no GPU — that
  is the whole reason the memory is keyed by content;
* a RE-TRANSLATION after a MODEL change needs it GONE, because the memory is deliberately not
  model-keyed (D-33: keying on the model would make editing a config string trigger a
  corpus-wide GPU spend on the next relabel). Without the purge, a re-translation after an
  upgrade reuses every cached unit and changes nothing.

`--fresh-translation` / `translation_discard_memory` is the second one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from podcast_scraper import config

pytestmark = pytest.mark.unit


class TestTheStageExists:
    def test_the_config_accepts_it(self) -> None:
        cfg = config.Config(rss="https://e.com/f.xml", pipeline_stage="translate_only")
        assert cfg.pipeline_stage == "translate_only"

    def test_it_NEVER_transcribes(self) -> None:
        """So it cannot demand an ASR credential it will never use — the stated reason that
        frozenset exists."""
        assert "translate_only" in config.STAGES_THAT_NEVER_TRANSCRIBE

    def test_it_reaches_the_transcription_stage_anyway(self) -> None:
        """The routing trick every reprocess mode uses: `transcribe_missing=true` purely so the
        episode gets far enough for the dispatch to intercept it and load from disk."""
        cfg = config.Config(rss="https://e.com/f.xml", pipeline_stage="translate_only")
        assert cfg.transcribe_missing is True

    def test_the_cli_offers_it(self) -> None:
        from pathlib import Path as _P

        src = (_P(__file__).resolve().parents[4] / "src/podcast_scraper/cli.py").read_text(
            encoding="utf-8"
        )
        assert '"translate_only",' in src
        assert "--fresh-translation" in src

    def test_the_server_treats_it_as_a_REPROCESS_mode(self) -> None:
        """Not a partial one: it reuses the on-disk transcript and never re-runs ASR, so the
        job API's scoping rules for reprocessing apply."""
        from podcast_scraper.server.jobs import (
            PIPELINE_STAGES_PARTIAL,
            PIPELINE_STAGES_REPROCESS,
        )

        assert "translate_only" in PIPELINE_STAGES_REPROCESS
        assert "translate_only" not in PIPELINE_STAGES_PARTIAL

    def test_it_counts_as_asking_for_reprocessing(self) -> None:
        """Otherwise `processing.py`'s "you asked to reprocess and nothing was downloadable"
        warning would never fire for this mode."""
        from pathlib import Path as _P

        src = (
            _P(__file__).resolve().parents[4] / "src/podcast_scraper/workflow/stages/processing.py"
        ).read_text(encoding="utf-8")
        assert '"translate_only",' in src


class TestItDownloadsNoAudio:
    def test_it_takes_the_no_download_exit(self, tmp_path: Path) -> None:
        """Its input is the on-disk SOURCE transcript. Downloading audio it will never open would
        make the cheap repair as expensive as the one it exists to avoid.

        Asserted on behaviour, not on the source text: the exit is the shared
        ``audio_route_leaves_before_skip_existing`` predicate, which the early skip-existing check
        (#2290) asks too, so the two cannot disagree about this stage.
        """
        import xml.etree.ElementTree as ET
        from unittest.mock import patch

        from podcast_scraper import models
        from podcast_scraper.workflow import episode_processor

        cfg = config.Config(rss="https://e.com/f.xml", pipeline_stage="translate_only")
        item = ET.Element("item")
        ET.SubElement(item, "guid").text = "g1"
        episode = models.Episode(idx=1, title="Ep", title_safe="Ep", item=item, transcript_urls=[])
        episode.media_url = "https://e.com/ep.mp3"

        with patch.object(
            episode_processor,
            "_download_or_reuse_media",
            side_effect=AssertionError("translate_only must not download audio"),
        ):
            job = episode_processor.download_media_for_transcription(
                episode, cfg, str(tmp_path / "tmp"), str(tmp_path / "out"), None
            )

        assert job is not None and job.temp_media == ""
        assert episode_processor.audio_route_leaves_before_skip_existing(cfg)


class TestTheMemoryFlag:
    def test_it_defaults_to_KEEPING_the_memory(self) -> None:
        """The default is the RETRY, which is what the RFC describes. Discarding by default
        would make every `translation_pending` recovery a full re-translation."""
        cfg = config.Config(rss="https://e.com/f.xml", pipeline_stage="translate_only")
        assert cfg.translation_discard_memory is False

    def test_it_can_be_turned_on(self) -> None:
        cfg = config.Config(
            rss="https://e.com/f.xml",
            pipeline_stage="translate_only",
            translation_discard_memory=True,
        )
        assert cfg.translation_discard_memory is True

    def test_the_memory_is_NOT_touched_by_the_swap_back(self, tmp_path: Path) -> None:
        """Which is why the flag has to exist at all: the swap-back moves BODIES, so nothing in it
        can remove the ledger.

        This used to assert that `english_artifact_relpaths` omitted `translation.json`. That list
        is gone with the withdrawal machinery (D-44) — there is no set of English files to delete,
        because the English body IS the canonical one and the source trades places back with it. So
        the property is now asserted against the real operation.
        """
        from podcast_scraper.workflow.episode_processor import _invalidate_translation

        tr = tmp_path / "transcripts"
        tr.mkdir(parents=True)
        (tr / "01 - ep.txt").write_text("stale translation", encoding="utf-8")
        (tr / "01 - ep.es.txt").write_text("el original", encoding="utf-8")
        ledger = tr / "01 - ep.translation.json"
        ledger.write_text('{"version": "1.0", "units": []}', encoding="utf-8")

        _invalidate_translation("transcripts/01 - ep.txt", str(tmp_path), "es")

        assert ledger.is_file(), "the translation memory was destroyed by the swap-back"
        assert "original" in (tr / "01 - ep.txt").read_text(encoding="utf-8")


class TestWhatItActuallyDOES:
    """Driven through the real stage function, with the relabel hand-off stubbed — the relabel
    path is covered by its own tests, and what is under test here is the discard."""

    @staticmethod
    def _episode(tmp_path: Path) -> Path:
        run = tmp_path / "run_20260101-000000"
        tx = run / "transcripts"
        tx.mkdir(parents=True)
        (tx / "01 - ep.txt").write_text("SPEAKER_00: Hola.\n", encoding="utf-8")
        (tx / "01 - ep.segments.json").write_text("[]", encoding="utf-8")
        for name in (
            "01 - ep.en.txt",
            "01 - ep.en.segments.json",
            "01 - ep.en.adfree.txt",
            "01 - ep.translation.json",
        ):
            (tx / name).write_text("{}", encoding="utf-8")
        return tx / "01 - ep.txt"

    def _run(self, tmp_path: Path, monkeypatch: Any, *, discard: bool) -> Any:
        from podcast_scraper.workflow import episode_processor as ep

        txt = self._episode(tmp_path)
        monkeypatch.setattr(ep, "_existing_transcript_for", lambda *a, **k: txt)
        monkeypatch.setattr(ep, "_relabel_existing_transcript", lambda *a, **k: (True, "rel", 1))
        cfg = config.Config(
            rss="https://e.com/f.xml",
            pipeline_stage="translate_only",
            translation_discard_memory=discard,
        )

        class _Job:
            idx = 1

        return ep._retranslate_existing_transcript(
            cast(Any, _Job()), cfg, None, str(tmp_path), None, None
        )

    def test_it_swaps_the_SOURCE_back_to_the_canonical_path(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """D-44: there is no English set to delete. The stale translation is discarded by the source
        trading places back with it, so the episode keeps a canonical transcript throughout."""
        tx = tmp_path / "run_20260101-000000" / "transcripts"
        self._run(tmp_path, monkeypatch, discard=False)
        assert (tx / "01 - ep.txt").is_file(), "the episode lost its canonical transcript"
        assert not (tx / "01 - ep.es.txt").exists(), "the tagged source should have moved back"
        assert not [q for q in tx.iterdir() if ".tmp" in q.name]

    def test_it_KEEPS_the_source_transcript(self, tmp_path: Path, monkeypatch: Any) -> None:
        """The source is the INPUT to the re-translation. Deleting it would make the mode
        unable to do the one thing it is for."""
        tx = tmp_path / "run_20260101-000000" / "transcripts"
        self._run(tmp_path, monkeypatch, discard=False)
        assert (tx / "01 - ep.txt").read_text(encoding="utf-8") == "SPEAKER_00: Hola.\n"
        assert (tx / "01 - ep.segments.json").exists()

    def test_by_default_it_keeps_the_memory(self, tmp_path: Path, monkeypatch: Any) -> None:
        tx = tmp_path / "run_20260101-000000" / "transcripts"
        self._run(tmp_path, monkeypatch, discard=False)
        assert (tx / "01 - ep.translation.json").exists(), (
            "the retry must keep the memory, or every `translation_pending` recovery becomes a "
            "full re-translation"
        )

    def test_with_the_flag_it_discards_the_memory(self, tmp_path: Path, monkeypatch: Any) -> None:
        tx = tmp_path / "run_20260101-000000" / "transcripts"
        self._run(tmp_path, monkeypatch, discard=True)
        assert not (tx / "01 - ep.translation.json").exists()

    def test_it_hands_off_to_the_relabel_path(self, tmp_path: Path, monkeypatch: Any) -> None:
        """The hand-off is the mode: relabel re-renders the source with ANONYMOUS labels for a
        non-English episode (D-34), which is what the translator must be given."""
        assert self._run(tmp_path, monkeypatch, discard=False) == (True, "rel", 1)

    def test_a_missing_transcript_is_recorded_not_crashed(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from podcast_scraper.workflow import episode_processor as ep

        recorded: list = []
        monkeypatch.setattr(ep, "_existing_transcript_for", lambda *a, **k: None)
        monkeypatch.setattr(ep, "_record_unresolved_transcript", lambda *a, **k: recorded.append(a))
        cfg = config.Config(rss="https://e.com/f.xml", pipeline_stage="translate_only")

        class _Job:
            idx = 1

        assert ep._retranslate_existing_transcript(
            cast(Any, _Job()), cfg, None, str(tmp_path), None, None
        ) == (False, None, 0)
        assert recorded, "an unresolvable episode must be recorded, not silently skipped"


class TestARelabelOfATranslatedEpisodeReadsTheSource:
    """relabel_only on a TRANSLATED episode read `<base>.txt` — under D-44 the English render — and
    named the speakers on it with the source language's vocabulary; then the post-processing
    swap-back moved that English aside and restored the source, discarding the relabel's work.
    Found re-labelling Radio Ambulante (es, 2026-10-09): the LLM proposed "Daniel Alarcón" from the
    English, and the episode ended with 0 named entries. translate_only already swaps back BEFORE
    handing off to the relabel; relabel_only now does the same."""

    @staticmethod
    def _translated_episode(tmp_path: Path) -> Path:
        tx = tmp_path / "run_20260101-000000" / "transcripts"
        tx.mkdir(parents=True)
        # Under D-44 the canonical pair is the ENGLISH render; the source sits at `.es.*`.
        (tx / "01 - ep.txt").write_text("SPEAKER_00: Hello, I'm Ana Ruiz.\n", encoding="utf-8")
        (tx / "01 - ep.es.txt").write_text("SPEAKER_00: Hola, soy Ana Ruiz.\n", encoding="utf-8")
        seg = [{"start": 0.0, "end": 2.0, "speaker_label": "SPEAKER_00", "text": "{}"}]
        for name, text in (
            ("01 - ep", "Hello, I'm Ana Ruiz."),
            ("01 - ep.es", "Hola, soy Ana Ruiz."),
        ):
            rows = [dict(seg[0], text=text)]
            (tx / f"{name}.segments.json").write_text(json.dumps(rows), encoding="utf-8")
        (tx / "01 - ep.translation.json").write_text("{}", encoding="utf-8")
        return tx / "01 - ep.txt"

    def test_the_naming_reads_the_spanish_source(self, tmp_path: Path, monkeypatch: Any) -> None:
        from podcast_scraper.providers.ml.diarization import pipeline as dp
        from podcast_scraper.workflow import episode_processor as ep

        txt = self._translated_episode(tmp_path)
        monkeypatch.setattr(ep, "_existing_transcript_for", lambda *a, **k: txt)
        seen: list = []

        class _Stop(Exception):
            pass

        def fake_apply(result: Any, *a: Any, **k: Any) -> Any:
            seen.append(result)
            raise _Stop

        monkeypatch.setattr(dp, "apply_diarization_to_result", fake_apply)
        cfg = config.Config(
            rss="https://e.com/f.xml", pipeline_stage="relabel_only", feed_declared_language="es"
        )

        class _Job:
            idx = 1
            episode = None
            detected_speaker_names = None
            metadata_named = None
            ep_title = "t"

        with pytest.raises(_Stop):
            ep._relabel_existing_transcript(cast(Any, _Job()), cfg, None, str(tmp_path), None, None)
        assert seen, "the naming step was never reached"
        texts = " ".join(str(s.get("text", "")) for s in seen[0]["segments"])
        assert "soy Ana Ruiz" in texts and "I'm" not in texts
        assert "Hola" in txt.read_text(encoding="utf-8"), "the source is back at the canonical path"
