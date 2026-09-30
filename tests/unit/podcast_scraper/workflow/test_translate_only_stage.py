"""`pipeline_stage=translate_only` — RFC-124 §5.2's retry, which did not exist.

The RFC names this mode in three places as the recovery for a `translation_pending` episode and
for a partial failure, and rests a rejected alternative on it ("a per-episode
`translation_pending` queue … `translate_only` reprocess covers retries"). Nothing implemented
it, so every one of those paths was unactionable and recovery meant deleting `.en.*` by hand.

It is deliberately thin, because the work already exists in order: discard the English render,
then take the relabel path, which loads the on-disk transcript and its frozen `SPEAKER_NN`
diarization — and for a non-English episode that re-naming is DEFERRED (D-34), so the re-rendered
source carries anonymous labels again, which is exactly what the translator must be given. The
seam then re-translates, names from the fresh English render, and cascades GI/KG.

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
    def test_it_takes_the_no_download_exit(self) -> None:
        """Its input is the on-disk SOURCE transcript. Downloading audio it will never open would
        make the cheap repair as expensive as the one it exists to avoid."""
        from pathlib import Path as _P

        src = (
            _P(__file__).resolve().parents[4] / "src/podcast_scraper/workflow/episode_processor.py"
        ).read_text(encoding="utf-8")
        assert (
            'if cfg.pipeline_stage in ("relabel_only", "retranscript_only", "translate_only"):'
            in src
        )


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

    def test_the_memory_is_NOT_in_the_invalidation_list(self) -> None:
        """Which is why the flag has to exist at all: `english_artifact_relpaths` deliberately
        omits `translation.json`, so the default invalidation cannot remove it."""
        from podcast_scraper.translation.artifacts import english_artifact_relpaths

        rels = english_artifact_relpaths("transcripts/01 - ep.txt")
        assert rels
        assert not any("translation.json" in r for r in rels)


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

    def test_it_deletes_the_english_render(self, tmp_path: Path, monkeypatch: Any) -> None:
        tx = tmp_path / "run_20260101-000000" / "transcripts"
        self._run(tmp_path, monkeypatch, discard=False)
        assert not (tx / "01 - ep.en.txt").exists()
        assert not (tx / "01 - ep.en.segments.json").exists()
        assert not (tx / "01 - ep.en.adfree.txt").exists()

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
