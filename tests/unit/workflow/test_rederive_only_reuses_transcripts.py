"""``--pipeline-stage rederive_only`` must actually re-derive, not exit 0 having done nothing.

THE BUG. ``rederive_only`` coerces ``transcribe_missing=false`` — correct, it must never call an
ASR provider. But the only other exit from ``process_episode_download`` was the
``if cfg.transcribe_missing and temp_dir:`` gate, so the function returned
``(False, None, None, 0)``, no ``ProcessingJob`` was queued (the caller requires a non-None
``transcript_source``), and the run reported success having re-derived nothing. It was
documented as broken in ``docs/guides/CORPUS_REPROCESSING.md`` and in the Makefile rather than
fixed, and a documented reprocess recipe was built on it.

Why the two sibling stages do not have this bug: ``relabel_only`` and ``rediarize_only`` set
``transcribe_missing=true`` precisely so the episode REACHES the transcription stage, where
``_maybe_dispatch_reprocess_stage`` intercepts and loads from disk. That route is closed to
rederive_only — ``transcribe_missing=true`` is also what makes the Deepgram-credential validator
demand an ASR key for a stage that calls no ASR. So rederive_only resolves the transcript in
``process_episode_download`` instead.

The tests below pin the three things that make it real work rather than the appearance of it:
a transcript is FOUND, a usable ``transcript_source`` comes back so the cascade queues, and a
missing transcript is a loud failure rather than a quiet success.
"""

from __future__ import annotations

import json
import queue
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow import episode_processor as ep


@pytest.fixture
def corpus(tmp_path):
    """A corpus-layout feed with one already-processed episode."""
    feed = tmp_path / "feeds" / "rss_example.com_abc123"
    run = feed / "run_20260814-055303"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir(parents=True)
    stem = "0001 - An Episode_20260814-055303"
    (run / "metadata" / f"{stem}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"guid": "guid-1", "episode_id": "guid-1", "title": "An Episode"},
                "content": {"transcript_source": "whisper_transcription"},
            }
        ),
        encoding="utf-8",
    )
    txt = run / "transcripts" / f"{stem}.txt"
    txt.write_text("Alice: hello.\nBob: hi.\n", encoding="utf-8")
    return SimpleNamespace(root=tmp_path, feed=feed, run=run, transcript=txt)


def _episode(guid: str = "guid-1"):
    """An episode whose guid is reachable the way the corpus index reads it.

    ``run_index._episode_guid`` resolves the guid from the RSS ``<item>`` element
    (``episode.item.find("guid").text``), NOT from an ``episode.guid`` attribute. A
    SimpleNamespace with ``guid="guid-1"`` looks right and resolves to None, so the whole
    corpus lookup silently returns "not present" — which is how the first draft of this file
    fooled itself into thinking the resolver was broken.
    """
    item = ET.Element("item")
    ET.SubElement(item, "guid").text = guid
    return SimpleNamespace(
        idx=1,
        title="An Episode",
        title_safe="An Episode",
        guid=guid,
        item=item,
        transcript_urls=[],
    )


def _cfg(corpus, **over):
    base = dict(
        pipeline_stage="rederive_only",
        transcribe_missing=False,
        skip_existing=True,
        rss_url="https://example.com/feed.xml",
        output_dir=str(corpus.feed),
        generate_metadata=True,
        prefer_types=[],
        delay_ms=0,
        metadata_format="json",
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestTranscriptIsFound:
    def test_resolver_returns_the_on_disk_transcript(self, corpus, monkeypatch):
        monkeypatch.setattr(
            ep.run_index, "corpus_root_from_cfg", lambda cfg: str(corpus.root), raising=False
        )
        path, source = ep._resolve_existing_transcript_for_rederive(
            _episode(), _cfg(corpus), str(corpus.run), "20260814-055303"
        )
        assert path is not None, "the transcript is on disk and must be found"
        assert Path(path).name.endswith(".txt")
        assert source in ("direct_download", "whisper_transcription")

    def test_source_is_read_from_metadata_not_assumed(self, corpus, monkeypatch):
        """A direct-download feed must not be relabelled as whisper_transcription."""
        meta = next((corpus.run / "metadata").glob("*.metadata.json"))
        d = json.loads(meta.read_text())
        d["content"]["transcript_source"] = "direct_download"
        meta.write_text(json.dumps(d), encoding="utf-8")

        monkeypatch.setattr(
            ep.run_index, "corpus_root_from_cfg", lambda cfg: str(corpus.root), raising=False
        )
        _path, source = ep._resolve_existing_transcript_for_rederive(
            _episode(), _cfg(corpus), str(corpus.run), "20260814-055303"
        )
        assert source == "direct_download"


class TestMetadataMarkerIsRejected:
    def test_metadata_without_a_transcript_is_not_a_transcript(self, corpus, monkeypatch, caplog):
        """``existing_transcript_path_in_corpus`` falls back to the METADATA path as a mere
        presence marker. That is fine for skip-existing, which only asks "was this processed?",
        but handing a ``.metadata.json`` back as a transcript would push an unusable path into
        the cascade — the same silent-success shape this whole stage suffered from.
        """
        corpus.transcript.unlink()
        monkeypatch.setattr(
            ep.run_index, "corpus_root_from_cfg", lambda cfg: str(corpus.root), raising=False
        )
        caplog.set_level("WARNING")
        path, source = ep._resolve_existing_transcript_for_rederive(
            _episode(), _cfg(corpus), str(corpus.run), "20260814-055303"
        )
        assert path is None and source is None
        assert any("no transcript file" in r.getMessage() for r in caplog.records)


class TestProcessEpisodeDownloadQueuesTheCascade:
    """The caller only queues a ProcessingJob when ``transcript_source`` is not None.

    This is the assertion that would have caught the original bug: the old code returned
    ``(False, None, None, 0)`` here, which reads as "no work to do" and exits 0.
    """

    def test_returns_a_queueable_result(self, corpus, monkeypatch):
        monkeypatch.setattr(
            ep,
            "_resolve_existing_transcript_for_rederive",
            lambda *a, **k: (str(corpus.transcript), "whisper_transcription"),
        )
        ok, path, source, nbytes = ep.process_episode_download(
            _episode(),
            _cfg(corpus),
            None,
            str(corpus.run),
            "20260814-055303",
            queue.Queue(),
            None,
        )
        assert ok is True
        assert path == str(corpus.transcript)
        assert source is not None, (
            "transcript_source must be non-None or the caller queues NO ProcessingJob and the "
            "run exits 0 having re-derived nothing — the original bug"
        )
        assert nbytes == 0, "rederive_only must not download anything"

    def test_no_transcript_is_a_loud_failure_not_a_quiet_success(self, corpus, monkeypatch, caplog):
        monkeypatch.setattr(
            ep, "_resolve_existing_transcript_for_rederive", lambda *a, **k: (None, None)
        )
        caplog.set_level("WARNING")
        ok, path, source, _ = ep.process_episode_download(
            _episode(),
            _cfg(corpus),
            None,
            str(corpus.run),
            "20260814-055303",
            queue.Queue(),
            None,
        )
        assert ok is False and path is None and source is None
        assert any("nothing to re-derive" in r.getMessage() for r in caplog.records)

    def test_no_asr_is_enqueued_even_though_temp_dir_exists(self, corpus, monkeypatch, tmp_path):
        """rederive_only must not fall through to the Whisper branch under any circumstance."""
        called = {"n": 0}
        monkeypatch.setattr(
            ep,
            "download_media_for_transcription",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )
        monkeypatch.setattr(
            ep,
            "_resolve_existing_transcript_for_rederive",
            lambda *a, **k: (str(corpus.transcript), "whisper_transcription"),
        )
        ep.process_episode_download(
            _episode(),
            _cfg(corpus, transcribe_missing=True),  # even if someone flips this
            str(tmp_path / "tmp"),
            str(corpus.run),
            "20260814-055303",
            queue.Queue(),
            None,
        )
        assert called["n"] == 0, "rederive_only must never reach the transcription branch"
