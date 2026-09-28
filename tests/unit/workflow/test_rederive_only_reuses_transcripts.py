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
        dry_run=False,
        generate_summaries=True,
        single_feed_uses_corpus_layout=False,
        require_transcript_speakers=False,
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


class TestTheDirectDownloadRouteAlsoReDerives:
    """The OTHER download route. Everything above tests an episode with no transcript URL.

    THE GAP THIS CLOSES (prod, 2026-09-28). ``process_episode_download`` has the reuse branch the
    tests above cover — but it calls ``process_transcript_download`` FIRST for any episode whose
    publisher serves a transcript, and returns that result directly. So for a direct-download feed
    the reuse branch is never reached, and ``process_transcript_download`` had no equivalent: it
    called ``_check_existing_transcript`` (which correctly finds the episode CORPUS-WIDE, so →
    skip), then looked for the transcript RUN-LOCALLY in ``effective_output_dir``. Under
    ``--single-feed-uses-corpus-layout`` every run gets a fresh run dir while the transcript lives
    in a prior one, so the glob missed and it returned ``(False, None, None, 0)``.

    Measured: a feed-scoped ``rederive_only`` over 50 Odd Lots episodes re-derived 2 and skipped
    48. The 2 were the only ones with no transcript URL — they reached the sibling branch. 48 stale
    KGs survived a run that exited 0, 19 of them carrying a misspelled duplicate of the host.

    The fresh-run-dir detail is load-bearing: pass the PRIOR run dir as ``effective_output_dir``
    and the old run-local glob accidentally passes.
    """

    @staticmethod
    def _episode_with_transcript_url(guid: str = "guid-1"):
        ep_obj = _episode(guid)
        ep_obj.transcript_urls = [
            SimpleNamespace(url="https://example.com/ep1.vtt", type="text/vtt")
        ]
        return ep_obj

    def test_a_published_transcript_episode_is_re_derived_not_skipped(
        self, corpus, tmp_path, monkeypatch
    ):
        """The regression: returns success + the on-disk transcript, without downloading."""
        monkeypatch.setattr(
            ep.run_index, "corpus_root_from_cfg", lambda cfg: str(corpus.root), raising=False
        )

        def _no_network(*a, **k):  # pragma: no cover — must never be reached
            raise AssertionError("rederive_only must not fetch the publisher transcript")

        monkeypatch.setattr(ep, "_fetch_transcript_content", _no_network)

        fresh_run = corpus.feed / "run_20260928-133000"
        (fresh_run / "transcripts").mkdir(parents=True)

        success, path, source, downloaded = ep.process_transcript_download(
            self._episode_with_transcript_url(),
            "https://example.com/ep1.vtt",
            "text/vtt",
            _cfg(corpus, generate_summaries=True, single_feed_uses_corpus_layout=True),
            str(fresh_run),
            "20260928-133000",
        )
        assert success is True, "the episode must be re-derived, not skipped"
        assert path is not None and str(path).endswith((".txt", ".vtt", ".srt"))
        assert source in ("direct_download", "whisper_transcription")
        assert downloaded == 0, "no bytes: the transcript came off disk"

    def test_a_missing_transcript_is_a_loud_failure_not_a_quiet_success(
        self, corpus, tmp_path, monkeypatch
    ):
        """Symmetry with the sibling route — nothing to re-derive must not look like success."""
        monkeypatch.setattr(
            ep.run_index, "corpus_root_from_cfg", lambda cfg: str(corpus.root), raising=False
        )
        monkeypatch.setattr(ep, "_check_existing_transcript", lambda *a, **k: True)
        monkeypatch.setattr(
            ep, "_resolve_existing_transcript_for_rederive", lambda *a, **k: (None, None)
        )
        fresh_run = corpus.feed / "run_20260928-134000"
        (fresh_run / "transcripts").mkdir(parents=True)
        success, path, source, _b = ep.process_transcript_download(
            self._episode_with_transcript_url(),
            "https://example.com/ep1.vtt",
            "text/vtt",
            _cfg(corpus, generate_summaries=True, single_feed_uses_corpus_layout=True),
            str(fresh_run),
            "20260928-134000",
        )
        assert success is False and path is None and source is None

    def test_a_non_rederive_stage_keeps_its_old_skip_behaviour(self, corpus, tmp_path, monkeypatch):
        """The fix must be scoped: a normal run still skips an already-present episode."""
        monkeypatch.setattr(
            ep.run_index, "corpus_root_from_cfg", lambda cfg: str(corpus.root), raising=False
        )
        monkeypatch.setattr(ep, "_check_existing_transcript", lambda *a, **k: True)
        fresh_run = corpus.feed / "run_20260928-135000"
        (fresh_run / "transcripts").mkdir(parents=True)
        success, _p, _s, _b = ep.process_transcript_download(
            self._episode_with_transcript_url(),
            "https://example.com/ep1.vtt",
            "text/vtt",
            _cfg(corpus, pipeline_stage=None, generate_summaries=True),
            str(fresh_run),
            "20260928-135000",
        )
        assert success is False, "a full run must still skip a present episode run-locally"
