"""``pipeline_stage=retranscript_only``: repair a transcript whose speakers we deleted.

The defect this stage exists for: a WebVTT cue can name its speaker with ``<v Speaker 3>``, and the
cue parser stripped that as an HTML tag. A transcript that labelled every single turn landed on disk
as ONE undifferentiated voice, so the episode could never be attributed to anyone. On the production
corpus every episode ingested from a publisher transcript ended with a single voice — 128 of 128.

`29e117a1` fixed the parser for NEW ingests. This stage repairs what is ALREADY STORED, without
audio, ASR or GPU: re-fetch, re-parse, overwrite, relabel.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper import config
from podcast_scraper.workflow import episode_processor as ep

VTT_WITH_VOICES = """WEBVTT

00:00:00.000 --> 00:00:04.000
<v Joe Wiesenthal>Hello and welcome to Odd Lots, I'm Joe Wiesenthal.

00:00:04.000 --> 00:00:08.000
<v Tracy Alloway>And I'm Tracy Alloway.
"""

VTT_WITHOUT_VOICES = """WEBVTT

00:00:00.000 --> 00:00:04.000
Hello and welcome to Odd Lots.
"""


def _job(tmp_path: Path, urls, idx=1):
    txt = tmp_path / "0001 - Episode.txt"
    txt.write_text("old text\n", encoding="utf-8")
    (tmp_path / "0001 - Episode.segments.json").write_text(
        json.dumps([{"start": 0.0, "end": 4.0, "text": "old text"}]), encoding="utf-8"
    )
    episode = SimpleNamespace(
        idx=idx, on_disk_transcript=str(txt), on_disk_transcript_urls=urls, title="Episode"
    )
    return SimpleNamespace(idx=idx, episode=episode), txt


class TestTheStageIsReachableAtAll:
    """It is no use having the repair if no operator can ask for it."""

    def test_the_stage_validates_and_never_demands_an_asr_credential(self, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        cfg = config.Config.model_validate(
            {"rss_url": "https://example.com/f.xml", "pipeline_stage": "retranscript_only"}
        )
        assert cfg.pipeline_stage == "retranscript_only"
        # transcribe_missing stays TRUE purely as ROUTING: it is how the episode reaches the
        # transcription stage where the reprocess dispatch intercepts it.
        assert cfg.transcribe_missing is True
        assert "retranscript_only" in config.STAGES_THAT_NEVER_TRANSCRIBE

    def test_the_dispatcher_routes_the_stage_to_the_refetch(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            ep,
            "_refetch_and_reparse_transcript",
            lambda *a, **k: (seen.update(called=True), (True, "p", 1))[1],
        )
        cfg = SimpleNamespace(pipeline_stage="retranscript_only")
        out = ep._maybe_dispatch_reprocess_stage(object(), cfg, None, "/out", None, None)
        assert seen == {"called": True}
        assert out == (True, "p", 1)


class TestItRefusesRatherThanPretending:
    """A no-op that looks like success is worse than a refusal, because the operator moves on."""

    def test_an_episode_with_no_stored_transcript_url_is_reported_not_guessed(
        self, tmp_path, monkeypatch
    ):
        job, txt = _job(tmp_path, [])
        monkeypatch.setattr(
            ep,
            "_fetch_transcript_content",
            lambda u, c: pytest.fail("nothing to fetch — there is no URL"),
        )
        assert ep._refetch_and_reparse_transcript(
            job, SimpleNamespace(), None, str(tmp_path), None, None
        ) == (False, None, 0)
        assert txt.read_text(encoding="utf-8") == "old text\n"

    def test_a_failed_fetch_leaves_the_episode_alone(self, tmp_path, monkeypatch):
        job, txt = _job(tmp_path, [{"url": "https://ex.com/t.vtt", "type": "text/vtt"}])
        monkeypatch.setattr(ep, "_fetch_transcript_content", lambda u, c: None)
        assert ep._refetch_and_reparse_transcript(
            job, SimpleNamespace(), None, str(tmp_path), None, None
        ) == (False, None, 0)
        assert txt.read_text(encoding="utf-8") == "old text\n"
