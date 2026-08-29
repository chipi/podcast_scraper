"""#1873 — every signal names its run, feed, episode and profile.

When an episode fails, reconstructing HOW it was running used to be archaeology: find the
run in the jobs registry, read its argv, assume the corpus config was not edited since. With
three profile layers (corpus < feed pin < request override) that assumption is unsound — one
corpus can produce two runs with different ASR routing on the same day. Each surface below
must therefore carry the context itself, and these tests exist so a future surface cannot
quietly drop back to partial attribution.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from podcast_scraper.obs.events import (
    clear_run_context,
    emit_event,
    run_context_from_config,
    set_run_context,
)
from podcast_scraper.utils import correlation

pytestmark = [pytest.mark.unit]

RUN = "run-abc123"
FEED = "https://acast.example/feed.xml"
EPISODE = "ep-42"
PROFILE = "cloud_with_dgx_primary"


@pytest.fixture()
def wired(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy-for-validation")
    from podcast_scraper import config as config_mod

    cfg = config_mod.Config(rss_url=FEED, profile=PROFILE)
    clear_run_context()
    correlation.set_run_id(RUN)
    correlation.set_episode_id(EPISODE)
    correlation.set_feed_id(FEED)
    correlation.set_profile(PROFILE)
    set_run_context(**run_context_from_config(cfg))
    yield cfg
    clear_run_context()
    correlation.set_run_id(None)
    correlation.set_episode_id(None)
    correlation.set_feed_id(None)
    correlation.set_profile(None)


class TestLogLines:
    def test_a_log_line_names_all_four(self, wired):
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(
            correlation.CorrelationFormatter(
                "[run=%(run_id)s feed=%(feed_id)s profile=%(profile)s "
                "ep=%(episode_id)s] %(message)s"
            )
        )
        log = logging.getLogger("test_1873_logs")
        log.handlers = [handler]
        log.setLevel(logging.INFO)
        log.propagate = False
        log.info("transcription failed")

        line = buf.getvalue()
        assert RUN in line and FEED in line and PROFILE in line and EPISODE in line, line

    def test_absent_context_renders_placeholders_not_errors(self):
        correlation.set_run_id(None)
        correlation.set_feed_id(None)
        correlation.set_profile(None)
        correlation.set_episode_id(None)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(
            correlation.CorrelationFormatter("[%(run_id)s %(feed_id)s %(profile)s] %(message)s")
        )
        log = logging.getLogger("test_1873_empty")
        log.handlers = [handler]
        log.setLevel(logging.INFO)
        log.propagate = False
        log.info("hello")
        assert "[- - -] hello" in buf.getvalue()


class TestEvents:
    def test_event_carries_all_four_plus_routing(self, wired):
        record = json.loads(emit_event("llm_cost", provider="litellm"))
        assert record["profile"] == PROFILE
        assert record["episode_id"] == EPISODE
        assert record["feed_id"] == FEED
        assert record["asr_provider"] == "tailnet_dgx_whisper"
        assert record["diarization_provider"] == "tailnet_dgx"

    def test_explicit_fields_still_win(self, wired):
        record = json.loads(emit_event("llm_cost", episode_id="other-ep"))
        assert record["episode_id"] == "other-ep"


class TestErrors:
    def test_sentry_tags_include_feed_and_profile(self, wired, monkeypatch):
        tags: dict[str, str] = {}

        class _FakeSentry:
            @staticmethod
            def set_tag(k, v):
                tags[k] = v

        monkeypatch.setitem(__import__("sys").modules, "sentry_sdk", _FakeSentry)
        from podcast_scraper.utils.sentry_init import set_run_tag

        set_run_tag(RUN, EPISODE)
        assert tags.get("run_id") == RUN
        assert tags.get("episode_id") == EPISODE
        assert tags.get("feed_id") == FEED, "a GlitchTip issue cannot say which feed produced it"
        assert tags.get("profile") == PROFILE, "…nor which profile"
