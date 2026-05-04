#!/usr/bin/env python3
"""Integration checks: VTT/SRT direct download and GI timing (issue #544)."""

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.gi.grounding import GroundedQuote
from podcast_scraper.gi.pipeline import _artifact_from_multi_insight
from podcast_scraper.transcript_formats import parse_webvtt
from podcast_scraper.workflow import episode_processor

pytestmark = [pytest.mark.integration]

_tests_dir = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "integration_parent_conftest", _tests_dir / "conftest.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("conftest not loadable")
_integration_ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_integration_ct)
create_test_config = _integration_ct.create_test_config
create_test_episode = _integration_ct.create_test_episode


def test_vtt_segments_yield_nonzero_quote_timestamps_in_gi_artifact() -> None:
    """Parsed VTT segments align with plain text so quotes get timestamp_*_ms."""
    vtt = (
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n\n"
        "00:00:01.000 --> 00:00:02.000\n world\n"
    )
    plain, segs = parse_webvtt(vtt)
    gq = GroundedQuote(
        char_start=0,
        char_end=5,
        text="Hello",
        qa_score=0.9,
        nli_score=0.85,
    )
    out = _artifact_from_multi_insight(
        "ep:1",
        [("One insight", "unknown")],
        [[gq]],
        model_version="m",
        prompt_version="v1",
        podcast_id="p",
        episode_title="T",
        date_str="2025-01-01T00:00:00Z",
        transcript_ref="t.txt",
        transcript_text=plain,
        transcript_segments=segs,
    )
    quotes = [n for n in out["nodes"] if n["type"] == "Quote"]
    assert len(quotes) == 1
    assert quotes[0]["properties"]["timestamp_start_ms"] == 0
    assert quotes[0]["properties"]["timestamp_end_ms"] == 1000


def test_process_transcript_download_srt_writes_txt_and_segments(tmp_path) -> None:
    """SRT bytes normalize to .txt and .segments.json on disk."""
    srt = b"""1
00:00:00,000 --> 00:00:01,000
Alpha

2
00:00:01,000 --> 00:00:02,000
 Beta
"""
    ep = create_test_episode(
        idx=2,
        title="SRT Show",
        transcript_urls=[("https://example.com/e.srt", "application/x-subrip")],
    )
    cfg = create_test_config(skip_existing=False)
    out_root = str(tmp_path)
    with (
        patch(
            "podcast_scraper.workflow.episode_processor._fetch_transcript_content",
            return_value=(srt, "application/x-subrip"),
        ),
        patch(
            "podcast_scraper.workflow.episode_processor._check_existing_transcript",
            return_value=False,
        ),
    ):
        ok, rel, src, _n = episode_processor.process_transcript_download(
            ep,
            "https://example.com/e.srt",
            "application/x-subrip",
            cfg,
            out_root,
            None,
        )
    assert ok
    assert rel is not None
    assert rel.endswith(".txt")
    full = os.path.join(out_root, rel)
    assert os.path.isfile(full)
    with open(full, encoding="utf-8") as fh:
        assert "Alpha" in fh.read()
    seg = os.path.splitext(full)[0] + ".segments.json"
    assert os.path.isfile(seg)
    with open(seg, encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data) == 2
    assert src == "direct_download"
