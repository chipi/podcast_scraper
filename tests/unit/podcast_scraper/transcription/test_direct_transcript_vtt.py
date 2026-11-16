#!/usr/bin/env python3
"""Integration checks: VTT/SRT direct download and GI timing (issue #544)."""

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

from podcast_scraper.workflow import episode_processor

# Reach ``tests/conftest.py`` from this file's location
# (tests/unit/podcast_scraper/transcription/<file>.py — 4 levels up).
_tests_dir = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("tests_parent_conftest", _tests_dir / "conftest.py")
if _spec is None or _spec.loader is None:
    raise ImportError("conftest not loadable")
_integration_ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_integration_ct)
create_test_config = _integration_ct.create_test_config
create_test_episode = _integration_ct.create_test_episode


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
