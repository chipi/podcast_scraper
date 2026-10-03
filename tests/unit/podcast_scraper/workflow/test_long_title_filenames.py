#!/usr/bin/env python3
"""A very long episode title must not break artifact filenames, and a failed write must not pass.

Prod 2026-10-01 (job 3f243d28): a 200-char title produced a transcript filename past the 255-byte
component limit, ``write_file`` raised ``[Errno 36] File name too long``, the error was logged and
swallowed, and the run reported ``ok=1 failed=0`` with nothing on disk.
"""

import errno
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.utils import filesystem
from podcast_scraper.workflow import episode_processor
from podcast_scraper.workflow.helpers import get_episode_id_from_episode
from podcast_scraper.workflow.metrics import Metrics

_tests_dir = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("tests_parent_conftest", _tests_dir / "conftest.py")
if _spec is None or _spec.loader is None:
    raise ImportError("conftest not loadable")
_ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ct)
create_test_config = _ct.create_test_config
create_test_episode = _ct.create_test_episode

NAME_MAX = 255
RUN_SUFFIX = "7096bc13-0000-4000-8000-798ce307b1d6_20261001-214318_285e51f2"
# Every sidecar the pipeline derives from a transcript stem, longest last.
SIDECAR_SUFFIXES = (
    ".txt",
    ".segments.json",
    ".cleaned.txt",
    ".adfree.txt",
    ".adfree.segments.json",
    ".cleaned.segments.json",
    ".speakers.diagnostics.json",
)
LONG_ASCII = "The Accountability Ladder, Learning to Lead by Leading Yourself " * 5
LONG_MULTIBYTE = "Лидерство и ответственность — 責任のはしご " * 12


def _safe(title: str) -> str:
    return filesystem.sanitize_filename(title)


@pytest.mark.parametrize("title", [LONG_ASCII, LONG_MULTIBYTE], ids=["ascii", "multibyte"])
@pytest.mark.parametrize("run_suffix", [None, RUN_SUFFIX], ids=["no-run", "run"])
def test_every_artifact_name_fits_the_filesystem_limit(title: str, run_suffix) -> None:
    assert len(title) >= 300
    base = filesystem.build_transcript_base_name(1, _safe(title), run_suffix)
    for suffix in SIDECAR_SUFFIXES:
        assert len((base + suffix).encode("utf-8")) <= NAME_MAX, suffix
    assert base.startswith("0001 - ")
    if run_suffix:
        assert base.endswith(f"_{run_suffix}")


def test_normal_titles_keep_their_exact_name() -> None:
    assert (
        filesystem.build_transcript_base_name(7, "Episode_Title", "20261001-214318")
        == "0007 - Episode_Title_20261001-214318"
    )
    assert filesystem.build_transcript_base_name(7, "Episode_Title", None) == "0007 - Episode_Title"
    mid = _safe("Привет мир " * 8)
    assert filesystem.build_transcript_base_name(3, mid, None) == f"0003 - {mid}"


@pytest.mark.parametrize("title", [LONG_ASCII, LONG_MULTIBYTE], ids=["ascii", "multibyte"])
def test_long_title_transcript_is_written_and_found_again(tmp_path, title: str) -> None:
    ep = create_test_episode(
        idx=1,
        title=title,
        title_safe=_safe(title),
        transcript_urls=[("https://example.com/e.txt", "text/plain")],
    )
    cfg = create_test_config(skip_existing=False)
    with (
        patch(
            "podcast_scraper.workflow.episode_processor._fetch_transcript_content",
            return_value=(b"hello world", "text/plain"),
        ),
        patch(
            "podcast_scraper.workflow.episode_processor._check_existing_transcript",
            return_value=False,
        ),
    ):
        ok, rel, src, _n = episode_processor.process_transcript_download(
            ep, "https://example.com/e.txt", "text/plain", cfg, str(tmp_path), RUN_SUFFIX
        )
    assert ok is True and src == "direct_download" and rel is not None
    written = tmp_path / rel
    assert written.read_bytes() == b"hello world"
    assert len(written.name.encode("utf-8")) <= NAME_MAX
    # The skip-existing lookup re-derives the name; it must land on the file just written.
    assert episode_processor._determine_output_path(
        ep, "https://example.com/e.txt", "text/plain", str(tmp_path), RUN_SUFFIX, ".txt"
    ) == str(written)


def test_failed_transcript_write_marks_the_episode_failed(tmp_path) -> None:
    ep = create_test_episode(
        idx=1,
        title="Some Title",
        title_safe="Some_Title",
        transcript_urls=[("https://example.com/e.txt", "text/plain")],
    )
    cfg = create_test_config(skip_existing=False)
    pm = Metrics()
    episode_id, number = get_episode_id_from_episode(ep, cfg.rss_url or "")
    pm.get_or_create_episode_status(episode_id, number or ep.idx)
    with (
        patch(
            "podcast_scraper.workflow.episode_processor._fetch_transcript_content",
            return_value=(b"hello world", "text/plain"),
        ),
        patch(
            "podcast_scraper.workflow.episode_processor._check_existing_transcript",
            return_value=False,
        ),
        patch.object(
            filesystem,
            "write_file",
            side_effect=OSError(errno.ENAMETOOLONG, "File name too long"),
        ),
    ):
        ok, rel, src, _n = episode_processor.process_transcript_download(
            ep,
            "https://example.com/e.txt",
            "text/plain",
            cfg,
            str(tmp_path),
            None,
            pipeline_metrics=pm,
        )
    assert (ok, rel, src) == (False, None, None)
    (status,) = pm.episode_statuses
    assert status.status == "failed"
    assert status.stage == "transcript_write"
    assert status.error_type == OSError(errno.ENAMETOOLONG, "").__class__.__name__
    assert "File name too long" in (status.error_message or "")
    assert pm.errors_total == 1
