"""A truncated metadata title must still find its own transcript — and never another episode's.

THE INCIDENT (prod, 2026-09-29). The metadata filename cuts the title at ~32 characters; the
transcript filename keeps it whole. ``run_index._transcript_beside`` only tried the exact metadata
stem, so 167 of 2,002 served episodes resolved to "no transcript" although the file was on disk and
the record's own ``transcript_file_path`` named it. ``rederive_only`` refused them, which left two
omnycontent knowledge graphs unrepairable. The shape below is one of those two, reproduced on prod.

The pointer is followed only when it names THIS episode: #2082 found 147 records pointing at
another episode's transcript, and following those would feed episode B's words into A's graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.workflow import run_index

pytestmark = pytest.mark.unit

_RUN = "run_d8d441c2-ee02-42f5-8cc3-f58a887f25a4_20260831-212513"
_TAIL = "_d8d441c2-ee02-42f5-8cc3-f58a887f25a4_20260831-212513"
_META_STEM = "0004 - The Tungsten Market Is Warning o" + _TAIL
_FULL_STEM = "0004 - The Tungsten Market Is Warning of an Upcoming War" + _TAIL


def _run_dir(tmp_path: Path, pointer: str | None, transcripts: list[str]) -> Path:
    run = tmp_path / "feeds" / "rss_www.omnycontent.com_c9fdce2d" / _RUN
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    for name in transcripts:
        (run / "transcripts" / name).write_text("HOST: words", encoding="utf-8")
    content = {} if pointer is None else {"transcript_file_path": pointer}
    meta = run / "metadata" / f"{_META_STEM}.metadata.json"
    meta.write_text(json.dumps({"content": content}), encoding="utf-8")
    return meta


def test_a_truncated_title_finds_the_transcript_its_record_names(tmp_path: Path) -> None:
    """THE bug: the prod layout for episode 6ea24954, exactly."""
    meta = _run_dir(
        tmp_path,
        f"transcripts/{_FULL_STEM}.txt",
        [f"{_FULL_STEM}.txt", f"{_FULL_STEM}.cleaned.txt", f"{_FULL_STEM}.adfree.txt"],
    )
    found = run_index._transcript_beside(meta)
    assert found is not None, "the transcript is on disk and named by the record, yet not found"
    assert found.endswith(f"{_FULL_STEM}.txt")


def test_a_pointer_to_another_episode_is_refused(tmp_path: Path) -> None:
    """#2082: same idx, same run, a DIFFERENT title — B's words must never reach A."""
    other = "0004 - Why Copper Prices Keep Rising" + _TAIL
    meta = _run_dir(tmp_path, f"transcripts/{other}.txt", [f"{other}.txt"])
    assert run_index._transcript_beside(meta) is None


def test_a_pointer_outside_this_runs_transcripts_is_refused(tmp_path: Path) -> None:
    meta = _run_dir(tmp_path, f"../../elsewhere/transcripts/{_FULL_STEM}.txt", [])
    elsewhere = tmp_path / "feeds" / "elsewhere" / "transcripts"
    elsewhere.mkdir(parents=True)
    (elsewhere / f"{_FULL_STEM}.txt").write_text("x", encoding="utf-8")
    assert run_index._transcript_beside(meta) is None


@pytest.mark.parametrize("variant", [".cleaned.txt", ".adfree.txt", ".segments.json"])
def test_a_pointer_to_a_derivative_is_refused(tmp_path: Path, variant: str) -> None:
    """Same accept-list as the exact-stem path: derived text and JSON are not the transcript."""
    meta = _run_dir(tmp_path, f"transcripts/{_FULL_STEM}{variant}", [f"{_FULL_STEM}{variant}"])
    assert run_index._transcript_beside(meta) is None


def test_a_pointer_to_a_missing_file_is_not_a_transcript(tmp_path: Path) -> None:
    meta = _run_dir(tmp_path, f"transcripts/{_FULL_STEM}.txt", [])
    assert run_index._transcript_beside(meta) is None


def test_no_pointer_still_means_no_transcript(tmp_path: Path) -> None:
    meta = _run_dir(tmp_path, None, [f"{_FULL_STEM}.txt"])
    assert run_index._transcript_beside(meta) is None


def test_the_exact_stem_still_wins_over_the_pointer(tmp_path: Path) -> None:
    meta = _run_dir(
        tmp_path, f"transcripts/{_FULL_STEM}.txt", [f"{_META_STEM}.txt", f"{_FULL_STEM}.txt"]
    )
    found = run_index._transcript_beside(meta)
    assert found is not None and found.endswith(f"{_META_STEM}.txt")
