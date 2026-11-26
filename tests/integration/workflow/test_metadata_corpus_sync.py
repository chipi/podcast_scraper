"""The metadata stage syncs an episode's transcript documents (w-public), and a strict failure
fails the episode's metadata."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from podcast_scraper.db import corpus_writer
from podcast_scraper.workflow import metadata_generation

_tests_dir = Path(__file__).parent.parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
_spec = importlib.util.spec_from_file_location("parent_conftest", _tests_dir / "conftest.py")
assert _spec is not None and _spec.loader is not None
parent_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent_conftest)

TRANSCRIPT = "transcripts/0001 - Episode_Title.txt"


def _generate(output_dir: Path) -> object:
    cfg = parent_conftest.create_test_config(
        output_dir=str(output_dir), generate_metadata=True, metadata_format="json"
    )
    return metadata_generation.generate_episode_metadata(
        feed=parent_conftest.create_test_feed(),
        episode=parent_conftest.create_test_episode(),
        feed_url=parent_conftest.TEST_FEED_URL,
        cfg=cfg,
        output_dir=str(output_dir),
        transcript_file_path=TRANSCRIPT,
        transcript_source="whisper_transcription",
    )


@pytest.mark.integration
@pytest.mark.critical_path
def test_the_metadata_stage_syncs_the_written_metadata_and_its_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, Path, str]] = []
    monkeypatch.setattr(
        corpus_writer,
        "sync_episode",
        lambda meta, transcript, change_kind: calls.append((meta, transcript, change_kind)),
    )
    metadata_path = _generate(tmp_path)
    assert calls == [(Path(str(metadata_path)), tmp_path / TRANSCRIPT, "retranscribed")]
    assert calls[0][0].is_file()


@pytest.mark.integration
@pytest.mark.critical_path
def test_a_strict_sync_failure_fails_the_episodes_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(meta: Path, transcript: Path, change_kind: str) -> str:
        raise corpus_writer.SyncFailed("corpus write refused")

    monkeypatch.setattr(corpus_writer, "sync_episode", fail)
    assert _generate(tmp_path) is None
