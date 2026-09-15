"""A reprocess must open the transcript its episode's metadata names, not one that looks like it.

`relabel_only` and `rediarize_only` OVERWRITE the transcript they find. They used to find it by
globbing ``"{idx} - *.txt"`` across the whole feed root and taking newest-mtime — but the on-disk
idx is unique inside a ``run_*`` directory and nowhere else, and a feed accumulates run dirs (397
across production). Measured on a16z: 48 staged episodes spread over 14 runs resolved to just 15
transcripts, and 33 were rewritten onto a DIFFERENT episode's file, behind a WARNING and exit 0.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from podcast_scraper.models.entities import Episode, TranscriptionJob
from podcast_scraper.workflow.episode_processor import _existing_transcript_for
from podcast_scraper.workflow.stages.scraping import (
    _on_disk_guid_index,
    _transcript_beside_metadata,
)


def _episode_on_disk(run: Path, idx: int, title: str, guid: str, *, segments: bool = True) -> Path:
    stem = f"{idx:04d} - {title}_{run.name}"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    (run / "transcripts" / f"{stem}.txt").write_text(f"transcript of {title}", encoding="utf-8")
    if segments:
        (run / "transcripts" / f"{stem}.segments.json").write_text("[]", encoding="utf-8")
    meta = run / "metadata" / f"{stem}.metadata.json"
    meta.write_text(
        json.dumps(
            {
                "episode": {"guid": guid, "episode_id": guid, "title": title},
                "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
            }
        ),
        encoding="utf-8",
    )
    return meta


def _job(idx: int, transcript: str | None) -> TranscriptionJob:
    """A real job, not a stub: the function reads `job.episode.on_disk_transcript`, and a
    SimpleNamespace standing in for the dataclass hid that from the type checker."""
    episode = Episode(
        idx=idx,
        title="t",
        title_safe="t",
        item=ET.Element("item"),
        transcript_urls=[],
        on_disk_transcript=transcript,
    )
    return TranscriptionJob(
        idx=idx, ep_title="t", ep_title_safe="t", temp_media="", episode=episode
    )


class TestTheEpisodeKnowsItsOwnFile:
    def test_two_runs_share_an_idx_and_the_right_file_is_still_chosen(self, tmp_path: Path) -> None:
        # Both runs hold an episode numbered 0001, and they are different episodes. The older
        # run's episode is the one under test; newest-mtime would hand back the other one.
        old = tmp_path / "run_old_20260101-000000"
        new = tmp_path / "run_new_20260201-000000"
        mine = _episode_on_disk(old, 1, "The episode I asked for", "guid-mine")
        _episode_on_disk(new, 1, "Somebody elses episode", "guid-other")

        resolved = _transcript_beside_metadata(mine)
        assert resolved is not None
        assert "The episode I asked for" in Path(resolved).name

        chosen = _existing_transcript_for(_job(1, resolved), str(new), "relabel_only")
        assert chosen is not None
        assert chosen.read_text(encoding="utf-8") == "transcript of The episode I asked for"

    def test_the_index_carries_the_metadata_path(self, tmp_path: Path) -> None:
        run = tmp_path / "run_only_20260101-000000"
        meta = _episode_on_disk(run, 3, "A show", "guid-1")
        idx, episode, path = _on_disk_guid_index(str(tmp_path))["guid-1"]
        assert idx == 3
        assert episode["title"] == "A show"
        assert Path(path) == meta


class TestItRefusesRatherThanGuessing:
    def test_a_transcript_with_no_segments_sidecar_is_not_resolved(self, tmp_path: Path) -> None:
        # Nothing the relabel stage can work on. Returning it would move the failure later and
        # make a missing input look like a naming failure.
        run = tmp_path / "run_only_20260101-000000"
        meta = _episode_on_disk(run, 1, "No segments", "guid-1", segments=False)
        assert _transcript_beside_metadata(meta) is None

    def test_a_missing_transcript_is_not_resolved(self, tmp_path: Path) -> None:
        run = tmp_path / "run_only_20260101-000000"
        meta = _episode_on_disk(run, 1, "Gone", "guid-1")
        for f in (run / "transcripts").iterdir():
            f.unlink()
        assert _transcript_beside_metadata(meta) is None


class TestTheFallbackStillWorksForJobsWithoutAnEpisode:
    def test_a_single_run_resolves_by_index_prefix(self, tmp_path: Path) -> None:
        run = tmp_path / "run_only_20260101-000000"
        _episode_on_disk(run, 2, "Only one run here", "guid-1")
        chosen = _existing_transcript_for(_job(2, None), str(run), "relabel_only")
        assert chosen is not None
        assert "Only one run here" in chosen.name

    def test_nothing_on_disk_returns_none(self, tmp_path: Path) -> None:
        assert _existing_transcript_for(_job(1, None), str(tmp_path), "relabel_only") is None
