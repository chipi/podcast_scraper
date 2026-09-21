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


class TestAPointerAtAnotherEpisodeIsRefused:
    """#2082 on the repair path: the stored pointer is the corrupted field.

    147 of 2,297 production records name ANOTHER episode's transcript, and that file exists with
    its `.segments.json` — which is exactly what makes the audit call it a confirmed
    misattribution. Following it meant a scoped `relabel_only` of A overwrote B's transcript with
    names re-resolved from B's words, and rewrote B's roster. The repair step re-inflicted the
    damage it was run to fix, silently, and left A untouched.
    """

    @staticmethod
    def _mispair(run: Path) -> tuple[Path, Path]:
        """A's record pointing at B's transcript, both on disk. Returns ``(a_meta, b_txt)``."""
        a_meta = _episode_on_disk(run, 1, "Krishna Rao on being a CFO", "guid-a")
        b_meta = _episode_on_disk(run, 2, "Sam Altman on abundance", "guid-b")
        b_txt = run / "transcripts" / f"{b_meta.name[: -len('.metadata.json')]}.txt"
        payload = json.loads(a_meta.read_text(encoding="utf-8"))
        payload["content"]["transcript_file_path"] = f"transcripts/{b_txt.name}"
        a_meta.write_text(json.dumps(payload), encoding="utf-8")
        return a_meta, b_txt

    def test_the_episodes_own_transcript_wins_over_the_pointer(self, tmp_path: Path) -> None:
        run = tmp_path / "run_only_20260101-000000"
        a_meta, _b_txt = self._mispair(run)

        resolved = _transcript_beside_metadata(a_meta)

        assert resolved is not None
        assert "Krishna Rao" in Path(resolved).name, resolved
        assert "Sam Altman" not in Path(resolved).name

    def test_a_pointer_elsewhere_is_refused_when_the_own_file_is_gone(self, tmp_path: Path) -> None:
        """The dangerous half: with A's transcript missing, falling through would overwrite B."""
        run = tmp_path / "run_only_20260101-000000"
        a_meta, b_txt = self._mispair(run)
        stem = a_meta.name[: -len(".metadata.json")]
        (run / "transcripts" / f"{stem}.txt").unlink()
        (run / "transcripts" / f"{stem}.segments.json").unlink()
        before = b_txt.read_bytes()

        assert _transcript_beside_metadata(a_meta) is None
        assert b_txt.read_bytes() == before

    def test_a_truncated_title_is_still_the_same_episode(self, tmp_path: Path) -> None:
        """The rule that makes equality the WRONG test — 280 of 2,298 records differ this way.

        The metadata filename truncates the title to 32 chars; the transcript filename does not.
        Refusing those would break every downloaded transcript in the corpus.
        """
        run = tmp_path / "run_only_20260101-000000"
        long_title = "This Funding Model is Helping Fight Climate Change"
        stem_full = f"0006 - {long_title}_{run.name}"
        (run / "metadata").mkdir(parents=True, exist_ok=True)
        (run / "transcripts").mkdir(parents=True, exist_ok=True)
        (run / "transcripts" / f"{stem_full}.txt").write_text("t", encoding="utf-8")
        (run / "transcripts" / f"{stem_full}.segments.json").write_text("[]", encoding="utf-8")
        stem_cut = f"0006 - {long_title[:32]}_{run.name}"
        meta = run / "metadata" / f"{stem_cut}.metadata.json"
        meta.write_text(
            json.dumps({"content": {"transcript_file_path": f"transcripts/{stem_full}.txt"}}),
            encoding="utf-8",
        )

        resolved = _transcript_beside_metadata(meta)

        assert resolved is not None and Path(resolved).name == f"{stem_full}.txt", resolved


class TestTheFallbackStillWorksForJobsWithoutAnEpisode:
    def test_a_single_run_resolves_by_index_prefix(self, tmp_path: Path) -> None:
        run = tmp_path / "run_only_20260101-000000"
        _episode_on_disk(run, 2, "Only one run here", "guid-1")
        chosen = _existing_transcript_for(_job(2, None), str(run), "relabel_only")
        assert chosen is not None
        assert "Only one run here" in chosen.name

    def test_nothing_on_disk_returns_none(self, tmp_path: Path) -> None:
        assert _existing_transcript_for(_job(1, None), str(tmp_path), "relabel_only") is None


class TestTheRunIndexIsUnique:
    """The on-disk idx is not unique, which is why `idx` can no longer be it (#2082).

    Every run directory numbers its episodes from 0001, so a feed with fourteen run dirs has
    fourteen "episode 1"s — and `idx` keys per-episode state and output filenames. On production
    that collided 275 episodes onto another episode's transcript, 119 of them confirmed crediting
    another episode's people.

    That the SELECTION renumbers them uniquely is asserted in
    `test_scraping.py::test_reprocess_carries_the_on_disk_idx_but_numbers_the_run_uniquely`, which
    goes through the public `prepare_episodes_from_feed`. This is the underlying fact it rests on.
    """

    def test_the_on_disk_index_collides_across_run_dirs(self, tmp_path: Path) -> None:
        from podcast_scraper.workflow.stages.scraping import _on_disk_guid_index

        for n, run_name in enumerate(
            ("run_a_20260101-000000", "run_b_20260201-000000", "run_c_20260301-000000")
        ):
            _episode_on_disk(tmp_path / run_name, 1, f"Episode from run {n}", f"guid-{n}")

        idx = _on_disk_guid_index(str(tmp_path))
        assert len(idx) == 3, "three distinct episodes"
        assert {v[0] for v in idx.values()} == {1}, "and all three claim to be episode 1"
        # each still resolves to its OWN transcript — the pairing is sound, the NUMBER is not
        resolved = {Path(str(_transcript_beside_metadata(Path(v[2])))).name for v in idx.values()}
        assert len(resolved) == 3, "each episode must resolve to a different transcript"
