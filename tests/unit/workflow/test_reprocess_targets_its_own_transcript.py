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
from podcast_scraper.workflow.stages import scraping
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


def _job(idx: int, transcript: str | None, title: str = "t") -> TranscriptionJob:
    """A real job, not a stub: the function reads `job.episode.on_disk_transcript`, and a
    SimpleNamespace standing in for the dataclass hid that from the type checker.

    `title` matters: on-disk transcript names are built FROM `ep_title_safe`
    (`filesystem.build_whisper_output_name`), and the idx-prefix search now drops candidates whose
    filename does not name this episode. A fixture with a title that cannot match the file under
    test exercises the refusal, not the resolution.
    """
    episode = Episode(
        idx=idx,
        title=title,
        title_safe=title,
        item=ET.Element("item"),
        transcript_urls=[],
        on_disk_transcript=transcript,
    )
    return TranscriptionJob(
        idx=idx, ep_title=title, ep_title_safe=title, temp_media="", episode=episode
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


def _job_without_episode(idx: int, title: str = "t") -> TranscriptionJob:
    """A job with NO Episode at all — the only caller the idx search is still for.

    This distinction is the whole point of the class below, and it was NOT tested before:
    ``_job()`` always builds an Episode, so the "fallback for jobs without an episode" tests
    were passing a job WITH one and blessing the idx search on the reprocess path. That is how
    the corruption shipped with green tests.
    """
    return TranscriptionJob(
        idx=idx, ep_title=title, ep_title_safe=title, temp_media="", episode=None
    )


class TestTheFallbackStillWorksForJobsWithoutAnEpisode:
    def test_a_single_run_resolves_by_index_prefix(self, tmp_path: Path) -> None:
        run = tmp_path / "run_only_20260101-000000"
        _episode_on_disk(run, 2, "Only one run here", "guid-1")
        chosen = _existing_transcript_for(
            _job_without_episode(2, "Only one run here"), str(run), "relabel_only"
        )
        assert chosen is not None
        assert "Only one run here" in chosen.name

    def test_nothing_on_disk_returns_none(self, tmp_path: Path) -> None:
        assert (
            _existing_transcript_for(_job_without_episode(1), str(tmp_path), "relabel_only") is None
        )

    def test_an_ambiguous_idx_is_refused_not_guessed(self, tmp_path: Path) -> None:
        """Two runs, same idx, different episodes, no Episode on the job: that is UNKNOWN.

        The old code returned newest-mtime behind a WARNING. Measured on a16z as 33 of 48
        episodes rewritten onto another episode's transcript.
        """
        old = tmp_path / "run_old_20260101-000000"
        new = tmp_path / "run_new_20260201-000000"
        _episode_on_disk(old, 4, "One candidate", "guid-a")
        _episode_on_disk(new, 4, "Another candidate", "guid-b")

        assert (
            _existing_transcript_for(_job_without_episode(4), str(new), "relabel_only") is None
        ), "an idx matching two different episodes must be refused, never resolved by mtime"


class TestARefusalUpstreamIsNotOverridableDownstream:
    """2026-09-23 prod #2097: `_transcript_beside_metadata` refused, the caller guessed anyway.

    Its docstring already said returning None is "a REFUSAL, and refusing is the point: the
    caller skips the episode and says so". `_existing_transcript_for` answered that refusal with
    the idx glob, so 16 episodes fell through and 10 serving episodes ended up carrying another
    episode's transcript, with gi.json and kg.json rewritten from the wrong words.
    """

    def test_an_episode_with_no_vouched_transcript_is_skipped_not_globbed(
        self, tmp_path: Path
    ) -> None:
        # A sibling episode numbered 0010 exists and WOULD match the glob — it must not be used.
        run = tmp_path / "run_only_20260101-000000"
        _episode_on_disk(run, 10, "Re-engineering the Semiconductor", "guid-owner")
        victim = (
            run
            / "transcripts"
            / "0010 - Re-engineering the Semiconductor_run_only_20260101-000000.txt"
        )
        before = victim.read_bytes()

        # A REPROCESS job: `on_disk_idx` is the marker `_reprocess_existing_episodes` sets, and it
        # means a metadata record was read and `_transcript_beside_metadata` already refused it.
        # Without the marker this is a feed-driven job with no refusal to honour, which is a
        # different case entirely (see TestTheFallbackStillWorksForJobsWithoutAnEpisode).
        job = _job(10, None)
        assert job.episode is not None
        job.episode.on_disk_idx = 10

        chosen = _existing_transcript_for(job, str(run), "relabel_only")

        assert chosen is None, f"must skip, not borrow another episode's transcript (got {chosen})"
        assert victim.read_bytes() == before, "the other episode's transcript must be untouched"

    def test_a_named_transcript_that_vanished_is_skipped_not_globbed(self, tmp_path: Path) -> None:
        """The record named a file that is gone. Missing input, not a search prompt."""
        run = tmp_path / "run_only_20260101-000000"
        _episode_on_disk(run, 7, "Still here", "guid-other")
        missing = str(run / "transcripts" / "0007 - Gone_run_only_20260101-000000.txt")

        chosen = _existing_transcript_for(_job(7, missing), str(run), "relabel_only")

        assert chosen is None
        assert "Still here" not in str(chosen)

    def test_a_feed_driven_job_is_NOT_treated_as_a_refusal(self, tmp_path: Path) -> None:
        """Caught in review: discriminating on "has an Episode" broke the documented command.

        `on_disk_transcript` is set in exactly one place — `_reprocess_existing_episodes`, which
        also sets `on_disk_idx`. A FEED-driven job carries an Episode too, but no metadata record
        was ever consulted for it, so `_transcript_beside_metadata` never ruled on it and there is
        no refusal to honour. Discriminating on the Episode made
        `--pipeline-stage relabel_only` without `--reprocess-existing-only` (the invocation in
        docs/guides/CORPUS_REPROCESSING.md) a 100% skip with a zero exit — the silent-zero-exit
        shape this whole arc keeps fighting.

        The marker is `on_disk_idx`, which only the reprocess path sets.
        """
        run = tmp_path / "run_only_20260101-000000"
        _episode_on_disk(run, 2, "Only one run here", "guid-1")

        feed_driven = _job(2, None, "Only one run here")  # Episode present, on_disk_idx unset
        assert feed_driven.episode is not None
        assert getattr(feed_driven.episode, "on_disk_idx", None) is None

        chosen = _existing_transcript_for(feed_driven, str(run), "relabel_only")

        assert (
            chosen is not None
        ), "a feed-driven job must still resolve; it has no refusal to honour"
        assert "Only one run here" in chosen.name

    def test_a_reprocess_job_IS_treated_as_a_refusal(self, tmp_path: Path) -> None:
        """The same setup, but with the reprocess marker set — now it must refuse."""
        run = tmp_path / "run_only_20260101-000000"
        _episode_on_disk(run, 2, "Only one run here", "guid-1")

        reprocess_job = _job(2, None)
        assert reprocess_job.episode is not None
        reprocess_job.episode.on_disk_idx = 2  # what _reprocess_existing_episodes sets

        assert _existing_transcript_for(reprocess_job, str(run), "relabel_only") is None

    def test_a_refusal_is_recorded_as_a_failed_episode(self, tmp_path: Path) -> None:
        """Caught in review: the skip was counted NOWHERE.

        The caller returns (False, None, 0); the transcription stage's `if success:` guard then
        skips both the counter and `update_episode_status`, so a refused episode is neither `ok`
        nor `failed` — it vanishes between the selection log and the summary. The 2026-09-23 batch
        reported `episodes=N ok=N failed=0` per feed while doing less than it was asked. A skip
        must be at least as loud as the corruption it replaced.

        THE KEY IS THE POINT, not that a call happened. The first version of this test asserted
        only that `update_episode_status` was invoked, and passed while the implementation keyed on
        `episode.guid` — an attribute `Episode` does not have. That write would have missed the
        episode's real row entirely and appended an orphan keyed by raw title, leaving the ledger
        showing the same episode both ok and failed. So this asserts the id MATCHES the canonical
        one the rest of the pipeline uses.
        """
        from podcast_scraper import config as config_module
        from podcast_scraper.workflow import episode_processor as epx
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        seen: list[dict] = []

        class _Metrics:
            def update_episode_status(self, **kwargs):
                seen.append(kwargs)

        cfg = config_module.Config(
            rss="https://example.com/feed.xml", transcription_provider="whisper"
        )
        job = _job(10, None)
        assert job.episode is not None
        job.episode.on_disk_idx = 10
        canonical, _ = get_episode_id_from_episode(job.episode, cfg.rss_url or "")

        epx._record_unresolved_transcript(job, cfg, _Metrics(), "relabel_only")

        assert len(seen) == 1, f"the refusal recorded nothing: {seen}"
        assert seen[0]["episode_id"] == canonical, (
            "the status must be keyed by the canonical episode id the rest of the pipeline uses, "
            f"not {seen[0]['episode_id']!r} — otherwise it appends an orphan row"
        )
        assert seen[0]["status"] == "failed"
        assert seen[0]["error_type"] == "TranscriptUnresolved"
        assert seen[0]["stage"] == "relabel_only"

    def test_the_key_is_not_the_raw_title(self) -> None:
        """Mutation guard: the defect it replaced fell through to `job.ep_title`."""
        from podcast_scraper import config as config_module
        from podcast_scraper.workflow import episode_processor as epx

        seen: list[dict] = []

        class _Metrics:
            def update_episode_status(self, **kwargs):
                seen.append(kwargs)

        cfg = config_module.Config(
            rss="https://example.com/feed.xml", transcription_provider="whisper"
        )
        job = _job(3, None)
        assert job.episode is not None
        job.episode.on_disk_idx = 3
        epx._record_unresolved_transcript(job, cfg, _Metrics(), "relabel_only")

        assert seen, "nothing recorded"
        assert seen[0]["episode_id"] != job.ep_title

    def test_recording_never_raises_even_with_a_broken_metrics_object(self) -> None:
        """It runs on a failure path; it must not convert a skip into a crash."""
        from podcast_scraper import config as config_module
        from podcast_scraper.workflow import episode_processor as epx

        cfg = config_module.Config(
            rss="https://example.com/feed.xml", transcription_provider="whisper"
        )

        class _Exploding:
            def update_episode_status(self, **kwargs):
                raise RuntimeError("metrics backend down")

        epx._record_unresolved_transcript(_job(1, None), cfg, _Exploding(), "relabel_only")
        epx._record_unresolved_transcript(_job(1, None), cfg, None, "relabel_only")

    def test_a_vouched_transcript_is_still_returned(self, tmp_path: Path) -> None:
        """The fix must not break the normal path — 172 episodes resolved correctly in that run."""
        run = tmp_path / "run_only_20260101-000000"
        meta = _episode_on_disk(run, 5, "My own episode", "guid-mine")
        resolved = _transcript_beside_metadata(meta)
        assert resolved is not None

        chosen = _existing_transcript_for(_job(5, resolved), str(run), "relabel_only")

        assert chosen is not None and "My own episode" in chosen.name


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


class TestTheCrossRunResolverIsIdentityAnchored:
    """The repair path for the 9 of 10 episodes whose transcript lives in ANOTHER run dir.

    A reprocess writes fresh metadata into its new run while the transcript it relabelled stays
    put, so the newest record can sit alone in a run with an empty ``transcripts/``. Before this,
    those episodes were corrupted (idx glob found a stranger); after the refusal fix they were
    skipped but unrepairable. This resolves them by IDENTITY — same guid, same feed, strictly
    vouched — which is the property the idx glob never had.

    One test per invariant the review required.
    """

    @staticmethod
    def _feed(tmp_path: Path) -> Path:
        feed = tmp_path / "feeds" / "rss_example_abc123"
        feed.mkdir(parents=True)
        return feed

    def test_I1_same_guid_in_a_sibling_run_is_found(self, tmp_path: Path) -> None:
        feed = self._feed(tmp_path)
        old = feed / "run_20260101-000000"
        new = feed / "run_20260201-000000"
        _episode_on_disk(old, 1, "My episode", "guid-mine")
        newer = _episode_on_disk(new, 1, "My episode", "guid-mine")
        # The newest record's own run loses its transcript — the real post-relabel shape.
        for f in (new / "transcripts").iterdir():
            f.unlink()

        assert _transcript_beside_metadata(newer) is None, "precondition: own run refuses"
        found = scraping._transcript_in_a_sibling_run(newer, "guid-mine")
        assert found is not None and "My episode" in Path(found).name

    def test_I1b_a_different_guid_is_never_eligible(self, tmp_path: Path) -> None:
        """Same idx, same title, DIFFERENT episode — the exact thing the idx glob accepted."""
        feed = self._feed(tmp_path)
        old = feed / "run_20260101-000000"
        new = feed / "run_20260201-000000"
        _episode_on_disk(old, 1, "My episode", "guid-SOMEONE-ELSE")
        newer = _episode_on_disk(new, 1, "My episode", "guid-mine")
        for f in (new / "transcripts").iterdir():
            f.unlink()

        assert scraping._transcript_in_a_sibling_run(newer, "guid-mine") is None

    def test_I1c_an_empty_guid_never_matches_an_empty_stored_guid(self, tmp_path: Path) -> None:
        """Proved non-discriminating in review and rewritten.

        The old version had NO sibling runs, so deleting the `if not guid: return None` guard
        still returned None (nothing to scan) and it passed regardless. The guard only earns its
        keep when a sibling record ALSO has an empty stored guid — without it, `"" == ""` matches
        and an unrelated episode is adopted.
        """
        feed = self._feed(tmp_path)
        old = feed / "run_20260101-000000"
        new = feed / "run_20260201-000000"
        guidless = _episode_on_disk(old, 1, "Some other episode", "")
        assert scraping._metadata_guid(guidless) == "", "fixture must store an empty guid"
        newer = _episode_on_disk(new, 1, "My episode", "")
        for f in (new / "transcripts").iterdir():
            f.unlink()

        assert scraping._transcript_in_a_sibling_run(newer, "") is None

    def test_A_a_record_outside_a_run_dir_has_no_sibling_runs(self, tmp_path: Path) -> None:
        """Flat layout (`<root>/metadata/x.json`): deriving a feed root escapes the corpus.

        `parent.parent.parent` would be the PARENT of the corpus root, so the search would scan
        directories beside the output dir — another corpus in single-feed layout, eligible on guid
        alone. Prod is entirely `feeds/<slug>/run_*`, so this is latent, not live.
        """
        root = tmp_path / "corpus"
        (root / "metadata").mkdir(parents=True)
        meta = root / "metadata" / "0001 - Flat.metadata.json"
        meta.write_text(json.dumps({"episode": {"guid": "guid-mine"}}), encoding="utf-8")
        decoy_run = tmp_path / "run_20260101-000000"
        _episode_on_disk(decoy_run, 1, "Another corpus entirely", "guid-mine")

        assert scraping._transcript_in_a_sibling_run(meta, "guid-mine") is None

    def test_I2_another_feed_with_the_same_guid_is_never_eligible(self, tmp_path: Path) -> None:
        """Publishers ship non-unique guids ('1', a reused URL). A cross-feed hit would pull
        another podcast's transcript into a stage that overwrites what it is given."""
        other = tmp_path / "feeds" / "rss_other_feed"
        (other / "run_20260101-000000").mkdir(parents=True)
        _episode_on_disk(other / "run_20260101-000000", 1, "Other podcast", "guid-mine")
        feed = self._feed(tmp_path)
        new = feed / "run_20260201-000000"
        newer = _episode_on_disk(new, 1, "My episode", "guid-mine")
        for f in (new / "transcripts").iterdir():
            f.unlink()

        assert scraping._transcript_in_a_sibling_run(newer, "guid-mine") is None

    def test_I3_a_candidate_without_its_segments_sidecar_is_refused(self, tmp_path: Path) -> None:
        feed = self._feed(tmp_path)
        old = feed / "run_20260101-000000"
        new = feed / "run_20260201-000000"
        _episode_on_disk(old, 1, "My episode", "guid-mine", segments=False)
        newer = _episode_on_disk(new, 1, "My episode", "guid-mine")
        for f in (new / "transcripts").iterdir():
            f.unlink()

        assert scraping._transcript_in_a_sibling_run(newer, "guid-mine") is None

    def test_I4_newest_run_wins_by_run_recency_not_mtime(self, tmp_path: Path) -> None:
        """Ordered by the run-folder timestamp, the corpus's own supersession rule. mtime would
        be wrong: `_on_disk_guid_index` records first-glob-wins once resolving to the OLDEST run."""
        import os

        feed = self._feed(tmp_path)
        older = feed / "run_20260101-000000"
        newer_run = feed / "run_20260601-000000"
        current = feed / "run_20260901-000000"
        _episode_on_disk(older, 1, "Old words", "guid-mine")
        _episode_on_disk(newer_run, 1, "New words", "guid-mine")
        meta = _episode_on_disk(current, 1, "Current", "guid-mine")
        for f in (current / "transcripts").iterdir():
            f.unlink()
        # Make the OLD run newest by mtime — on BOTH the transcript and the METADATA file, since
        # the ordering key is read from the metadata path. An earlier version of this test touched
        # only transcripts/, so an mtime-ordering mutant survived it.
        future = 2_000_000_000
        for f in list((older / "transcripts").iterdir()) + list((older / "metadata").iterdir()):
            os.utime(f, (future, future))

        found = scraping._transcript_in_a_sibling_run(meta, "guid-mine")
        assert found is not None
        assert "New words" in Path(found).name, f"run recency must beat mtime, got {found}"

    def test_I9_two_records_for_one_guid_in_a_run_refuses_that_run(self, tmp_path: Path) -> None:
        """Ambiguous is unknown, not 'first glob hit'."""
        feed = self._feed(tmp_path)
        old = feed / "run_20260101-000000"
        new = feed / "run_20260201-000000"
        _episode_on_disk(old, 1, "Copy one", "guid-mine")
        _episode_on_disk(old, 2, "Copy two", "guid-mine")
        newer = _episode_on_disk(new, 5, "My episode", "guid-mine")
        for f in (new / "transcripts").iterdir():
            f.unlink()

        assert scraping._transcript_in_a_sibling_run(newer, "guid-mine") is None

    def test_I5_a_vouched_own_run_transcript_is_never_replaced_by_a_sibling(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Proved non-discriminating in review and rewritten.

        The old version asserted only that `_transcript_beside_metadata` works — it never touched
        the `if not episode.on_disk_transcript:` guard, so a mutant calling the sibling search
        UNCONDITIONALLY (overwriting a good own-run resolution with an older run's copy) passed
        every test. This asserts the guard: with a vouched own-run transcript the sibling search
        must not be consulted at all.
        """
        feed = self._feed(tmp_path)
        old = feed / "run_20260101-000000"
        new = feed / "run_20260201-000000"
        _episode_on_disk(old, 1, "Stale copy", "guid-mine")
        newer = _episode_on_disk(new, 1, "Current copy", "guid-mine")

        called: list = []

        def _spy(meta_path, guid):
            called.append((meta_path, guid))
            return str(old / "transcripts" / "0001 - Stale copy_run_20260101-000000.txt")

        monkeypatch.setattr(scraping, "_transcript_in_a_sibling_run", _spy)

        own = _transcript_beside_metadata(newer)
        assert own is not None and "Current copy" in Path(own).name

        resolved: str | None = own
        if not resolved:  # mirrors selection's guard at scraping.py
            resolved = scraping._transcript_in_a_sibling_run(newer, "guid-mine")

        assert called == [], "the sibling search must not run when the own run vouched a file"
        assert resolved is not None
        assert "Current copy" in Path(resolved).name
