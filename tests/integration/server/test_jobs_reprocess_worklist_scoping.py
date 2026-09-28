"""The Jobs API must be able to scope a reprocess to named episodes — not only to a whole feed.

WHY THIS EXISTS (prod, 2026-09-28). A reprocess run's episode set is *every episode already on disk
for the feed*: ``prepare_episodes_from_feed`` returns via ``_reprocess_existing_episodes`` BEFORE the
offset/limit block, so ``max_episodes`` / ``episode_offset`` / ``episode_selection`` / ``--since`` /
``--until`` are all silently inert in that mode (``config.py`` says so too: those caps "are ignored
so every matched existing episode is processed").

A ``reprocess_episode_ids`` work-list is therefore the ONLY narrowing mechanism — and the API did not
expose it. Repairing one episode via the supported path was impossible; the attempt smuggled the
work-list through the operator YAML, where a CLI merge bug discarded it, and 50 episodes were
rewritten in place instead of 1.

The ids travel via a FILE because that is how the CLI reads them, and because a 50-id repair has no
business in the process table or in the registry's ``argv_summary``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.jobs import build_pipeline_argv, normalize_reprocess_episode_ids

pytestmark = [pytest.mark.integration]

_FEED = "https://example.com/podcast.xml"


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: local\n", encoding="utf-8")
    return corpus, op


def _worklist_path(argv: list[str]) -> Path:
    return Path(argv[argv.index("--reprocess-episode-ids") + 1])


class TestTheWorkListReachesTheRun:
    def test_ids_become_a_worklist_file_referenced_on_argv(self, tmp_path: Path) -> None:
        corpus, op = _corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus,
            op,
            run_id="job-1",
            feed_url=_FEED,
            pipeline_stage="retranscript_only",
            reprocess_episode_ids=["ep-a", "ep-b"],
        )
        wl = _worklist_path(argv)
        assert wl.is_file()
        assert wl.read_text(encoding="utf-8").split() == ["ep-a", "ep-b"]

    def test_the_worklist_lives_beside_the_job_log_and_is_named_from_the_job_id(
        self, tmp_path: Path
    ) -> None:
        """The job id names the file, so a run is reconstructable from the corpus alone."""
        corpus, op = _corpus(tmp_path)
        job_id = "a3f1c2d4-5e6b-4a7c-8d9e-0f1a2b3c4d5e"
        argv = build_pipeline_argv(
            corpus, op, run_id=job_id, pipeline_stage="relabel_only", reprocess_episode_ids=["x"]
        )
        wl = _worklist_path(argv)
        assert wl.parent == corpus / ".viewer" / "jobs"
        assert wl.name == f"{job_id}.worklist.txt"

    @pytest.mark.parametrize(
        "hostile",
        [
            "../../../../etc/passwd",
            "/etc/passwd",
            "..",
            "a/b",
            "job-7",  # merely non-UUID: not a job id, so not trusted either
        ],
    )
    def test_a_run_id_that_is_not_a_uuid_cannot_shape_the_path(
        self, tmp_path: Path, hostile: str
    ) -> None:
        """CodeQL py/path-injection, two sinks: ``run_id`` reached a filename from the request layer.

        The fix rebuilds the stem from a PARSED ``uuid.UUID`` rather than inspecting the string, so
        the taint cannot survive. Asserted as containment — the file must land inside the jobs dir
        with a UUID stem whatever it is handed.
        """
        import uuid as _uuid

        corpus, op = _corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus, op, run_id=hostile, pipeline_stage="relabel_only", reprocess_episode_ids=["x"]
        )
        wl = _worklist_path(argv)
        assert (
            wl.parent.resolve() == (corpus / ".viewer" / "jobs").resolve()
        ), "escaped the jobs dir"
        stem = wl.name.removesuffix(".worklist.txt")
        _uuid.UUID(stem)  # raises unless the stem is a real UUID
        assert hostile not in wl.name

    def test_it_pairs_with_reprocess_existing_only(self, tmp_path: Path) -> None:
        """Both flags are needed: one restricts to on-disk, the other to the named ids."""
        corpus, op = _corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus,
            op,
            run_id="job-2",
            pipeline_stage="retranscript_only",
            reprocess_episode_ids=["ep-a"],
        )
        assert "--reprocess-existing-only" in argv
        assert "--reprocess-episode-ids" in argv

    def test_a_comma_separated_string_is_accepted(self, tmp_path: Path) -> None:
        """Convenience for a query parameter — one string, many ids."""
        corpus, op = _corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus,
            op,
            run_id="job-3",
            pipeline_stage="rederive_only",
            reprocess_episode_ids="ep-1,ep-2,ep-3",
        )
        assert _worklist_path(argv).read_text(encoding="utf-8").split() == ["ep-1", "ep-2", "ep-3"]

    def test_fifty_ids_survive_intact(self, tmp_path: Path) -> None:
        """The real repair size. Nothing may be truncated on the way through."""
        corpus, op = _corpus(tmp_path)
        ids = [f"ep-{i:02d}" for i in range(50)]
        argv = build_pipeline_argv(
            corpus,
            op,
            run_id="job-50",
            pipeline_stage="retranscript_only",
            reprocess_episode_ids=ids,
        )
        assert _worklist_path(argv).read_text(encoding="utf-8").split() == ids


class TestNoWorkListIsUnchangedBehaviour:
    def test_omitting_ids_adds_no_flag_and_writes_no_file(self, tmp_path: Path) -> None:
        corpus, op = _corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus, op, run_id="job-4", feed_url=_FEED, pipeline_stage="retranscript_only"
        )
        assert "--reprocess-episode-ids" not in argv
        assert not (corpus / ".viewer" / "jobs").exists()

    def test_an_empty_string_is_treated_as_absent(self, tmp_path: Path) -> None:
        corpus, op = _corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus, op, run_id="job-5", pipeline_stage="rederive_only", reprocess_episode_ids=""
        )
        assert "--reprocess-episode-ids" not in argv


class TestIdsThatWouldWIDENTheRepairAreRejected:
    """Rejected, not sanitised. A cleaned-up id silently changes which episodes are repaired."""

    def test_a_newline_would_split_one_id_into_two(self) -> None:
        with pytest.raises(ValueError, match="illegal character"):
            normalize_reprocess_episode_ids(["ep-a\nep-everything-else"])

    def test_a_hash_would_comment_the_rest_of_the_line_out(self) -> None:
        with pytest.raises(ValueError, match="illegal character"):
            normalize_reprocess_episode_ids(["ep-a#ep-b"])

    def test_an_absurdly_long_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="longer than"):
            normalize_reprocess_episode_ids(["q" * 300])

    def test_a_request_cannot_write_an_unbounded_worklist(self) -> None:
        with pytest.raises(ValueError, match="exceeds the"):
            normalize_reprocess_episode_ids([f"ep-{i}" for i in range(501)])

    def test_duplicates_collapse_but_order_is_kept(self) -> None:
        assert normalize_reprocess_episode_ids(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]

    def test_blank_entries_are_dropped(self) -> None:
        assert normalize_reprocess_episode_ids(["a", "  ", "", "b"]) == ["a", "b"]


class TestTheRouteRefusesAWorkListItCannotHonour:
    """A dropped work-list is the defect's signature — so the route 400s instead of ignoring it."""

    @staticmethod
    def _client(corpus: Path, captured: list[list[str]]):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from podcast_scraper.server.app import create_app

        class _FakeProc:
            pid = 91501

            async def wait(self) -> int:
                return 0

        async def _factory(argv, corpus_root: Path, log_abs: Path):  # noqa: ARG001
            captured.append(list(argv))
            log_abs.parent.mkdir(parents=True, exist_ok=True)
            log_abs.write_bytes(b"fake-log\n")
            return _FakeProc()

        app = create_app(corpus, static_dir=False, enable_jobs_api=True)
        app.state.jobs_subprocess_factory = _factory
        return TestClient(app)

    def test_ids_without_a_reprocess_stage_are_a_400(self, tmp_path: Path) -> None:
        """Accepting this would start a FULL ingest while silently discarding the ids."""
        client = self._client(tmp_path, [])
        r = client.post(
            "/api/jobs", params={"path": str(tmp_path), "reprocess_episode_ids": "ep-a,ep-b"}
        )
        assert r.status_code == 400
        assert "requires a reprocess pipeline_stage" in r.json()["detail"]

    def test_ids_with_a_NON_reprocess_stage_are_a_400(self, tmp_path: Path) -> None:
        client = self._client(tmp_path, [])
        r = client.post(
            "/api/jobs",
            params={
                "path": str(tmp_path),
                "pipeline_stage": "download_only",
                "reprocess_episode_ids": "ep-a",
            },
        )
        assert r.status_code == 400

    def test_an_id_containing_a_newline_is_a_400(self, tmp_path: Path) -> None:
        client = self._client(tmp_path, [])
        r = client.post(
            "/api/jobs",
            params={
                "path": str(tmp_path),
                "pipeline_stage": "retranscript_only",
                "reprocess_episode_ids": "ep-a\nep-b",
            },
        )
        assert r.status_code == 400
        assert "illegal character" in r.json()["detail"]

    def test_a_scoped_reprocess_is_accepted_and_the_ids_reach_the_argv(
        self, tmp_path: Path
    ) -> None:
        """The capability that was missing: repair exactly these episodes via the supported path."""
        captured: list[list[str]] = []
        client = self._client(tmp_path, captured)
        r = client.post(
            "/api/jobs",
            params={
                "path": str(tmp_path),
                "pipeline_stage": "retranscript_only",
                "reprocess_episode_ids": "7ed31c1e-3cf5-467d-a302-b47b014db8c5",
            },
        )
        assert r.status_code == 202, r.text
        client.get("/api/jobs", params={"path": str(tmp_path)})  # drain the kickoff
        assert captured, "subprocess factory was never invoked"
        argv = captured[0]
        assert "--reprocess-episode-ids" in argv
        wl = Path(argv[argv.index("--reprocess-episode-ids") + 1])
        assert wl.read_text(encoding="utf-8").split() == ["7ed31c1e-3cf5-467d-a302-b47b014db8c5"]
