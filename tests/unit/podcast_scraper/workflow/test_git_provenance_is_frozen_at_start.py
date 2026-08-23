"""Provenance must name the code that RAN, not whatever HEAD drifted to mid-run.

ADR-132 makes ``git_sha`` the exact-code backstop — the field you consult when an artifact looks
wrong and you need to know which commit produced it. It is therefore read exactly when something
is already suspicious, so a value that can disagree with reality is worse than no value at all.

The defect these tests lock out (observed 2026-08-16, 14-episode acceptance run): the probe
shelled out to git on every manifest write, so a commit landing mid-run split the manifests
across two SHAs even though one commit's code executed all of them.

The reason a cache is CORRECT rather than merely convenient: Python imports its modules once at
startup. An edit — or a production deploy — landing mid-run does not change the code the running
process executes. So the true answer is fixed at process start, and re-reading the working tree
later answers a different question than the field asks.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from podcast_scraper.workflow import run_manifest
from podcast_scraper.workflow.processing_manifest import git_ground_truth


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test starts with an empty capture and leaves one behind for nobody."""
    run_manifest.reset_git_info_cache()
    yield
    run_manifest.reset_git_info_cache()


def _probe_returning(values: List[tuple]) -> Any:
    """A fake probe that hands out a different answer on each call, then repeats the last."""
    calls = {"n": 0}

    def fake() -> tuple:
        idx = min(calls["n"], len(values) - 1)
        calls["n"] += 1
        return values[idx]

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def test_a_commit_landing_mid_run_does_not_change_recorded_provenance(monkeypatch):
    """THE regression. Two manifests written either side of a commit must agree."""
    probe = _probe_returning([("e055286aaa", "main", False), ("2ceb653bbb", "main", False)])
    monkeypatch.setattr(run_manifest, "_probe_git_info", probe)

    first = git_ground_truth()  # episode 1, before the commit
    second = git_ground_truth()  # episode 2, after a commit landed

    assert first == second, (
        "git_sha changed between two episodes of the same run — the probe is re-reading the "
        "working tree instead of reporting the code the process loaded at startup"
    )
    assert first["git_sha"] == "e055286"
    assert probe.calls["n"] == 1, "git was shelled out to more than once"


def test_run_manifest_and_episode_manifests_report_the_same_commit(monkeypatch):
    """One cache, not two: the run-level and per-episode provenance must not straddle a commit."""
    probe = _probe_returning([("e055286aaa", "main", False), ("2ceb653bbb", "main", True)])
    monkeypatch.setattr(run_manifest, "_probe_git_info", probe)

    run_level_sha, _branch, run_level_dirty = run_manifest._get_git_info()
    episode_level = git_ground_truth()

    assert episode_level["git_sha"] == run_level_sha[:7]
    assert episode_level["git_dirty"] is bool(run_level_dirty)


def test_the_dirty_flag_is_frozen_too(monkeypatch):
    """A clean tree that gets edited mid-run must not retroactively mark earlier episodes dirty."""
    probe = _probe_returning([("abc1234def", "main", False), ("abc1234def", "main", True)])
    monkeypatch.setattr(run_manifest, "_probe_git_info", probe)

    assert git_ground_truth()["git_dirty"] is False
    assert git_ground_truth()["git_dirty"] is False


def test_concurrent_episodes_all_see_one_answer(monkeypatch):
    """Episodes run concurrently; the capture must be race-free, not merely cached."""
    barrier = threading.Barrier(8)
    probe = _probe_returning([("aaaaaaa111", "main", False), ("bbbbbbb222", "main", False)])
    monkeypatch.setattr(run_manifest, "_probe_git_info", probe)

    seen: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()  # maximise the chance of a real race on first use
        value = git_ground_truth()
        with lock:
            seen.append(value)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 8
    assert all(v == seen[0] for v in seen), f"threads disagreed about provenance: {seen}"
    assert probe.calls["n"] == 1, "the capture raced — git was probed more than once"


def test_a_repo_less_environment_still_reports_cleanly(monkeypatch):
    """No git available (a container built from a tarball) must be None, not a crash."""
    monkeypatch.setattr(run_manifest, "_probe_git_info", lambda: (None, None, False))

    assert git_ground_truth() == {"git_sha": None, "git_dirty": False}


class TestTheBuiltImageHasProvenanceAtAll:
    """In the container production runs, shelling out to git returns NOTHING.

    ``.dockerignore`` excludes ``.git/`` from the build context and the runtime stage never
    installs the git binary, so ``git rev-parse`` raises FileNotFoundError and the probe reports
    (None, None, False). Every manifest the pipeline image wrote recorded ``git_sha: null`` —
    ADR-132's exact-code backstop absent from precisely the artifacts it exists to explain.

    The 14-episode acceptance run showed one clean SHA, which looked like proof the chain
    worked. That run executed from source, where ``.git`` is present. It proved nothing about
    the image, which is why the SHA is now baked in at build time.
    """

    def test_the_baked_in_sha_is_used_when_git_is_unavailable(self, monkeypatch):
        """The container case: no git binary, but the build arg was set."""
        monkeypatch.setenv(run_manifest.GIT_SHA_ENV, "abc1234def5678")
        monkeypatch.setenv(run_manifest.GIT_BRANCH_ENV, "main")
        monkeypatch.setenv(run_manifest.GIT_DIRTY_ENV, "false")
        monkeypatch.setattr(
            run_manifest.subprocess,
            "check_output",
            _explode_as_if_git_were_missing,
        )

        assert git_ground_truth() == {"git_sha": "abc1234", "git_dirty": False}

    def test_a_dirty_build_is_reported_dirty(self, monkeypatch):
        monkeypatch.setenv(run_manifest.GIT_SHA_ENV, "abc1234def5678")
        monkeypatch.setenv(run_manifest.GIT_DIRTY_ENV, "true")

        assert git_ground_truth()["git_dirty"] is True

    def test_an_empty_build_arg_falls_back_to_git(self, monkeypatch):
        """An unparameterised ``docker build`` sets GIT_SHA="" — that must not mean "no SHA"."""
        monkeypatch.setenv(run_manifest.GIT_SHA_ENV, "")
        monkeypatch.setattr(run_manifest, "_probe_git_info", lambda: ("fa11bac0999", "b", False))
        run_manifest.reset_git_info_cache()

        assert git_ground_truth()["git_sha"] == "fa11bac"  # short SHA is 7 chars

    def test_a_source_checkout_is_unaffected(self, monkeypatch):
        """No env vars set — dev boxes and CI keep shelling out to git exactly as before."""
        monkeypatch.delenv(run_manifest.GIT_SHA_ENV, raising=False)
        monkeypatch.delenv(run_manifest.GIT_BRANCH_ENV, raising=False)
        monkeypatch.delenv(run_manifest.GIT_DIRTY_ENV, raising=False)

        assert run_manifest._git_info_from_env() is None

    def test_the_baked_value_is_still_captured_only_once(self, monkeypatch):
        """The env path must not bypass the freeze — os.environ is mutable at runtime too."""
        monkeypatch.setenv(run_manifest.GIT_SHA_ENV, "first1234")
        first = git_ground_truth()
        monkeypatch.setenv(run_manifest.GIT_SHA_ENV, "second567")

        assert git_ground_truth() == first


def _explode_as_if_git_were_missing(*_args, **_kwargs):
    raise FileNotFoundError("git: command not found")


def test_callers_cannot_corrupt_the_shared_capture(monkeypatch):
    """``git_ground_truth`` hands out a fresh dict; mutating it must not poison later writers."""
    monkeypatch.setattr(run_manifest, "_probe_git_info", lambda: ("abc1234def", "main", False))

    first = git_ground_truth()
    first["git_sha"] = "TAMPERED"

    assert git_ground_truth()["git_sha"] == "abc1234"
