"""The build record the images bake and the processes read (``podcast_scraper.build_info``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper import __version__, build_info


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PODCAST_VERSION_FILE", "PODCAST_GIT_SHA", "CL_RELEASE", "PODCAST_RELEASE"):
        monkeypatch.delenv(name, raising=False)


def test_the_baked_file_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "VERSION.json"
    f.write_text(
        json.dumps(
            {
                "component": "api",
                "version": "9.9.9",
                "git_sha": "abc",
                "built_at": "2026-11-05T00:00:00Z",
            }
        )
    )
    monkeypatch.setenv("PODCAST_VERSION_FILE", str(f))
    monkeypatch.setenv("PODCAST_GIT_SHA", "env-sha")
    assert build_info.build_info() == {
        "component": "api",
        "version": "9.9.9",
        "git_sha": "abc",
        "built_at": "2026-11-05T00:00:00Z",
    }


def test_no_file_falls_back_to_the_env_sha_then_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_VERSION_FILE", str(tmp_path / "missing.json"))
    assert build_info.build_info() == {
        "component": None,
        "version": __version__,
        "git_sha": None,
        "built_at": None,
    }
    monkeypatch.setenv("PODCAST_GIT_SHA", "deadbeef")
    assert build_info.build_info()["git_sha"] == "deadbeef"


def test_an_empty_baked_sha_is_none_not_an_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "VERSION.json"
    f.write_text(json.dumps({"version": "1.0", "git_sha": ""}))
    monkeypatch.setenv("PODCAST_VERSION_FILE", str(f))
    assert build_info.build_info()["git_sha"] is None


def test_a_corrupt_file_reads_as_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "VERSION.json"
    f.write_text("{not json")
    monkeypatch.setenv("PODCAST_VERSION_FILE", str(f))
    assert build_info.build_info()["version"] == __version__


def test_the_two_release_ids_stay_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    assert build_info.release() == (None, None)
    monkeypatch.setenv("PODCAST_RELEASE", "sha-f493b65")
    assert build_info.release() == ("sha-f493b65", "PODCAST_RELEASE")
    assert (build_info.deploy_release(), build_info.image_release()) == (None, "sha-f493b65")
    monkeypatch.setenv("CL_RELEASE", "b5a243d")
    assert build_info.release() == ("b5a243d", "CL_RELEASE")
    assert (build_info.deploy_release(), build_info.image_release()) == ("b5a243d", "sha-f493b65")


def test_a_module_record_is_read_from_the_modules_dir(tmp_path: Path) -> None:
    (tmp_path / "player.json").write_text(json.dumps({"git_sha": "c0ffee"}))
    assert build_info.module_version_file("player", tmp_path) == {"git_sha": "c0ffee"}
    assert build_info.module_version_file("absent", tmp_path) == {}
