"""Root-run admin commands must not leave app-data files the API (uid 1000) cannot write."""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

from podcast_scraper.utils import privileges

_APP = 1000


def _stat(uid: int, gid: int | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(st_uid=uid, st_gid=uid if gid is None else gid)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record every privilege-changing call; nothing here really changes the test process."""
    rec: list[tuple] = []
    monkeypatch.delenv(privileges.KEEP_ROOT_ENV, raising=False)
    monkeypatch.setattr(os, "setgroups", lambda g: rec.append(("setgroups", list(g))))
    monkeypatch.setattr(os, "initgroups", lambda n, g: rec.append(("initgroups", g)))
    monkeypatch.setattr(os, "setgid", lambda g: rec.append(("setgid", g)))
    monkeypatch.setattr(os, "setuid", lambda u: rec.append(("setuid", u)))
    monkeypatch.setattr(os, "chown", lambda p, u, g, **kw: rec.append(("chown", Path(p), u, g)))
    return rec


def _as_root(monkeypatch: pytest.MonkeyPatch, owners: dict[Path, int]) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    def fake_lstat(path: Path) -> types.SimpleNamespace:
        if Path(path) in owners:
            return _stat(owners[Path(path)])
        raise FileNotFoundError(path)

    monkeypatch.setattr(privileges, "_lstat", fake_lstat)


def test_drop_as_non_root_never_changes_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    privileges.drop_root_to_owner(tmp_path)
    assert calls == []


def test_drop_as_root_sets_gid_then_uid_to_the_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: list,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _as_root(monkeypatch, {tmp_path: _APP})
    privileges.drop_root_to_owner(tmp_path)
    names = [c[0] for c in calls]
    assert names.index("setgid") < names.index("setuid")
    assert ("setgid", _APP) in calls and ("setuid", _APP) in calls
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "uid 1000" in err


def test_keep_root_env_skips_the_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list
) -> None:
    _as_root(monkeypatch, {tmp_path: _APP})
    monkeypatch.setenv(privileges.KEEP_ROOT_ENV, "1")
    privileges.drop_root_to_owner(tmp_path)
    assert calls == []


def test_drop_skips_a_root_owned_or_missing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list
) -> None:
    _as_root(monkeypatch, {tmp_path: 0})
    privileges.drop_root_to_owner(tmp_path)
    privileges.drop_root_to_owner(tmp_path / "missing")
    assert calls == []


def test_match_owner_chowns_new_root_owned_entries_to_the_data_dir_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list
) -> None:
    data = tmp_path / "appdata"
    user_dir = data / "users" / "u1"
    profile = user_dir / "profile.json"
    _as_root(
        monkeypatch,
        {profile: 0, user_dir: 0, data / "users": 0, data: _APP, tmp_path: 0},
    )
    privileges.match_data_owner(profile)
    chowned = {c[1] for c in calls if c[0] == "chown"}
    assert chowned == {profile, user_dir, data / "users"}
    assert all(c[2:] == (_APP, _APP) for c in calls if c[0] == "chown")


def test_match_owner_leaves_files_someone_else_owns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list
) -> None:
    f = tmp_path / "f.json"
    _as_root(monkeypatch, {f: _APP, tmp_path: 0})
    privileges.match_data_owner(f)
    assert calls == []


def test_match_owner_is_a_noop_when_no_ancestor_is_non_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list
) -> None:
    f = tmp_path / "f.json"
    _as_root(monkeypatch, {f: 0, tmp_path: 0})
    privileges.match_data_owner(f)
    assert calls == []


def test_match_owner_as_non_root_never_chowns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    privileges.match_data_owner(tmp_path / "x")
    assert calls == []


def test_owned_file_lock_chowns_the_lock_file_it_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list
) -> None:
    lock = tmp_path / "users" / "u1" / ".profile.lock"
    lock.parent.mkdir(parents=True)
    _as_root(monkeypatch, {lock: 0, lock.parent: 0, tmp_path / "users": _APP})
    with privileges.OwnedFileLock(str(lock), timeout=1):
        pass
    assert ("chown", lock, _APP, _APP) in calls
