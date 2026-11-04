"""Every app-data write path chowns what root creates to the data dir's owner (and only as root)."""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

from podcast_scraper.server import app_user_store
from podcast_scraper.server.atomic_write import atomic_write_text
from podcast_scraper.utils import privileges

_APP = 1000


@pytest.fixture
def chowned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Path]:
    """Run as 'root' with the data dir owned by uid 1000 and everything newly written root-owned."""
    rec: list[Path] = []
    real = os.lstat

    def fake_lstat(path: Path) -> object:
        st = real(path)
        uid = _APP if Path(path) == tmp_path else 0
        return types.SimpleNamespace(st_uid=uid, st_gid=uid, st_mode=st.st_mode)

    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(privileges, "_lstat", fake_lstat)
    monkeypatch.setattr(os, "chown", lambda p, u, g, **kw: rec.append(Path(p)))
    return rec


def test_atomic_write_chowns_the_file_and_the_dirs_it_made(
    tmp_path: Path, chowned: list[Path]
) -> None:
    target = tmp_path / "users" / "u1" / "x.json"
    atomic_write_text(target, "{}")
    assert target in chowned and target.parent in chowned


def test_atomic_write_as_non_root_never_chowns(
    tmp_path: Path, chowned: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    atomic_write_text(tmp_path / "x.json", "{}")
    assert chowned == []


def test_profile_write_and_its_lock_are_chowned(tmp_path: Path, chowned: list[Path]) -> None:
    user = app_user_store.get_or_create_user(
        tmp_path, provider="google", subject="s", email="a@x.com", name="A"
    )
    app_user_store.set_mcp_access(tmp_path, user.user_id, True)
    udir = tmp_path / "users" / user.user_id
    assert udir / "profile.json" in chowned
    assert udir / ".profile.lock" in chowned


def test_audit_append_is_chowned(tmp_path: Path, chowned: list[Path]) -> None:
    from podcast_scraper.server.app_audit import append_audit

    append_audit(tmp_path / "audit.jsonl", {"action": "x"})
    assert tmp_path / "audit.jsonl" in chowned


def test_graph_event_append_is_chowned(tmp_path: Path, chowned: list[Path]) -> None:
    from podcast_scraper.server import app_graph_telemetry

    app_graph_telemetry.record_events(tmp_path, "u1", [{"action": "open"}])
    assert app_graph_telemetry._events_path(tmp_path, "u1") in chowned


def test_apple_token_write_is_chowned(tmp_path: Path, chowned: list[Path]) -> None:
    from podcast_scraper.server import app_account_deletion

    app_account_deletion.store_apple_refresh_token(tmp_path, "u_" + "a" * 24, "tok")
    assert any(p.parent == tmp_path / "users" / ("u_" + "a" * 24) for p in chowned)


def test_magic_link_throttle_marker_is_chowned(tmp_path: Path, chowned: list[Path]) -> None:
    from podcast_scraper.server.routes import app_auth

    app_auth._throttled(tmp_path, "a@x.com", now=10)
    assert app_auth._recent_request_marker(tmp_path, "a@x.com") in chowned
