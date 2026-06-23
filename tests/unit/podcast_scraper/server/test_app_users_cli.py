"""Unit tests for the operator user-admin CLI (#1064)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.server import app_users_cli
from podcast_scraper.server.app_user_store import get_or_create_user, get_user, User


def _seed(data_dir: Path) -> User:
    return get_or_create_user(
        data_dir, provider="google", subject="s1", email="a@x.com", name="Alice"
    )


def test_list(tmp_path: Path, capsys) -> None:
    _seed(tmp_path)
    assert app_users_cli.main(["--data-dir", str(tmp_path), "list"]) == 0
    assert "a@x.com" in capsys.readouterr().out


def test_disable_then_enable(tmp_path: Path) -> None:
    user = _seed(tmp_path)
    assert app_users_cli.main(["--data-dir", str(tmp_path), "disable", user.user_id]) == 0
    loaded = get_user(tmp_path, user.user_id)
    assert loaded is not None and loaded.disabled is True
    assert app_users_cli.main(["--data-dir", str(tmp_path), "enable", user.user_id]) == 0
    loaded = get_user(tmp_path, user.user_id)
    assert loaded is not None and loaded.disabled is False


def test_delete(tmp_path: Path) -> None:
    user = _seed(tmp_path)
    assert app_users_cli.main(["--data-dir", str(tmp_path), "delete", user.user_id]) == 0
    assert get_user(tmp_path, user.user_id) is None


def test_export(tmp_path: Path, capsys) -> None:
    user = _seed(tmp_path)
    assert app_users_cli.main(["--data-dir", str(tmp_path), "export", user.user_id]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["email"] == "a@x.com"
    assert data["user_id"] == user.user_id


def test_unknown_user_exits_nonzero(tmp_path: Path) -> None:
    assert app_users_cli.main(["--data-dir", str(tmp_path), "disable", "u_nope"]) == 1
