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


def test_grant_mcp_all_grants_the_old_default_and_skips_a_deliberate_revoke(
    tmp_path, capsys
) -> None:
    """Profiles from before the default-on change (2026-10-16) hold mcp_access=false for everyone
    never granted. grant-mcp-all fixes those, but not a user an admin revoked on purpose: the
    audit log's latest admin.user.mcp_access entry decides. Dry run writes nothing."""
    import json as _json

    from podcast_scraper.server.app_user_store import get_or_create_user, get_user, set_mcp_access
    from podcast_scraper.server.app_users_cli import main as cli_main

    old = get_or_create_user(tmp_path, provider="google", subject="old", email="o@x.com", name="O")
    revoked = get_or_create_user(
        tmp_path, provider="google", subject="rev", email="r@x.com", name="R"
    )
    has = get_or_create_user(tmp_path, provider="google", subject="has", email="h@x.com", name="H")
    set_mcp_access(tmp_path, old.user_id, False)  # as written by the old default
    set_mcp_access(tmp_path, revoked.user_id, False)
    audit = [
        {
            "action": "admin.user.mcp_access",
            "by": "admin",
            "user": revoked.user_id,
            "mcp_access": True,
        },
        {
            "action": "admin.user.mcp_access",
            "by": "admin",
            "user": revoked.user_id,
            "mcp_access": False,
        },
        {"event": "mcp.token.issued", "user": old.user_id},
    ]
    (tmp_path / "audit.jsonl").write_text("\n".join(_json.dumps(r) for r in audit) + "\n")

    assert cli_main(["--data-dir", str(tmp_path), "grant-mcp-all"]) == 0
    assert get_user(tmp_path, old.user_id).mcp_access is False  # dry run
    assert "would grant " + old.user_id in capsys.readouterr().out

    assert cli_main(["--data-dir", str(tmp_path), "grant-mcp-all", "--apply"]) == 0
    out = capsys.readouterr()
    assert get_user(tmp_path, old.user_id).mcp_access is True
    assert get_user(tmp_path, revoked.user_id).mcp_access is False
    assert get_user(tmp_path, has.user_id).mcp_access is True
    assert "skip    " + revoked.user_id in out.out
    assert "1 granted, 1 already had it, 1 skipped" in out.err
