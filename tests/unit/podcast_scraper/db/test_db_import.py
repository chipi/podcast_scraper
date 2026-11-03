"""``db import --from-files``: the users importer and the whole-directory inventory (ADR-169)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
from sqlalchemy import update

from podcast_scraper.cli import parse_args
from podcast_scraper.db import rw_engine
from podcast_scraper.db.cli_handlers import parse_db_argv, run_db_cli
from podcast_scraper.db.importers import registry
from podcast_scraper.db.importers.users import read_profile
from podcast_scraper.db.schema import users
from podcast_scraper.server.app_user_store import get_user, list_users, user_id_for
from tests._database import worker_database_url

LOG = logging.getLogger("test_db_import")


@pytest.fixture(autouse=True)
def _database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    monkeypatch.setenv("PODCAST_DATABASE_URL", url)


def _profile(data_dir: Path, uid: str, **fields: object) -> Path:
    doc = {"email": "a@x.com", "name": "A", "provider": "google", "subject": uid, **fields}
    path = data_dir / "users" / uid / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _import(
    capsys: pytest.CaptureFixture[str], data_dir: Path, *extra: str, plane: str = "player"
) -> tuple[int, str]:
    args = parse_db_argv(["import", "--from-files", str(data_dir), "--plane", plane, *extra])
    code = run_db_cli(args, LOG)
    return code, capsys.readouterr().out


def test_round_trip_is_record_for_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b, c = (user_id_for("google", s) for s in ("s1", "s2", "s3"))
    _profile(
        tmp_path,
        a,
        username="alpha",
        image="https://cdn/a.png",
        role="admin",
        mcp_access=False,
        disabled=True,
        analytics_id="11111111-1111-4111-8111-111111111111",
    )
    _profile(tmp_path, b, username="beta", subject="s2")
    # Older shapes: no username / role / analytics_id / mcp_access keys at all.
    path = tmp_path / "users" / c / "profile.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"email": "o@x.com", "provider": "google", "subject": "s3"}))

    code, out = _import(capsys, tmp_path)

    assert code == 0, out
    assert "store users: read 3, inserted 3, already present 0, mismatched 0" in out
    for uid in (a, b, c):
        assert get_user(tmp_path, uid) == read_profile(
            tmp_path / "users" / uid / "profile.json", uid
        )
    assert [u.user_id for u in list_users(tmp_path)] == sorted([a, b, c])
    # The ids are the directory names, verbatim.
    assert get_user(tmp_path, a).username == "alpha"  # type: ignore[union-attr]


def test_a_second_run_inserts_nothing_and_still_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _profile(tmp_path, user_id_for("google", "s1"), username="alpha", subject="s1")
    assert _import(capsys, tmp_path)[0] == 0
    code, out = _import(capsys, tmp_path)
    assert code == 0
    assert "inserted 0, already present 1, mismatched 0" in out


def test_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _profile(tmp_path, user_id_for("google", "s1"), username="alpha", subject="s1")
    code, out = _import(capsys, tmp_path, "--dry-run")
    assert code == 0
    assert "would insert 1" in out
    assert list_users(tmp_path) == []


def test_a_row_that_differs_from_its_file_is_listed_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uid = user_id_for("google", "s1")
    _profile(tmp_path, uid, username="alpha", subject="s1", name="Original")
    assert _import(capsys, tmp_path)[0] == 0
    with rw_engine().begin() as conn:
        conn.execute(
            update(users).where(users.c.user_id == uid).values(name="Edited", disabled=True)
        )

    code, out = _import(capsys, tmp_path)

    assert code == 1
    assert f"MISMATCH {uid}" in out
    assert "name: file='Original' database='Edited'" in out
    assert "disabled: file=False database=True" in out


def test_a_handle_two_files_claim_is_a_listed_conflict_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = user_id_for("google", "s1"), user_id_for("google", "s2")
    _profile(tmp_path, first, username="same", subject="s1")
    _profile(tmp_path, second, username="same", subject="s2")
    code, out = _import(capsys, tmp_path)
    assert code == 1
    assert "MISMATCH" in out and "username already belongs to" in out
    assert len(list_users(tmp_path)) == 1


def test_an_unreadable_profile_and_a_foreign_directory_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uid = user_id_for("google", "s1")
    (tmp_path / "users" / uid).mkdir(parents=True)
    (tmp_path / "users" / uid / "profile.json").write_text("{not json")
    _profile(tmp_path, "not-a-user-id")
    code, out = _import(capsys, tmp_path)
    assert code == 1
    assert f"MISMATCH {uid}: profile.json is unreadable" in out
    assert "MISMATCH not-a-user-id: directory name is not a user id" in out


def test_an_unknown_file_under_a_user_directory_is_unclaimed_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uid = user_id_for("google", "s1")
    _profile(tmp_path, uid, username="alpha", subject="s1")
    (tmp_path / "users" / uid / "mystery.txt").write_text("?")

    code, out = _import(capsys, tmp_path)
    assert code == 1
    assert f"UNCLAIMED users/{uid}/mystery.txt" in out

    dry_code, dry_out = _import(capsys, tmp_path, "--dry-run")
    assert dry_code == 1 and f"UNCLAIMED users/{uid}/mystery.txt" in dry_out


def test_known_not_yet_migrated_files_are_listed_and_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uid = user_id_for("google", "s1")
    _profile(tmp_path, uid, username="alpha", subject="s1")
    udir = tmp_path / "users" / uid
    (udir / "queue.json").write_text("[]")
    (udir / "listen_events.jsonl").write_text("")
    (udir / ".queue.lock").write_text("")
    (tmp_path / "audit.jsonl").write_text("")
    (tmp_path / "outbox").mkdir()
    (tmp_path / "outbox" / ("ab" * 32 + ".json")).write_text("{}")

    code, out = _import(capsys, tmp_path)

    assert code == 0, out
    assert "imported by a store: 1" in out
    assert "still a file (not yet migrated): 4" in out
    assert "ignored (lock files): 1" in out
    assert "UNCLAIMED: 0" in out


def test_only_restricts_the_importers_not_the_inventory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _profile(tmp_path, user_id_for("google", "s1"), username="alpha", subject="s1")
    (tmp_path / "stray.bin").write_text("?")
    code, out = _import(capsys, tmp_path, "--only", "users")
    assert code == 1 and "UNCLAIMED stray.bin" in out
    assert _import(capsys, tmp_path, "--only", "nope")[0] == 1


def test_the_operator_plane_is_a_separate_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _profile(tmp_path, user_id_for("google", "s1"), username="alpha", subject="s1")
    assert _import(capsys, tmp_path, plane="operator")[0] == 0
    assert list_users(tmp_path) == []  # the player plane, which this process serves, is untouched


def test_the_subcommand_is_wired_into_the_main_parser() -> None:
    args = parse_args(["db", "status", "--plane", "operator"])
    assert (args.command, args.db_subcommand, args.plane) == ("db", "status", "operator")


def test_status_exits_two_when_behind_and_upgrade_fixes_it(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy import create_engine, text

    from podcast_scraper.db.settings import normalize_url

    url = os.environ["PODCAST_DATABASE_URL"]
    eng = create_engine(normalize_url(url))
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
    eng.dispose()

    behind = run_db_cli(parse_db_argv(["status", "--plane", "operator"]), LOG)
    assert behind == 2 and "current none" in capsys.readouterr().out
    assert run_db_cli(parse_db_argv(["upgrade", "--plane", "operator"]), LOG) == 0
    assert run_db_cli(parse_db_argv(["status", "--plane", "operator"]), LOG) == 0


def test_the_registry_has_the_users_importer() -> None:
    assert list(registry()) == ["users"]
