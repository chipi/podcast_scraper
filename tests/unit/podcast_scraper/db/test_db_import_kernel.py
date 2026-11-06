"""``db import --from-files`` for the second moment's stores (ADR-169, E02 PR 2).

The data dir below has the shape prod's had on 2026-11-05 (the prod session's listing: names and
counts only): per-user comms / push / preferences / an Apple token, the dead-token push backup, root
``access_policy.json`` and ``audit.jsonl``, ``outbox/`` envelopes with their lock files, one
magic-link throttle marker and one used token with its dot-lock, the release file in its own dir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import pytest
from sqlalchemy import select, update

from podcast_scraper.db import ro_engine, rw_engine
from podcast_scraper.db.cli_handlers import parse_db_argv, run_db_cli
from podcast_scraper.db.schema import comms, outbox, users
from podcast_scraper.extensions import use_extensions
from podcast_scraper.server import (
    app_access_store,
    app_comms_store,
    app_graph_telemetry,
    app_magic_link,
    app_outbox_store,
    app_push_store,
    app_release_store,
    app_user_preferences,
)
from podcast_scraper.server.app_account_deletion import apple_refresh_token, delete_account
from podcast_scraper.server.app_audit import read_audit
from podcast_scraper.server.app_user_store import get_user, user_id_for
from tests._database import worker_database_url

LOG = logging.getLogger("test_db_import_kernel")

A = user_id_for("google", "a")
B = user_id_for("google", "b")


@pytest.fixture(autouse=True)
def _database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    monkeypatch.setenv("PODCAST_DATABASE_URL", url)


@pytest.fixture(autouse=True)
def _platform_only():
    with use_extensions([]):
        yield


def _write(path: Path, content: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), "utf-8")
    return path


def _envelope_file(data_dir: Path, envelope: dict, status: str = "pending", at: int = 100) -> Path:
    name = hashlib.sha256(envelope["id"].encode("utf-8")).hexdigest()
    (data_dir / "outbox" / f"{name}.lock").parent.mkdir(parents=True, exist_ok=True)
    (data_dir / "outbox" / f"{name}.lock").write_text("")
    record = {"envelope": envelope, "status": status, "detail": None, "updated_at": at}
    return _write(data_dir / "outbox" / f"{name}.json", record)


def _prod_shaped(data_dir: Path) -> None:
    for uid, sub in ((A, "a"), (B, "b")):
        _write(
            data_dir / "users" / uid / "profile.json",
            {"email": f"{sub}@x.com", "name": sub, "provider": "google", "subject": sub},
        )
    _write(
        data_dir / "users" / A / "comms.json",
        {
            "types": {"digest": {"email": True}},
            "timezone": "Europe/Belgrade",
            "unsubscribe_ref": "r1",
        },
    )
    _write(data_dir / "users" / B / "comms.json", {"types": {"product": {"push": True}}})
    _write(
        data_dir / "users" / A / "push_subscriptions.json",
        [{"endpoint": "https://p/1", "keys": {}}, {"endpoint": "https://p/2"}, {"no": "endpoint"}],
    )
    _write(data_dir / "users" / A / "push_subscriptions.json.bak-dead-tokens-20261007083919", "[]")
    _write(data_dir / "users" / A / "preferences.json", {"theme": "dark", "graph": {"lens": 1}})
    _write(data_dir / "users" / B / "apple_token.json", {"refresh_token": "apple-tok"})
    (data_dir / "users" / A / ".comms.json.lock").write_text("")
    (data_dir / "users" / ".handles.lock").write_text("")
    _write(
        data_dir / "access_policy.json",
        {"mode": "allowlist", "allowed_emails": ["b@x.com", "a@x.com"], "allowed_domains": []},
    )
    _write(
        data_dir / "audit.jsonl",
        "\n".join(json.dumps(r) for r in ({"ts": 1, "event": "a"}, {"ts": 2, "event": "b"}))
        + "\nnot json\n",
    )
    _envelope_file(
        data_dir,
        {
            "id": "digest-1",
            "channel": "email",
            "type": "digest",
            "user_id": A,
            "recipient": {"email": "A@x.com"},
        },
    )
    _envelope_file(
        data_dir,
        {
            "id": "auth-1",
            "channel": "email",
            "type": "auth_link",
            "user_id": "",
            "recipient": {"email": "new@x.com"},
        },
        status="delivered",
        at=50,
    )
    used = _write(data_dir / "magic_link_used" / "B4Xm6qz_zE0FRCJivV2G3A.json", '{"used_at": 1700}')
    (used.parent / ".B4Xm6qz_zE0FRCJivV2G3A.lock").write_text("")
    marker = _write(
        data_dir / "magic_link_recent" / (app_magic_link.email_hash("q@x.com") + ".txt"), "9"
    )
    os.utime(marker, (1800, 1800))


def _import(capsys: pytest.CaptureFixture[str], data_dir: Path, *extra: str) -> tuple[int, str]:
    args = parse_db_argv(["import", "--from-files", str(data_dir), "--plane", "player", *extra])
    code = run_db_cli(args, LOG)
    return code, capsys.readouterr().out


def test_a_prod_shaped_dir_imports_record_for_record_with_nothing_unclaimed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prod_shaped(tmp_path)

    code, out = _import(capsys, tmp_path)

    assert code == 0, out
    assert "UNCLAIMED: 0" in out
    assert "still a file (not yet migrated): 0" in out
    for line in (
        "store users: read 2, inserted 2",
        "store comms: read 2, inserted 2",
        "store push: read 1, inserted 1",
        "store user_preferences: read 1, inserted 1",
        "store apple_tokens: read 1, inserted 1",
        "store access_policy: read 1, inserted 1",
        "store outbox: read 2, inserted 2",
        "store magic_link: read 2, inserted 2",
        "store audit: read 1, inserted 2",
    ):
        assert line in out, line
    # What the stores now return is what they returned from the files.
    assert app_comms_store.get_comms(tmp_path, A)["types"]["digest"]["email"] is True
    assert app_comms_store.get_comms(tmp_path, A)["timezone"] == "Europe/Belgrade"
    assert [s["endpoint"] for s in app_push_store.list_subscriptions(tmp_path, A)] == [
        "https://p/1",
        "https://p/2",
    ]
    assert app_user_preferences.get_preferences(tmp_path, A) == {
        "theme": "dark",
        "graph": {"lens": 1},
    }
    assert apple_refresh_token(tmp_path, B) == "apple-tok"
    policy = app_access_store.load_policy(tmp_path)
    assert policy is not None and policy.is_allowed("a@x.com") and not policy.is_allowed("c@x.com")
    assert [r["event"] for r in read_audit(tmp_path / "audit.jsonl")] == ["a", "b"]
    # A used link stays used across the switch; the throttle keeps its time.
    assert app_magic_link.consume(tmp_path, "B4Xm6qz_zE0FRCJivV2G3A", now=1700) is False
    assert app_magic_link.request_throttled("q@x.com", now=1830, min_interval_s=60) is True
    pending = app_outbox_store.list_pending(tmp_path, channel="email", now=200)
    assert [e["id"] for e in pending] == ["digest-1"]
    # The one-click link finds its user by the imported ref, and the digest then stops.
    assert app_comms_store.unsubscribe(tmp_path, "r1") is True
    assert app_outbox_store.list_pending(tmp_path, channel="email", now=200) == []


def test_a_second_run_inserts_nothing_and_verifies_everything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prod_shaped(tmp_path)
    assert _import(capsys, tmp_path)[0] == 0
    code, out = _import(capsys, tmp_path)
    assert code == 0, out
    assert "inserted 1" not in out and "inserted 2" not in out
    assert "mismatched 1" not in out


def test_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _prod_shaped(tmp_path)
    code, out = _import(capsys, tmp_path, "--dry-run")
    assert code == 0, out
    assert "store outbox: read 2, would insert 2" in out
    with ro_engine().connect() as conn:
        assert conn.execute(select(outbox)).all() == []
        assert conn.execute(select(users)).all() == []


def test_a_changed_row_is_a_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _prod_shaped(tmp_path)
    assert _import(capsys, tmp_path)[0] == 0
    with rw_engine().begin() as conn:
        conn.execute(update(comms).where(comms.c.user_id == A).values(unsubscribe_ref="other"))
    code, out = _import(capsys, tmp_path, "--only", "comms")
    assert code == 1 and f"MISMATCH {A}" in out


def test_a_per_user_file_without_an_account_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ghost = "u_" + "9" * 24
    _write(tmp_path / "users" / ghost / "preferences.json", {"theme": "dark"})
    code, out = _import(capsys, tmp_path)
    assert code == 1 and f"MISMATCH {ghost}: no account row" in out


def test_an_unreadable_file_is_reported_never_imported_as_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path / "users" / A / "profile.json", {"provider": "google", "subject": "a"})
    _write(tmp_path / "users" / A / "comms.json", "{not json")
    _write(tmp_path / "access_policy.json", "[1, 2]")
    code, out = _import(capsys, tmp_path)
    assert code == 1
    assert f"MISMATCH {A}: comms.json:" in out
    assert "MISMATCH access_policy: unreadable" in out


def test_an_outbox_file_under_a_name_that_is_not_its_ids_hash_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tmp_path / "outbox" / ("0" * 64 + ".json"),
        {"envelope": {"id": "x", "channel": "email"}, "status": "pending", "updated_at": 1},
    )
    code, out = _import(capsys, tmp_path)
    assert code == 1 and "file name is not the id's hash" in out


def test_the_release_dir_imports_into_the_shared_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prod keeps ``player_release.json`` alone in the shared release dir; it is imported there,
    with ``--only releases``, once, for both planes."""
    release_dir = tmp_path / "player-release"
    _write(release_dir / "player_release.json", {"player_version": "1.0.4"})
    code, out = _import(capsys, release_dir, "--only", "releases")
    assert code == 0, out
    assert "store releases: read 1, inserted 1" in out and "UNCLAIMED: 0" in out
    assert app_release_store.load_released_versions(release_dir) == {"player": "1.0.4"}
    # The newer file wins over the older one, as the store read them.
    newer = tmp_path / "newer"
    _write(newer / "app_releases.json", {"versions": {"player": "2.0", "news": "0.1", "bad": "x"}})
    _write(newer / "player_release.json", {"player_version": "1.0.4"})
    code, out = _import(capsys, newer, "--only", "releases", "--dry-run")
    assert code == 1 and "MISMATCH player" in out  # 2.0 in the file, 1.0.4 already stored


def test_signed_out_graph_events_import_as_no_account(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path / "users" / A / "profile.json", {"provider": "google", "subject": "a"})
    _write(tmp_path / "users" / A / "graph_events.jsonl", '{"action": "a"}\nbad\n{"action": "b"}\n')
    _write(tmp_path / "users" / "anon" / "graph_events.jsonl", '{"action": "z"}\n')
    code, out = _import(capsys, tmp_path)
    assert code == 0, out
    assert "store graph_events: read 2, inserted 3" in out
    assert [e["action"] for e in app_graph_telemetry.read_events(tmp_path, A)] == ["a", "b"]
    assert [e["action"] for e in app_graph_telemetry.read_events(tmp_path, "anon")] == ["z"]


def test_account_deletion_after_the_import_removes_every_row_that_names_the_account(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prod_shaped(tmp_path)
    assert _import(capsys, tmp_path)[0] == 0
    user = get_user(tmp_path, A)
    assert user is not None

    report = delete_account(tmp_path, user)

    assert report.outbox_envelopes == 1 and report.user_dir is True
    assert get_user(tmp_path, A) is None
    assert app_comms_store.unsubscribe(tmp_path, "r1") is False
    assert app_push_store.list_subscriptions(tmp_path, A) == []
    assert app_user_preferences.get_preferences(tmp_path, A) == {}
    assert app_outbox_store.list_pending(tmp_path, channel="email", now=200) == []
    # The other account is untouched, and the audit trail stays (#2273).
    assert apple_refresh_token(tmp_path, B) == "apple-tok"
    assert len(read_audit(tmp_path / "audit.jsonl")) == 2
