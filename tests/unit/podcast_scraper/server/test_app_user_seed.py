"""Unit tests for fixed-roster user seeding (#1128)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server.app_user_seed import (
    _mock_identity,
    seed_from_env,
    seed_users,
    seeds_from_env,
)
from podcast_scraper.server.app_user_store import get_user, list_users, user_id_for

ROSTER = [
    {"hint": "ada-admin", "name": "Ada Admin", "role": "admin"},
    {"hint": "cory-creator", "name": "Cory Creator", "role": "creator"},
    {"hint": "cleo-creator", "name": "Cleo Creator", "role": "creator"},
    {"hint": "pat-player", "name": "Pat Player", "role": "listener"},
    {"hint": "pip-player", "name": "Pip Player", "role": "listener"},
]


def test_mock_identity_matches_an_as_login() -> None:
    # A seed must resolve to the SAME identity as a ``?as=<hint>`` mock login.
    assert _mock_identity("ada-admin") == ("mock", "e2e-ada-admin", "ada-admin@e2e.local")
    # ``_safe_hint`` sanitisation is applied (so a messy hint still hooks cleanly).
    assert _mock_identity("Ada Admin!")[1] == _mock_identity("adaadmin")[1]


def test_seed_users_creates_the_roster_with_roles(tmp_path: Path) -> None:
    assert seed_users(tmp_path, ROSTER) == 5
    by_role = sorted((u.email, u.role) for u in list_users(tmp_path))
    assert by_role == [
        ("ada-admin@e2e.local", "admin"),
        ("cleo-creator@e2e.local", "creator"),
        ("cory-creator@e2e.local", "creator"),
        ("pat-player@e2e.local", "listener"),
        ("pip-player@e2e.local", "listener"),
    ]
    # 1 admin / 2 creators / 2 listeners
    roles = sorted(u.role for u in list_users(tmp_path))
    assert roles == ["admin", "creator", "creator", "listener", "listener"]


def test_seed_users_is_idempotent_and_non_destructive(tmp_path: Path) -> None:
    seed_users(tmp_path, ROSTER)
    # an admin changes a seeded user's role at runtime
    from podcast_scraper.server.app_user_store import set_role

    uid = user_id_for("mock", "e2e-pat-player")
    set_role(tmp_path, uid, "creator")
    # re-seeding creates nothing new and leaves the runtime change intact
    assert seed_users(tmp_path, ROSTER) == 0
    pat = get_user(tmp_path, uid)
    assert pat is not None and pat.role == "creator"


def test_seed_users_skips_bad_entries(tmp_path: Path) -> None:
    bad: list = [{"hint": "", "role": "admin"}, {"name": "no hint"}, "not-a-dict", {"hint": "ok"}]
    created = seed_users(tmp_path, bad)
    assert created == 1
    assert {u.email for u in list_users(tmp_path)} == {"ok@e2e.local"}


def test_seeds_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_SEED_USERS_FILE", raising=False)
    assert seeds_from_env() == []
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(ROSTER), encoding="utf-8")
    monkeypatch.setenv("APP_SEED_USERS_FILE", str(path))
    assert len(seeds_from_env()) == 5
    # missing file / malformed JSON → [] (never raises)
    monkeypatch.setenv("APP_SEED_USERS_FILE", str(tmp_path / "nope.json"))
    assert seeds_from_env() == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("APP_SEED_USERS_FILE", str(bad))
    assert seeds_from_env() == []


def test_seed_from_env_wires_file_to_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(ROSTER), encoding="utf-8")
    monkeypatch.setenv("APP_SEED_USERS_FILE", str(path))
    data_dir = tmp_path / "appdata"
    assert seed_from_env(data_dir) == 5
    assert seed_from_env(None) == 0  # no data dir → no-op
    assert len(list_users(data_dir)) == 5
