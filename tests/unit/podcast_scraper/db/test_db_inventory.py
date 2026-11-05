"""The inventory must know every file name the stores write (ADR-169)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from podcast_scraper.db import inventory
from podcast_scraper.extensions import Extension, use_extensions

SERVER = Path(__file__).resolve().parents[4] / "src" / "podcast_scraper" / "server"

#: ``data_dir / "users" / user_id / "name.ext"`` — a per-user file named by a string literal.
_USER_LITERAL = re.compile(r'"users"\s*/\s*\w+\s*/\s*"([A-Za-z0-9_.]+\.(?:jsonl?|txt))"')


@pytest.fixture(autouse=True)
def _kernel_only():
    """The platform alone: whatever extensions the environment installs claim their own files in
    their own suites."""
    with use_extensions([]):
        yield


def _classify(rel: str) -> str:
    claims = inventory.declared_claims()
    hit = next((c for c in claims if c.pattern.fullmatch(rel)), None)
    return hit.kind if hit else "unclaimed"


def test_no_kernel_store_writes_a_per_user_file_any_more() -> None:
    """Since the second moment every kernel per-user store is a table (ADR-169); only avatars stay
    files. A new per-user file in the server code is a store that belongs in the database, or one
    the inventory must at least declare."""
    assert _USER_LITERAL.search('data_dir / "users" / user_id / "comms.json"'), "pattern is stale"
    names = {
        m.group(1)
        for path in SERVER.rglob("*.py")
        for m in _USER_LITERAL.finditer(path.read_text())
    }
    undeclared = sorted(n for n in names if _classify(f"users/u_x/{n}") == "unclaimed")
    assert not undeclared, f"per-user files no inventory entry claims: {undeclared}"
    assert names == set(), f"per-user files written by kernel code: {sorted(names)}"


def test_declared_shapes() -> None:
    assert _classify("users/u_x/profile.json") == "imported"
    assert _classify("users/u_x/avatar.png") == "file"
    assert _classify("users/u_x/.profile.lock") == "ignored"
    assert _classify("users/.handles.lock") == "ignored"
    assert _classify("oauth_clients.lock") == "ignored"
    # The second moment's stores (E02 PR 2).
    for rel in (
        "users/u_x/comms.json",
        "users/u_x/push_subscriptions.json",
        "users/u_x/preferences.json",
        "users/u_x/apple_token.json",
        "users/u_x/graph_events.jsonl",
        "users/anon/graph_events.jsonl",
        "access_policy.json",
        "app_releases.json",
        "player_release.json",
        "audit.jsonl",
        "outbox/" + "0" * 64 + ".json",
        "magic_link_recent/" + "f" * 32 + ".txt",
        "magic_link_used/abc_DEF-123.json",
    ):
        assert _classify(rel) == "imported", rel
    assert _classify("outbox/" + "0" * 64 + ".lock") == "ignored"
    assert _classify("magic_link_used/.abc_DEF-123.lock") == "ignored"
    assert _classify("ranking_config.json") == "file"  # Player's, until E07
    # Shapes the writing code never produces are not waved through.
    assert _classify("outbox/notes.txt") == "unclaimed"
    assert _classify("users/u_x/sub/queue.json") == "unclaimed"
    assert _classify("users/u_x/queue.json.bak") == "unclaimed"
    assert _classify("random.json") == "unclaimed"


def test_retired_and_ops_shapes_from_prod() -> None:
    """The shapes the first dry run on prod's app data found unclaimed (2026-11-04)."""
    assert _classify("users/u_x/ranking_events.jsonl") == "retired"
    assert _classify("users/u_x/push_subscriptions.json.bak-dead-tokens-20261007083919") == "ops"
    # Only those exact shapes: an unknown backup is still unclaimed.
    assert _classify("users/u_x/push_subscriptions.json.bak-dead-tokens-1") == "unclaimed"
    assert _classify("users/u_x/push_subscriptions.json.bak-other-20261007083919") == "unclaimed"
    assert _classify("users/u_x/comms.json.bak-dead-tokens-20261007083919") == "unclaimed"
    assert _classify("users/u_x/playback_events.jsonl.bak-dedupe-milestones") == "unclaimed"


def test_retired_files_are_not_written_by_any_server_code() -> None:
    for name in inventory.RETIRED_USER_FILES:
        writers = [p.name for p in SERVER.rglob("*.py") if name in p.read_text()]
        assert not writers, f"{name} is retired but {writers} still name it"


def _fake_extension(name: str = "fake") -> Extension:
    return Extension(
        name=name,
        app_data_files=lambda: [
            inventory.user_file("fake_state.json", "file", name),
            inventory.root_file("fake_ledger.jsonl", "file", name),
            inventory.dir_files("fake_dir", r"[a-z]+\.json", "ops", name),
        ],
    )


def test_the_platform_alone_claims_no_extension_file() -> None:
    assert _classify("users/u_x/fake_state.json") == "unclaimed"
    assert _classify("fake_ledger.jsonl") == "unclaimed"
    assert _classify("users/u_x/profile.json") == "imported"


def test_an_installed_extensions_files_are_claimed() -> None:
    with use_extensions([_fake_extension()]):
        assert _classify("users/u_x/fake_state.json") == "file"
        assert _classify("fake_ledger.jsonl") == "file"
        assert _classify("fake_dir/abc.json") == "ops"
        # Only the shapes it declared.
        assert _classify("users/u_x/fake_state.json.bak") == "unclaimed"
        assert _classify("fake_dir/ABC.json") == "unclaimed"
        assert _classify("users/u_x/comms.json") == "imported"
    assert _classify("users/u_x/fake_state.json") == "unclaimed"


def test_scan_reports_a_not_installed_extensions_file_as_unclaimed(tmp_path: Path) -> None:
    (tmp_path / "users" / "u_x").mkdir(parents=True)
    (tmp_path / "users" / "u_x" / "comms.json").write_text("{}")
    (tmp_path / "users" / "u_x" / "fake_state.json").write_text("{}")
    with use_extensions([_fake_extension()]):
        assert inventory.scan(tmp_path).unclaimed == []
    inv = inventory.scan(tmp_path)
    assert inv.unclaimed == ["users/u_x/fake_state.json"]
    assert inv.by_class["imported"] == 1
