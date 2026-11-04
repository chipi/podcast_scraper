"""The inventory must know every file name the stores write (ADR-169)."""

from __future__ import annotations

import re
from pathlib import Path

from podcast_scraper.db import inventory

SERVER = Path(__file__).resolve().parents[4] / "src" / "podcast_scraper" / "server"

#: ``data_dir / "users" / user_id / "name.ext"`` — a per-user file named by a string literal.
_USER_LITERAL = re.compile(r'"users"\s*/\s*\w+\s*/\s*"([A-Za-z0-9_.]+\.(?:jsonl?|txt))"')


def _classify(rel: str) -> str:
    claims = inventory.declared_claims()
    hit = next((c for c in claims if c.pattern.fullmatch(rel)), None)
    return hit.kind if hit else "unclaimed"


def test_every_literal_per_user_filename_in_the_server_code_is_declared() -> None:
    names = {
        m.group(1)
        for path in SERVER.rglob("*.py")
        for m in _USER_LITERAL.finditer(path.read_text())
    }
    assert names, "the scan found nothing; the pattern is stale"
    undeclared = sorted(n for n in names if _classify(f"users/u_x/{n}") == "unclaimed")
    assert not undeclared, f"per-user files no inventory entry claims: {undeclared}"


def test_every_state_document_name_is_declared() -> None:
    from podcast_scraper.server import app_user_state

    source = (SERVER / "app_user_state.py").read_text()
    used = set(
        re.findall(
            r"_(?:read|write|rows_for_update|mapping_for_update)"
            r'\(\s*data_dir,\s*user_id,\s*"([a-z_]+)"',
            source,
        )
    )
    assert used, "the scan found nothing; the pattern is stale"
    assert used <= set(app_user_state.STATE_NAMES), used - set(app_user_state.STATE_NAMES)
    for name in app_user_state.STATE_NAMES:
        assert _classify(f"users/u_x/{name}.json") == "file"


def test_declared_shapes() -> None:
    assert _classify("users/u_x/profile.json") == "imported"
    assert _classify("users/u_x/avatar.png") == "file"
    assert _classify("users/u_x/.profile.lock") == "ignored"
    assert _classify("users/.handles.lock") == "ignored"
    assert _classify("oauth_clients.json") == "file"
    assert _classify("oauth_clients.lock") == "ignored"
    assert _classify("outbox/" + "0" * 64 + ".json") == "file"
    assert _classify("magic_link_recent/" + "f" * 32 + ".txt") == "file"
    assert _classify("magic_link_used/abc_DEF-123.json") == "file"
    # Shapes the writing code never produces are not waved through.
    assert _classify("outbox/notes.txt") == "unclaimed"
    assert _classify("users/u_x/sub/queue.json") == "unclaimed"
    assert _classify("users/u_x/queue.json.bak") == "unclaimed"
    assert _classify("random.json") == "unclaimed"


def test_a_state_document_named_by_another_module_is_declared() -> None:
    """Modules besides app_user_state keep documents through its helpers (``_read(..., NAME, ...)``
    with ``NAME`` a module constant); each such constant must be in the inventory."""
    named = re.compile(r"_(?:read|write|user_lock)\(\s*data_dir,\s*user_id,\s*([A-Z_]+)\b")
    found: dict[str, str] = {}
    for path in SERVER.rglob("*.py"):
        for m in named.finditer(path.read_text()):
            const = m.group(1)
            value = re.search(rf'^{const}\s*=\s*"([a-z_]+)"', path.read_text(), re.M)
            assert value, f"{path.name}: cannot resolve {const}"
            found[value.group(1)] = path.name
    assert found, "the scan found nothing; the pattern is stale"
    undeclared = sorted(n for n in found if _classify(f"users/u_x/{n}.json") == "unclaimed")
    assert not undeclared, undeclared


def test_retired_and_ops_shapes_from_prod() -> None:
    """The shapes the first dry run on prod's app data found unclaimed (2026-11-04)."""
    assert _classify("users/u_x/ranking_events.jsonl") == "retired"
    assert _classify("users/u_x/playback_events.jsonl.bak-dedupe-milestones") == "ops"
    assert _classify("dedupe_playback_milestones.jsonl") == "ops"
    # Only those exact shapes: an unknown backup is still unclaimed.
    assert _classify("users/u_x/push_subscriptions.json.bak-dead-tokens-1") == "unclaimed"
    assert _classify("users/u_x/listen_events.jsonl.bak-dedupe-milestones") == "unclaimed"
    assert _classify("users/u_x/dedupe_playback_milestones.jsonl") == "unclaimed"


def test_the_dedupe_names_match_the_ops_script() -> None:
    script = (SERVER.parents[2] / "scripts" / "ops" / "dedupe_playback_milestones.py").read_text()
    for const, value in (
        ("LOG_NAME", inventory.DEDUPE_LOG_NAME),
        ("BACKUP_SUFFIX", inventory.DEDUPE_BACKUP_SUFFIX),
        ("RECEIPTS_FILE", inventory.DEDUPE_RECEIPTS_FILE),
    ):
        assert re.search(rf'^{const} = "{re.escape(value)}"$', script, re.M), const


def test_retired_files_are_not_written_by_any_server_code() -> None:
    for name in inventory.RETIRED_USER_FILES:
        writers = [p.name for p in SERVER.rglob("*.py") if name in p.read_text()]
        assert not writers, f"{name} is retired but {writers} still name it"
