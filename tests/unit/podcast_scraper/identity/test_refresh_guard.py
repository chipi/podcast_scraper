"""The ledger refresh holds back the respelling until 0041 is applied (record 0041)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.identity import ledger as L, refresh_guard as G


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch):
    seen = {"refresh": 0, "write": []}
    monkeypatch.setattr(L, "build_ledger", lambda root: {"built": True})
    monkeypatch.setattr(
        L,
        "write_ledger",
        lambda root, built, extra=None: seen["write"].append(extra) or {"joins": 3},
    )

    def refresh(root, *, after, log=None):
        seen["refresh"] += 1
        return True

    monkeypatch.setattr(L, "refresh_ledger", refresh)
    return seen


def _applied(monkeypatch: pytest.MonkeyPatch, ids) -> None:
    from podcast_scraper.upgrade import state

    monkeypatch.setattr(state.FilesystemStateStore, "applied_migration_ids", lambda self: set(ids))


def test_while_0041_is_pending_the_ledger_is_written_and_nothing_is_respelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls
) -> None:
    _applied(monkeypatch, {"0039_adfree_segment_offsets_realigned"})
    assert G.refresh_ledger_after(tmp_path, after="upgrade run") is True
    assert calls["refresh"] == 0
    assert calls["write"] == [
        {"respell": {"deferred": True, "reason": "0041_joined_names_respelled pending"}}
    ]


def test_from_0041_on_the_refresh_respells_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls
) -> None:
    _applied(monkeypatch, {"0041_joined_names_respelled"})
    assert G.refresh_ledger_after(tmp_path, after="enrichment") is True
    assert calls["refresh"] == 1 and calls["write"] == []


def test_a_failed_build_leaves_the_previous_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls
) -> None:
    _applied(monkeypatch, set())

    def boom(root):
        raise RuntimeError("no corpus")

    monkeypatch.setattr(L, "build_ledger", boom)
    assert G.refresh_ledger_after(tmp_path, after="enrichment") is False
    assert calls["write"] == []


def test_upgrade_run_and_enrich_use_the_guard() -> None:
    src = Path(G.__file__).parents[1]
    for rel in ("upgrade/cli_handlers.py", "enrichment/cli.py"):
        text = (src / rel).read_text(encoding="utf-8")
        assert "refresh_ledger_after(" in text and "import refresh_ledger\n" not in text, rel


def test_the_guard_is_outside_every_d2_migrations_measured_code() -> None:
    from podcast_scraper.upgrade import outcome_hook as oh

    for mid in ("0039_adfree_segment_offsets_realigned", "0041_joined_names_respelled"):
        assert "podcast_scraper.identity.refresh_guard" not in oh.module_closure(
            f"podcast_scraper.upgrade.migrations.m{mid}"
        )
