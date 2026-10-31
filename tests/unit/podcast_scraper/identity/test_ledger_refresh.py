"""The ledger step at the end of every corpus-writing path (plan V3, D1; ADR-166)."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.identity import ledger as L

pytestmark = pytest.mark.unit
LOG = logging.getLogger("test")


def test_refresh_writes_the_ledger(tmp_path: Path) -> None:
    assert L.refresh_ledger(tmp_path, after="test") is True
    assert L.read_ledger(tmp_path)[1] == L.STATE_OK


def test_a_failed_refresh_leaves_the_previous_ledger(tmp_path: Path, monkeypatch) -> None:
    L.write_ledger(tmp_path, L.build_ledger(tmp_path))
    before = (tmp_path / L.LEDGER_NAME).read_bytes()

    def boom(_root):
        raise RuntimeError("corpus unreadable")

    monkeypatch.setattr(L, "build_ledger", boom)
    assert L.refresh_ledger(tmp_path, after="test") is False
    assert (tmp_path / L.LEDGER_NAME).read_bytes() == before


# ---- enrich ----------------------------------------------------------------------------------


def _enrich(tmp_path: Path, monkeypatch, *extra: str, status: str = "ok"):
    from podcast_scraper.enrichment import cli as ecli

    class FakeExecutor:
        def __init__(self, **_kw):
            pass

        async def run(self, **_kw):
            return SimpleNamespace(run_id="r", status=status, duration_ms=1)

    monkeypatch.setattr(ecli, "EnrichmentExecutor", FakeExecutor)
    args = ecli.build_arg_parser().parse_args(["--output-dir", str(tmp_path), *extra])
    return asyncio.run(ecli.run_cli(args))


def test_a_successful_enrich_run_rebuilds_the_ledger(tmp_path: Path, monkeypatch) -> None:
    assert _enrich(tmp_path, monkeypatch) == 0
    assert L.read_ledger(tmp_path)[1] == L.STATE_OK


def test_no_identity_ledger_skips_it(tmp_path: Path, monkeypatch) -> None:
    assert _enrich(tmp_path, monkeypatch, "--no-identity-ledger") == 0
    assert L.read_ledger(tmp_path)[1] == L.STATE_MISSING


def test_a_failed_enrich_run_does_not_touch_the_ledger(tmp_path: Path, monkeypatch) -> None:
    assert _enrich(tmp_path, monkeypatch, status="failed") == 1
    assert L.read_ledger(tmp_path)[1] == L.STATE_MISSING


def test_a_failed_refresh_fails_the_enrich_run(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(L, "refresh_ledger", lambda *_a, **_k: False)
    assert _enrich(tmp_path, monkeypatch) == 4
    assert "identity ledger refresh FAILED" in capsys.readouterr().out


# ---- upgrade run -----------------------------------------------------------------------------


def _upgrade(tmp_path: Path, *argv: str) -> int:
    from podcast_scraper.upgrade.cli_handlers import parse_upgrade_argv, run_upgrade_cli

    return run_upgrade_cli(
        parse_upgrade_argv([argv[0], "--corpus-dir", str(tmp_path), *argv[1:]]), LOG
    )


def test_an_upgrade_run_that_applies_rebuilds_the_ledger(tmp_path: Path) -> None:
    assert _upgrade(tmp_path, "run", "--yes", "--no-snapshot") == 0
    assert L.read_ledger(tmp_path)[1] == L.STATE_OK


def test_a_dry_run_does_not_write_the_ledger(tmp_path: Path) -> None:
    assert _upgrade(tmp_path, "run", "--yes", "--dry-run", "--no-snapshot") == 0
    assert L.read_ledger(tmp_path)[1] == L.STATE_MISSING


def test_a_failed_refresh_fails_the_upgrade_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(L, "refresh_ledger", lambda *_a, **_k: False)
    assert _upgrade(tmp_path, "run", "--yes", "--no-snapshot") == 4


# ---- enrich-edges ----------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location(
    "spoken_by_fixture",
    Path(__file__).resolve().parents[2] / "search" / "test_spoken_by_cross_episode.py",
)
assert _spec and _spec.loader
_fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fx)


def _edges(corpus: Path) -> int:
    from podcast_scraper.search.cli_handlers import parse_enrich_edges_argv, run_enrich_edges_cli

    return run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(corpus)]), LOG)


def test_an_enrich_edges_pass_that_rewrites_rebuilds_the_ledger(tmp_path: Path) -> None:
    _fx._write_episode(
        tmp_path,
        "ep1",
        "ep1",
        host="Maya",
        quote_text="Reliability matters.",
        insight_text="Reliability matters most.",
    )
    assert _edges(tmp_path) == 0
    assert L.read_ledger(tmp_path)[1] == L.STATE_OK
    builds = len((tmp_path / L.CHANGES_NAME).read_text(encoding="utf-8").splitlines())
    assert _edges(tmp_path) == 0  # nothing rewritten the second time: no rebuild
    assert len((tmp_path / L.CHANGES_NAME).read_text(encoding="utf-8").splitlines()) == builds


def test_a_failed_refresh_fails_enrich_edges(tmp_path: Path, monkeypatch) -> None:
    from podcast_scraper.search.cli_handlers import EXIT_LEDGER_FAILED

    _fx._write_episode(
        tmp_path,
        "ep1",
        "ep1",
        host="Maya",
        quote_text="Reliability matters.",
        insight_text="Reliability matters most.",
    )
    monkeypatch.setattr(L, "refresh_ledger", lambda *_a, **_k: False)
    assert _edges(tmp_path) == EXIT_LEDGER_FAILED
