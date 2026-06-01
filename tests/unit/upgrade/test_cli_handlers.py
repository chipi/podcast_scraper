"""Unit tests for the upgrade CLI surface (#862): list, --json, non-TTY refusal."""

from __future__ import annotations

import json
import logging

import pytest

from podcast_scraper.upgrade.cli_handlers import parse_upgrade_argv, run_upgrade_cli

pytestmark = pytest.mark.unit
log = logging.getLogger("test")


def _run(corpus, *argv, capsys=None):
    args = parse_upgrade_argv([argv[0], "--corpus-dir", str(corpus), *argv[1:]])
    return run_upgrade_cli(args, log)


def test_missing_corpus_dir_errors(monkeypatch):
    monkeypatch.delenv("CORPUS_DIR", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    args = parse_upgrade_argv(["status"])
    assert run_upgrade_cli(args, log) == 1  # no --corpus-dir → error


def test_status_json(tmp_path, capsys):
    assert _run(tmp_path, "status", "--json") == 2  # pending → exit 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["up_to_date"] is False
    assert "0001_faiss_to_lance" in payload["pending"]


def test_list_text_and_json(tmp_path, capsys):
    assert _run(tmp_path, "list") == 0
    assert "0001_faiss_to_lance" in capsys.readouterr().out
    assert _run(tmp_path, "list", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert all(r["state"] == "pending" for r in rows)


def test_verify_json_with_nothing_applied(tmp_path, capsys):
    assert _run(tmp_path, "verify", "--json") == 0  # nothing applied → vacuously ok
    assert json.loads(capsys.readouterr().out) == []


def test_run_without_yes_refuses_on_non_tty(tmp_path, capsys):
    # pytest stdin is not a TTY → must refuse (safety gate), not apply.
    assert _run(tmp_path, "run") == 1
    assert "Aborted" in capsys.readouterr().out
    assert not (tmp_path / "upgrade_ledger.json").exists()


def test_run_up_to_date_is_noop(tmp_path, monkeypatch, capsys):
    # Force "nothing pending" so the up-to-date branch runs.
    from podcast_scraper.upgrade import cli_handlers

    class _UpToDate:
        state = type("S", (), {"current_version": staticmethod(lambda: "2.7.0")})()

        def status(self):
            return type("St", (), {"pending": []})()

    monkeypatch.setattr(cli_handlers, "_runner_for", lambda root: _UpToDate())
    args = parse_upgrade_argv(["run", "--corpus-dir", str(tmp_path), "--yes"])
    assert run_upgrade_cli(args, log) == 0
    assert "Up to date" in capsys.readouterr().out
