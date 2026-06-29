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


def test_run_yes_json_on_empty_corpus(tmp_path, capsys):
    # No FAISS, no metadata → 0001 no-ops, 0002 native build finds nothing; --json prints.
    import json as _json

    rc = _run(tmp_path, "run", "--yes", "--json")
    assert rc == 0
    payload = _json.loads(capsys.readouterr().out)
    assert {r["id"] for r in payload} == {
        "0001_faiss_to_lance",
        "0002_two_tier_native_reindex",
        "0003_gi_v3_typed_mentions",
    }


def test_unknown_subcommand_errors(tmp_path):
    from argparse import Namespace

    from podcast_scraper.upgrade.cli_handlers import run_upgrade_cli

    args = Namespace(
        command="upgrade", upgrade_subcommand="bogus", corpus_dir=str(tmp_path), json=False
    )
    assert run_upgrade_cli(args, log) == 1


def test_confirm_prompt_tty(monkeypatch):
    from podcast_scraper.upgrade import cli_handlers as ch

    status = type("St", (), {"pending": [type("M", (), {"id": "0001_x"})()]})()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_a: "y")
    assert ch._confirm(status, log) is True
    monkeypatch.setattr("builtins.input", lambda *_a: "n")
    assert ch._confirm(status, log) is False


def test_verify_text_mode_no_applied(tmp_path, capsys):
    assert _run(tmp_path, "verify") == 0  # text mode (no --json)
    assert "No applied migrations" in capsys.readouterr().out
