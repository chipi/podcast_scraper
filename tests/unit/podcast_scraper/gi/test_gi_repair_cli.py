"""The ``gi-repair`` CLI wiring — the layer where the only real bug so far actually lived.

test_gi_repair.py calls ``repair_placeholder_artifacts`` directly with a stub config, so it
never touches argument parsing, config loading, exit codes, or the audit-path default. The first
live run of the subcommand died with::

    NameError: name 'load_config_file' is not defined

because the handler called the bare name instead of ``config.load_config_file``. Twelve green
unit tests said nothing about it. These tests exercise the entrypoint an operator actually types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.cli import _parse_gi_repair_argv, _run_gi_repair_cli
from podcast_scraper.gi.corpus import LEGACY_PLACEHOLDER_INSIGHT_TEXT

pytestmark = [pytest.mark.unit]


def _corpus_with_placeholder(root: Path) -> Path:
    run = root / "feeds" / "feed_a" / "run_20260815-120000"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    name = "0001 - Episode"
    (run / "transcripts" / f"{name}.txt").write_text("Some spoken words.", encoding="utf-8")
    (run / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": "ep-1", "title": name},
                "feed": {"feed_id": "https://example.com/feed.xml"},
                "content": {"transcript_file_path": f"transcripts/{name}.txt"},
                "grounded_insights": {"artifact_path": f"metadata/{name}.gi.json"},
            }
        ),
        encoding="utf-8",
    )
    gi = run / "metadata" / f"{name}.gi.json"
    gi.write_text(
        json.dumps(
            {
                "episode_id": "ep-1",
                "nodes": [
                    {"id": "ep-1", "type": "Episode", "properties": {}},
                    {
                        "id": "ep-1:i0",
                        "type": "Insight",
                        "properties": {"text": LEGACY_PLACEHOLDER_INSIGHT_TEXT},
                    },
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    return gi


class _Log:
    def info(self, *_a: Any, **_k: Any) -> None:
        pass

    def warning(self, *_a: Any, **_k: Any) -> None:
        pass


def test_the_parser_accepts_the_documented_flags():
    args = _parse_gi_repair_argv(
        [
            "--output-dir",
            "/tmp/x",
            "--config",
            "p.yaml",
            "--dry-run",
            "--audit-file",
            "/tmp/a.jsonl",
        ]
    )

    assert args.command == "gi-repair"
    assert args.output_dir == "/tmp/x"
    assert args.config == "p.yaml"
    assert args.dry_run is True
    assert args.audit_file == "/tmp/a.jsonl"


def test_loading_a_real_config_file_does_not_explode(tmp_path):
    """THE regression: the handler used a bare ``load_config_file``, which is not in scope.

    Any config file will do — the point is that the load path executes at all.
    """
    _corpus_with_placeholder(tmp_path)
    profile = tmp_path / "profile.yaml"
    profile.write_text("generate_summaries: true\ngenerate_gi: false\n", encoding="utf-8")

    args = _parse_gi_repair_argv(
        ["--output-dir", str(tmp_path), "--config", str(profile), "--dry-run"]
    )
    rc = _run_gi_repair_cli(args, _Log())

    assert rc == 0, "dry-run over a corpus with one placeholder must succeed"


def test_a_missing_output_dir_is_a_clean_error_not_a_traceback(tmp_path):
    args = _parse_gi_repair_argv(["--output-dir", str(tmp_path / "nope")])

    assert _run_gi_repair_cli(args, _Log()) == 2


def test_dry_run_exits_zero_and_writes_nothing(tmp_path, capsys):
    gi = _corpus_with_placeholder(tmp_path)
    before = gi.read_bytes()

    args = _parse_gi_repair_argv(["--output-dir", str(tmp_path), "--dry-run"])
    rc = _run_gi_repair_cli(args, _Log())

    assert rc == 0
    assert gi.read_bytes() == before
    assert "would be repaired" in capsys.readouterr().out
    assert not (tmp_path / "gi_repair_report.jsonl").exists(), "dry-run must not write an audit"


def test_a_failed_repair_exits_NON_ZERO(tmp_path, monkeypatch, capsys):
    """The property that makes this safe to script: a failure cannot look like success."""
    _corpus_with_placeholder(tmp_path)

    def _explode(*_a: Any, **_k: Any) -> Dict[str, Any]:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("podcast_scraper.gi.build_artifact", _explode, raising=False)
    monkeypatch.setattr("podcast_scraper.gi.pipeline.build_artifact", _explode, raising=False)

    args = _parse_gi_repair_argv(["--output-dir", str(tmp_path)])
    rc = _run_gi_repair_cli(args, _Log())

    assert rc == 1, "a failed episode must produce a non-zero exit"
    assert "VERDICT: FAIL" in capsys.readouterr().out


def test_the_audit_file_defaults_into_the_corpus(tmp_path, monkeypatch):
    """Operators should get an audit trail without having to ask for one."""
    _corpus_with_placeholder(tmp_path)

    def _explode(*_a: Any, **_k: Any) -> Dict[str, Any]:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("podcast_scraper.gi.build_artifact", _explode, raising=False)
    monkeypatch.setattr("podcast_scraper.gi.pipeline.build_artifact", _explode, raising=False)

    args = _parse_gi_repair_argv(["--output-dir", str(tmp_path)])
    _run_gi_repair_cli(args, _Log())

    audit = tmp_path / "gi_repair_report.jsonl"
    assert audit.is_file(), "the default audit path must land inside the corpus"
    rows: List[Dict[str, Any]] = [
        json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()
    ]
    assert rows and rows[0]["ok"] is False
    assert "provider unavailable" in (rows[0]["error"] or "")


def test_an_explicit_audit_file_is_honoured(tmp_path, monkeypatch):
    _corpus_with_placeholder(tmp_path)
    monkeypatch.setattr(
        "podcast_scraper.gi.build_artifact",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")),
        raising=False,
    )
    elsewhere = tmp_path / "audits" / "custom.jsonl"

    args = _parse_gi_repair_argv(["--output-dir", str(tmp_path), "--audit-file", str(elsewhere)])
    _run_gi_repair_cli(args, _Log())

    assert elsewhere.is_file()


def test_a_corpus_with_nothing_to_repair_exits_zero(tmp_path, capsys):
    (tmp_path / "feeds").mkdir()

    args = _parse_gi_repair_argv(["--output-dir", str(tmp_path)])
    rc = _run_gi_repair_cli(args, _Log())

    assert rc == 0
    assert "Nothing to repair" in capsys.readouterr().out
