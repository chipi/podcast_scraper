"""The board-summarizer sidecar runner (RFC-127), loaded by path as the digest runner's tests do.

These lock what the sidecar adds on top of ``board_summaries.sweep``: the kill switch is read
before the model is even built, a model that cannot be reached ends the cycle quietly, a provider
without ``complete_prose`` is refused, and a bad cycle never kills the loop.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")  # _run_once imports the server package

pytestmark = pytest.mark.integration

_RUNNER = Path(__file__).resolve().parents[2] / "infra" / "deploy" / "board_summarizer.py"


def _load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("BOARD_SUMMARY_HEARTBEAT_FILE", str(tmp_path / "hb" / "tick"))
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app"))
    monkeypatch.setenv("BOARD_SUMMARY_CORPUS_ROOT", str(tmp_path / "corpus"))
    spec = importlib.util.spec_from_file_location("board_summarizer_under_test", _RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_profile_is_the_dgx_one_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BOARD_SUMMARY_PROFILE", raising=False)
    assert _load(tmp_path, monkeypatch).PROFILE == "prod_dgx_full"


def test_switched_off_never_builds_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load(tmp_path, monkeypatch)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "board_summaries_settings.json").write_text(json.dumps({"enabled": False}))

    def boom() -> Any:
        raise AssertionError("the model was built while switched off")

    monkeypatch.setattr(mod, "_provider", boom)
    mod._run_once()
    assert '"stopped": "switched_off"' in capsys.readouterr().out


def test_a_model_that_is_away_ends_the_cycle_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load(tmp_path, monkeypatch)

    def away() -> Any:
        raise ConnectionError("dgx-llm-1:8003 refused")

    monkeypatch.setattr(mod, "_provider", away)
    mod._run_once()
    assert '"stopped": "model_unreachable"' in capsys.readouterr().out


def test_a_provider_without_complete_prose_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load(tmp_path, monkeypatch)

    class Cfg:
        summary_provider = "anthropic"

    monkeypatch.setattr(mod, "_provider", lambda: (Cfg(), object()))
    mod._run_once()
    assert '"stopped": "unsupported_provider"' in capsys.readouterr().out


def test_a_bad_cycle_never_kills_the_loop_and_still_beats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load(tmp_path, monkeypatch)

    def broken() -> None:
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(mod, "_run_once", broken)
    mod._cycle()
    assert (tmp_path / "hb" / "tick").read_text().isdigit()
