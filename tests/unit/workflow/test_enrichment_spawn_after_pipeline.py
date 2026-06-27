"""Mode B integration: the pipeline spawns enrichment in the background.

RFC-088 chunk-9 follow-up. The pipeline finalize step calls
``_maybe_spawn_enrichment_after_pipeline()`` which detaches a subprocess
running ``python -m podcast_scraper.enrichment.cli``. Tests stub
``subprocess.Popen`` so we never actually fork — we only assert the
gate (cfg.enrichment.enabled) + argv shape + log redirection.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from podcast_scraper.workflow.orchestration import (
    _maybe_spawn_enrichment_after_pipeline,
)


class _PopenSpy:
    """Drop-in stub for subprocess.Popen — records every invocation."""

    calls: list[dict[str, Any]] = []

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        _PopenSpy.calls.append({"argv": list(argv), "kwargs": dict(kwargs)})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    _PopenSpy.calls.clear()


def _cfg(**overrides: Any) -> Any:
    """Minimal cfg-shaped object the helper reads from. Typed as Any so
    callers can pass it into ``_maybe_spawn_enrichment_after_pipeline``
    which accepts the duck-typed cfg."""
    base: dict[str, Any] = {"enrichment": {}, "profile": None}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_spawn_skipped_when_enrichment_block_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cfg.enrichment unset → no Popen call."""
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", _PopenSpy)
    _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment=None), str(tmp_path))
    assert _PopenSpy.calls == []


def test_spawn_skipped_when_enabled_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cfg.enrichment.enabled = false → no Popen call."""
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", _PopenSpy)
    _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": False}), str(tmp_path))
    assert _PopenSpy.calls == []


def test_spawn_fires_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cfg.enrichment.enabled = true → Popen invoked with the CLI argv."""
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", _PopenSpy)
    _maybe_spawn_enrichment_after_pipeline(
        _cfg(enrichment={"enabled": True}, profile="airgapped_thin"), str(tmp_path)
    )
    assert len(_PopenSpy.calls) == 1
    argv = _PopenSpy.calls[0]["argv"]
    assert argv[1:3] == ["-m", "podcast_scraper.enrichment.cli"]
    assert "--output-dir" in argv
    assert str(tmp_path) in argv
    assert "--profile" in argv
    assert "airgapped_thin" in argv


def test_spawn_is_detached_with_devnull_stdin_and_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subprocess must be detached so SIGINT to the parent pipeline
    doesn't kill enrichment, and stdin is closed so it can't block."""
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", _PopenSpy)
    _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": True}), str(tmp_path))
    kwargs = _PopenSpy.calls[0]["kwargs"]
    assert kwargs.get("stdin") == subprocess.DEVNULL
    assert kwargs.get("start_new_session") is True
    assert kwargs.get("close_fds") is True


def test_spawn_redirects_output_to_viewer_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log file must land under .viewer/ so operators can tail it."""
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", _PopenSpy)
    _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": True}), str(tmp_path))
    log_path = tmp_path / ".viewer" / "enrichment_pipeline_spawn.log"
    assert log_path.is_file()  # opened for append by the helper


def test_spawn_failure_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OSError from Popen must not propagate — the pipeline keeps going."""
    import subprocess

    def boom(*a: Any, **kw: Any) -> None:
        raise OSError("simulated PATH issue")

    monkeypatch.setattr(subprocess, "Popen", boom)
    # No exception expected:
    _maybe_spawn_enrichment_after_pipeline(_cfg(enrichment={"enabled": True}), str(tmp_path))


def test_spawn_threads_profile_from_block_when_top_level_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If cfg.profile is unset but the enrichment block carries a
    profile name, that one wins."""
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", _PopenSpy)
    _maybe_spawn_enrichment_after_pipeline(
        _cfg(enrichment={"enabled": True, "profile": "cloud_balanced"}),
        str(tmp_path),
    )
    argv = _PopenSpy.calls[0]["argv"]
    assert "cloud_balanced" in argv
