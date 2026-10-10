"""Guard which services a default DGX converge touches (2026-10-10).

A dry run of ``make dgx-deploy`` showed a default converge would build and START the retired
openai-whisper server (:8002) and install an observability stack whose container names clash
with the exporters the homelab repo already runs on the DGX. Both are opt-in now; the live
services (faster-whisper, pyannote, moss) must still converge by default.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DEPLOY_PY = REPO / "infra" / "dgx" / "converge" / "deploy.py"
FLAGS = ("DGX_CONVERGE_WHISPER_SERVER", "DGX_CONVERGE_OBSERVABILITY")


class _Recorder:
    """Stands in for a pyinfra operations module and records each operation's ``name``."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def __getattr__(self, attr: str):
        def op(*args, **kwargs) -> None:
            self._names.append(str(kwargs.get("name", attr)))

        return op


def _operation_names(monkeypatch: pytest.MonkeyPatch, **env: str) -> list[str]:
    for key in FLAGS + ("DGX_OPERATOR_USER",):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DGX_SSH_USER", "alice")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    names: list[str] = []
    operations = types.ModuleType("pyinfra.operations")
    operations.files = _Recorder(names)  # type: ignore[attr-defined]
    operations.server = _Recorder(names)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyinfra", types.ModuleType("pyinfra"))
    monkeypatch.setitem(sys.modules, "pyinfra.operations", operations)
    spec = importlib.util.spec_from_file_location("dgx_converge_deploy_ops", DEPLOY_PY)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return names


def _touches(names: list[str], *needles: str) -> bool:
    return any(n in name for name in names for n in needles)


def test_default_converge_skips_retired_and_duplicate_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = _operation_names(monkeypatch)
    # "faster-whisper-server/Dockerfile" is the LIVE service; match the retired one exactly.
    assert not _touches(names, "/opt/whisper-server", "ship: whisper-server/", "whisper-openai")
    assert not _touches(names, "/opt/observability", "observability/docker-compose.yml", "DCGM")
    for live in ("/opt/faster-whisper", "pyannote", "moss"):
        assert _touches(names, live), live


@pytest.mark.parametrize(
    ("flag", "needle"),
    [
        ("DGX_CONVERGE_WHISPER_SERVER", "whisper-openai"),
        ("DGX_CONVERGE_OBSERVABILITY", "/opt/observability"),
    ],
)
def test_each_block_comes_back_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, flag: str, needle: str
) -> None:
    assert _touches(_operation_names(monkeypatch, **{flag: "1"}), needle)
