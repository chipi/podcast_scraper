"""Guard the DGX converge's env_file path (2026-10-09).

be99e8718 replaced the operator's username in infra/dgx/converge/deploy.py with a literal
``<OPERATOR_USER>``, so every generated compose pointed ``env_file`` at a path that does not
exist and compose would refuse all four services on the next converge. The user now comes
from the deploy environment, and a missing one stops the deploy before it writes anything.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DEPLOY_PY = REPO / "infra" / "dgx" / "converge" / "deploy.py"
COMPOSES = (
    "COMPOSE_CONTENT",
    "PYANNOTE_COMPOSE_CONTENT",
    "WHISPER_COMPOSE_CONTENT",
    "MOSS_COMPOSE_CONTENT",
)


class _NoOp:
    """Stands in for pyinfra's operation modules: loading deploy.py must not touch a host."""

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


def _load_deploy(monkeypatch: pytest.MonkeyPatch, **env: str) -> types.ModuleType:
    for key in ("DGX_OPERATOR_USER", "DGX_SSH_USER"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    operations = types.ModuleType("pyinfra.operations")
    operations.files = _NoOp()  # type: ignore[attr-defined]
    operations.server = _NoOp()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyinfra", types.ModuleType("pyinfra"))
    monkeypatch.setitem(sys.modules, "pyinfra.operations", operations)
    spec = importlib.util.spec_from_file_location("dgx_converge_deploy", DEPLOY_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("env", "user"),
    [
        ({"DGX_OPERATOR_USER": "alice", "DGX_SSH_USER": "root"}, "alice"),
        ({"DGX_SSH_USER": "bob"}, "bob"),
        ({"DGX_OPERATOR_USER": "carol", "DGX_SSH_USER": "dave"}, "carol"),
    ],
)
def test_every_service_compose_reads_the_operators_env(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], user: str
) -> None:
    deploy = _load_deploy(monkeypatch, **env)
    for name in COMPOSES:
        compose = getattr(deploy, name)
        assert f"- /home/{user}/.env\n" in compose, name
        assert "<OPERATOR_USER>" not in compose, name


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"DGX_SSH_USER": "root"},
        {"DGX_OPERATOR_USER": "root"},
        {"DGX_OPERATOR_USER": "../etc", "DGX_SSH_USER": "root"},
        {"DGX_OPERATOR_USER": "has space"},
    ],
)
def test_no_usable_operator_stops_before_writing(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    with pytest.raises(SystemExit, match="DGX_OPERATOR_USER"):
        _load_deploy(monkeypatch, **env)
