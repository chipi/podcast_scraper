#!/usr/bin/env python3
"""Compose-config contract tests for the PUBLIC consumer player (#1163 / ADR-116).

Encodes the T-01 safety claim as a gate: the public player backend must be
low-privilege — ``PODCAST_SERVE_APP_ONLY=1``, **no ``docker.sock``**, **no provider
keys** — so exposing it on the edge cannot reach host-root. Also asserts the hardening
(cap_drop, no-new-privileges, read-only frontend) and loopback-only ports so the Caddy
edge is the only public front.

Reverting any of these on ``docker-compose.player-public.yml`` must fail at least one
test here before it ever reaches real infra.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_YML = REPO_ROOT / "compose" / "docker-compose.player-public.yml"

pytestmark.append(
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker CLI not on PATH; compose-config contract test requires docker",
    )
)

# Provider keys that must NEVER appear on the public app-only backend (T-01).
_FORBIDDEN_KEYS = {
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROK_API_KEY",
}


def _render() -> Dict[str, Any]:
    env = {**os.environ, "PODCAST_CORPUS_VOLUME": "compose_corpus_data"}
    cmd = ["docker", "compose", "-f", str(PLAYER_YML), "config", "--format", "yaml"]
    proc = subprocess.run(  # noqa: S603 - hardcoded cmd
        cmd, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise AssertionError(f"`docker compose config` exited {proc.returncode}\n{proc.stderr}")
    parsed: Dict[str, Any] = yaml.safe_load(proc.stdout)
    return parsed


@pytest.fixture(scope="module")
def resolved() -> Dict[str, Any]:
    return _render()


def _svc(resolved: Dict[str, Any], name: str) -> Dict[str, Any]:
    svc = (resolved.get("services") or {}).get(name)
    assert svc is not None, f"player-public compose has no ``{name}`` service"
    out: Dict[str, Any] = svc
    return out


def _volumes(svc: Dict[str, Any]) -> str:
    return " ".join(str(v) for v in (svc.get("volumes") or []))


def test_backend_is_app_only(resolved: Dict[str, Any]) -> None:
    env = _svc(resolved, "api").get("environment") or {}
    assert str(env.get("PODCAST_SERVE_APP_ONLY")) == "1", "public backend must be app-only"


def test_backend_has_no_docker_socket(resolved: Dict[str, Any]) -> None:
    # The T-01 claim: a public-reachable backend must not hold host-root via docker.sock.
    assert "docker.sock" not in _volumes(_svc(resolved, "api"))


def test_backend_corpus_is_read_only(resolved: Dict[str, Any]) -> None:
    # #2: the corpus is shared read-during-write with the operator stack. The player
    # mounts it READ-ONLY, so it can't corrupt the corpus even via a route bug, and the
    # concurrent read is safe (LanceDB versioning + index_pool mtime-invalidation).
    corpus_mounts = [
        v
        for v in (_svc(resolved, "api").get("volumes") or [])
        if isinstance(v, dict) and v.get("target") == "/app/output"
    ]
    assert corpus_mounts, "player backend must mount the corpus at /app/output"
    assert all(
        m.get("read_only") is True for m in corpus_mounts
    ), "corpus must be mounted read-only"


def test_backend_has_no_provider_keys(resolved: Dict[str, Any]) -> None:
    env = _svc(resolved, "api").get("environment") or {}
    present = _FORBIDDEN_KEYS & set(env.keys())
    assert not present, f"public backend must carry no provider keys; found {present}"


def test_backend_trusts_forwarded_headers(resolved: Dict[str, Any]) -> None:
    env = _svc(resolved, "api").get("environment") or {}
    assert env.get("FORWARDED_ALLOW_IPS") == "*"
    assert str(env.get("APP_OAUTH_PROVIDER")) == "google"


def test_backend_is_hardened(resolved: Dict[str, Any]) -> None:
    api = _svc(resolved, "api")
    assert "no-new-privileges:true" in " ".join(api.get("security_opt") or [])
    assert (api.get("cap_drop") or []) == ["ALL"]


def test_frontend_loopback_only_and_hardened(resolved: Dict[str, Any]) -> None:
    fe = _svc(resolved, "learning-app")
    ports = " ".join(str(p) for p in (fe.get("ports") or []))
    assert (
        "127.0.0.1" in ports
    ), "player frontend must bind loopback only (edge is the public front)"
    assert fe.get("read_only") is True
    assert "no-new-privileges:true" in " ".join(fe.get("security_opt") or [])
