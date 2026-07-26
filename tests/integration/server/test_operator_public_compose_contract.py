#!/usr/bin/env python3
"""Compose-config contract tests for the PUBLIC operator surface (RFC-108 / epic #1320).

Encodes the T-01 safety claim as a gate: the public operator backend must be
low-privilege — ``PODCAST_SERVE_OPERATOR_PUBLIC=1`` (curated read-only + ≥creator gate),
**no ``docker.sock``**, **no provider keys** — so exposing it on the edge cannot reach
host-root or the privileged operator plane. Also asserts the operator differentiator
(``APP_ADMIN_EMAILS``), the hardening, and loopback-only ports (Caddy edge is the front).

Reverting any of these on ``docker-compose.operator-public.yml`` must fail here before it
ever reaches real infra.
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
OPERATOR_YML = REPO_ROOT / "compose" / "docker-compose.operator-public.yml"
OPERATOR_SECRETS_YML = REPO_ROOT / "compose" / "docker-compose.operator-secrets.yml"

pytestmark.append(
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker CLI not on PATH; compose-config contract test requires docker",
    )
)

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
    cmd = ["docker", "compose", "-f", str(OPERATOR_YML), "config", "--format", "yaml"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(f"`docker compose config` exited {proc.returncode}\n{proc.stderr}")
    parsed: Dict[str, Any] = yaml.safe_load(proc.stdout)
    return parsed


@pytest.fixture(scope="module")
def resolved() -> Dict[str, Any]:
    return _render()


def _svc(resolved: Dict[str, Any], name: str) -> Dict[str, Any]:
    svc = (resolved.get("services") or {}).get(name)
    assert svc is not None, f"operator-public compose has no ``{name}`` service"
    out: Dict[str, Any] = svc
    return out


def _volumes(svc: Dict[str, Any]) -> str:
    return " ".join(str(v) for v in (svc.get("volumes") or []))


def test_backend_is_operator_public_not_app_only(resolved: Dict[str, Any]) -> None:
    env = _svc(resolved, "api").get("environment") or {}
    assert str(env.get("PODCAST_SERVE_OPERATOR_PUBLIC")) == "1", "backend must be operator-public"
    # Must NOT also be app-only (that would drop the operator-read plane entirely).
    assert str(env.get("PODCAST_SERVE_APP_ONLY", "")) not in ("1", "true")


def test_backend_has_no_docker_socket(resolved: Dict[str, Any]) -> None:
    # T-01: a public-reachable backend must not hold host-root via docker.sock.
    assert "docker.sock" not in _volumes(_svc(resolved, "api"))


def test_backend_has_no_provider_keys(resolved: Dict[str, Any]) -> None:
    env = _svc(resolved, "api").get("environment") or {}
    present = _FORBIDDEN_KEYS & set(env.keys())
    assert not present, f"public operator backend must carry no provider keys; found {present}"


def test_backend_corpus_is_read_only(resolved: Dict[str, Any]) -> None:
    corpus_mounts = [
        v
        for v in (_svc(resolved, "api").get("volumes") or [])
        if isinstance(v, dict) and v.get("target") == "/app/output"
    ]
    assert corpus_mounts, "operator backend must mount the corpus at /app/output"
    assert all(m.get("read_only") is True for m in corpus_mounts), "corpus must be read-only"


def test_backend_is_operator_role_gated(resolved: Dict[str, Any]) -> None:
    # The operator differentiator: APP_ADMIN_EMAILS is present (→ admin/creator roles). The
    # ≥creator router gate (require_viewer_access) is enforced by the OPERATOR_PUBLIC mode.
    env = _svc(resolved, "api").get("environment") or {}
    assert "APP_ADMIN_EMAILS" in env, "operator backend must wire APP_ADMIN_EMAILS (role bootstrap)"
    assert str(env.get("APP_OAUTH_PROVIDER")) == "google"
    assert env.get("FORWARDED_ALLOW_IPS") == "*"


def test_backend_is_hardened(resolved: Dict[str, Any]) -> None:
    api = _svc(resolved, "api")
    assert "no-new-privileges:true" in " ".join(api.get("security_opt") or [])
    assert (api.get("cap_drop") or []) == ["ALL"]


def test_viewer_loopback_only_and_hardened(resolved: Dict[str, Any]) -> None:
    fe = _svc(resolved, "viewer")
    ports = " ".join(str(p) for p in (fe.get("ports") or []))
    assert "127.0.0.1" in ports, "operator viewer must bind loopback only (edge is the front)"
    assert fe.get("read_only") is True
    assert "no-new-privileges:true" in " ".join(fe.get("security_opt") or [])


def test_operator_secrets_overlay_delivers_runtime_secrets_via_tmpfs() -> None:
    # ADR-115 Option A: under OPERATOR_SECRETS_VIA_FILES=1, deploy-operator.sh joins this
    # overlay so the 3 runtime secrets (OAuth client secret, session secret, backend Sentry
    # DSN) are delivered as /run/secrets files sourced from /dev/shm (RAM) and exported by
    # the image shim — none at rest on disk. This guards the overlay from going missing again
    # (its absence makes an Option-A deploy fail on the `-f` join → keyless 503).
    assert OPERATOR_SECRETS_YML.exists(), "operator secrets overlay must exist (deploy joins it)"
    env = {**os.environ, "PODCAST_CORPUS_VOLUME": "compose_corpus_data"}
    cmd = [
        "docker",
        "compose",
        "-f",
        str(OPERATOR_YML),
        "-f",
        str(OPERATOR_SECRETS_YML),
        "config",
        "--format",
        "yaml",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)  # noqa: S603
    assert (
        proc.returncode == 0
    ), f"`docker compose config` (with secrets overlay) failed:\n{proc.stderr}"
    resolved: Dict[str, Any] = yaml.safe_load(proc.stdout)

    api = _svc(resolved, "api")
    attached = {(s.get("source") if isinstance(s, dict) else s) for s in (api.get("secrets") or [])}
    expected = {"app_oauth_google_client_secret", "app_session_secret", "podcast_sentry_dsn_api"}
    assert (
        expected <= attached
    ), f"api must mount the runtime secrets under Option A; got {attached}"

    top = resolved.get("secrets") or {}
    for name in expected:
        src = str((top.get(name) or {}).get("file", ""))
        assert src.startswith(
            "/dev/shm/operator-secrets/"
        ), f"{name} must source from the operator tmpfs (RAM, per-tenant), got {src!r}"

    # The plaintext must NOT also be pinned into the api environment (that would defeat
    # the point — no secret at rest / in the compose env under Option A).
    env_block = api.get("environment") or {}
    for leaky in ("APP_OAUTH_GOOGLE_CLIENT_SECRET", "APP_SESSION_SECRET"):
        val = str(env_block.get(leaky, ""))
        assert not val, f"{leaky} must NOT be a hardcoded env value under Option A (got a value)"


def test_viewer_defaults_corpus_path(resolved: Dict[str, Any]) -> None:
    # A fresh operator sign-in must land on the shared corpus (mounted /app/output:ro),
    # not an empty selection. The viewer image's nginx seeds ``ps_corpus_path`` from
    # PODCAST_DEFAULT_CORPUS_PATH; unset → no data on first login. Default it to the mount.
    env = _svc(resolved, "viewer").get("environment") or {}
    assert (
        env.get("PODCAST_DEFAULT_CORPUS_PATH") == "/app/output"
    ), "operator viewer must default the SPA corpus to the /app/output mount"
