"""Every traced compose service stamps its traces with the release, not only the environment.

2026-10-15 prod (sha-f35a446): GlitchTip events carried release=sha-f35a446, but player-api,
player-mcp and podcast-pipeline traces had no service.version — OTEL_RESOURCE_ATTRIBUTES held only
deployment.environment. The api starts under ``opentelemetry-instrument``, whose resource comes from
the env alone, so the release has to be in the env (ADR-0010: release = sha-<7>, on every signal).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = sorted((ROOT / "compose").glob("docker-compose*.yml"))


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates compose's own tags (``!reset``, ``!override``)."""


_ComposeLoader.add_multi_constructor("!", lambda loader, suffix, node: None)


def _traced_services() -> list[tuple[str, str, str]]:
    out = []
    for f in COMPOSE:
        # _ComposeLoader is a SafeLoader subclass; bandit cannot see through the subclass.
        data = yaml.load(f.read_text(), Loader=_ComposeLoader)  # nosec B506
        services = (data or {}).get("services") or {}
        for name, svc in services.items():
            env = (svc or {}).get("environment") or {}
            if isinstance(env, dict) and "OTEL_RESOURCE_ATTRIBUTES" in env:
                out.append((f.name, name, str(env["OTEL_RESOURCE_ATTRIBUTES"])))
    return out


def test_there_are_traced_services_to_check() -> None:
    # This repository's compose files trace api, pipeline and pipeline-llm. The deployment's own
    # overlays carry the rest and are checked where they live.
    assert len(_traced_services()) >= 3


@pytest.mark.parametrize("compose_file,service,attrs", _traced_services())
def test_traces_carry_environment_and_release(compose_file: str, service: str, attrs: str) -> None:
    keys = dict(item.split("=", 1) for item in attrs.split(","))
    assert keys.get("deployment.environment") == "${PODCAST_ENV:-dev}", (compose_file, service)
    assert keys.get("service.version") == "${PODCAST_RELEASE:-unreleased}", (compose_file, service)
