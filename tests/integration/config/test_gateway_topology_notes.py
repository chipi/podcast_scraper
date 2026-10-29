"""Lock in the two WRONG FIXES that the "prod calls homelab" misreading produces.

Prose alone did not stop this: two separate agent sessions re-derived the deployment topology
from config files and both got it wrong, one nearly shipping a change to the homelab gateway
repo for a problem that lives entirely on the prod VPS. AGENTS.md now states the topology as
ground truth; these tests make the concrete bad fixes fail loudly instead of looking reasonable.

What is actually true (see AGENTS.md "Deployment topology" and ADR-142): the homelab hosts
observability only and is never on an LLM path. Since 2026-10-29 the profiles name no gateway
host at all: ``litellm_api_base`` is ``${LITELLM_API_BASE}``, filled from the environment by the
profile's own expansion, with no default; each deployment sets it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).resolve().parents[3]
_LOOPBACK = re.compile(r"https?://(127\.0\.0\.1|localhost)(:\d+)?(/|$)")


def test_config_never_reads_the_litellm_base_from_the_environment() -> None:
    """Config code must not bind the gateway field to an env var of its own.

    The profile states the gateway as ``${LITELLM_API_BASE}`` and the profile loader's generic
    ``${VAR}`` expansion fills it, so the profile stays the one place that says where LLM calls go.
    A second, hidden path in Config (a field default or validator reading the variable) would let
    the effective gateway differ from what the profile says.
    """
    source = (REPO / "src" / "podcast_scraper" / "config.py").read_text(encoding="utf-8")
    assert "LITELLM_API_BASE" not in source, (
        "config.py references LITELLM_API_BASE. Config must NOT read the gateway base from the "
        "environment — prod sets it as an explicit operator-config field (the D4 pin), not via "
        "env-detect. See AGENTS.md 'Deployment topology'."
    )


def test_profiles_name_no_gateway_host() -> None:
    """Every ``litellm_api_base`` in the profiles is the ``${LITELLM_API_BASE}`` slot (or loopback).

    A literal host there is how this repository came to name a deployment's gateway, and how
    the "prod calls homelab" misreading starts. The value comes from the environment, with no
    default (Config refuses it unset).
    """
    offenders = []
    for path in sorted((REPO / "config" / "profiles").glob("*.yaml")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            m = re.match(r"\s*litellm_api_base:\s*(\S+)", line)
            value = m.group(1).strip("'\"") if m else ""
            # Loopback is a local stand-in (fixture runs), not anyone's deployment.
            if m and value != "${LITELLM_API_BASE}" and not _LOOPBACK.match(value):
                offenders.append(f"{path.name}:{i + 1}: {m.group(1)}")
    assert not offenders, (
        "litellm_api_base must be ${LITELLM_API_BASE} (set in .env / the deployment), not a "
        "literal host:\n  " + "\n  ".join(offenders)
    )
