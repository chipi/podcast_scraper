"""Lock in the two WRONG FIXES that the "prod calls homelab" misreading produces.

Prose alone did not stop this: two separate agent sessions re-derived the deployment topology
from config files and both got it wrong, one nearly shipping a change to the homelab gateway
repo for a problem that lives entirely on the prod VPS. AGENTS.md now states the topology as
ground truth; these tests make the two concrete bad fixes fail loudly instead of looking
reasonable.

What is actually true (see AGENTS.md "Deployment topology" and ADR-142): the homelab hosts
observability only and is never on an LLM path. The homelab base URL in the profiles is the
laptop-dev default, and prod overrides it every deploy via the D4 pin into viewer_operator.yaml.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).resolve().parents[3]


def test_config_never_reads_the_litellm_base_from_the_environment() -> None:
    """The tempting "fix" for the anomaly is env-detect in Config. It is forbidden.

    Someone will observe that deploy-prod plumbs LITELLM_API_BASE, that Config ignores it, and
    "fix" the inconsistency by binding the field to the env var. That would invert the operator's
    profiles-are-the-source-of-truth rule and make the effective gateway depend on ambient process
    state instead of the profile + explicit operator config.

    LITELLM_API_BASE has exactly two legitimate consumers, neither of them Config: the
    /api/ops/gateway/auth probe and scripts/eval/gateway_spend.py.
    """
    source = (REPO / "src" / "podcast_scraper" / "config.py").read_text(encoding="utf-8")
    assert "LITELLM_API_BASE" not in source, (
        "config.py references LITELLM_API_BASE. Config must NOT read the gateway base from the "
        "environment — prod sets it as an explicit operator-config field (the D4 pin), not via "
        "env-detect. See AGENTS.md 'Deployment topology'."
    )


def test_every_homelab_gateway_default_is_annotated_as_a_dev_default() -> None:
    """A bare ``http://homelab:4001/v1`` line is what starts the misreading.

    Each occurrence must carry a nearby comment naming it a dev default, so an agent reading the
    profile in isolation cannot conclude prod routes LLM traffic through the homelab. Strip the
    comment and this fails — which is the point, because the comment is load-bearing context, not
    decoration.
    """
    offenders = []
    for path in sorted((REPO / "config" / "profiles").glob("*.yaml")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not re.match(r"\s*litellm_api_base:.*homelab", line):
                continue
            window = "\n".join(lines[max(0, i - 6) : i]).lower()
            if "dev default" not in window and "adr-142" not in window:
                offenders.append(f"{path.name}:{i + 1}")
    assert not offenders, (
        "these homelab litellm_api_base lines are unannotated, so an agent reading them in "
        "isolation will conclude prod routes LLM calls through the homelab (it does not — the "
        "homelab is observability-only). Add a comment naming it the laptop-dev default and "
        "citing ADR-142:\n  " + "\n  ".join(offenders)
    )
