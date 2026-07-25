"""Guard for the tailscale/github-action authkey→OAuth migration (#1289).

Every workflow that joins the tailnet must authenticate via the **OAuth client**
(``oauth-client-id`` / ``oauth-secret``), never the deprecated ``authkey`` input. Locks
the migration so a copy-paste can't reintroduce ``authkey:`` on a prod deploy/backup/drill.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _tailnet_joining_workflows() -> list[Path]:
    return sorted(
        p
        for p in WORKFLOWS.glob("*.yml")
        if "tailscale/github-action" in p.read_text(encoding="utf-8")
    )


@pytest.mark.unit
def test_there_are_tailnet_joining_workflows() -> None:
    # Sanity: the guard is actually scanning something (it was 11 at migration time).
    assert len(_tailnet_joining_workflows()) >= 8


@pytest.mark.unit
@pytest.mark.parametrize("wf", _tailnet_joining_workflows(), ids=lambda p: p.name)
def test_tailscale_action_uses_oauth_not_authkey(wf: Path) -> None:
    text = wf.read_text(encoding="utf-8")
    # Must use the OAuth client inputs.
    assert "oauth-client-id:" in text, f"{wf.name}: tailscale action must use oauth-client-id"
    assert "oauth-secret:" in text, f"{wf.name}: tailscale action must use oauth-secret"
    # Must NOT use the deprecated authkey action input (secrets.TS_AUTHKEY as the join key).
    assert not re.search(
        r"^\s*authkey:\s*\$\{\{\s*secrets\.TS_AUTHKEY", text, re.MULTILINE
    ), f"{wf.name}: deprecated authkey input reintroduced — use OAuth"
