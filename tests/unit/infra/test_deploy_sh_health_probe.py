"""Guard deploy.sh VPS-local health probe (GH-745)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEPLOY_SH = REPO / "infra" / "deploy" / "deploy.sh"


def test_deploy_sh_uses_compose_exec_for_api_health() -> None:
    """Api :8000 is not published on the host; loopback curl is a false negative."""
    text = DEPLOY_SH.read_text(encoding="utf-8")
    assert "exec -T api" in text
    assert "GH-745" in text
    assert "127.0.0.1:8000/api/health" in text
