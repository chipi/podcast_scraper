"""Every registered MCP tool has a zero series from the start (moved from tests/unit/mcp).

Integration tier because it needs the real ``prometheus_client`` (an optional extra): the 3-tier
policy forbids ``importorskip`` in tests/unit, and mocking the registry would test the mock rather
than the counter.
"""

from __future__ import annotations

import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import REGISTRY  # noqa: E402

pytestmark = pytest.mark.integration


def test_every_registered_tool_has_a_zero_series_from_the_start() -> None:
    """After a restart, a tool nobody has called yet must read 0, not be missing: a labelled
    counter has no series until its first increment, so 'unused' looked like 'not scraped'."""
    from podcast_scraper.mcp.server import build_server

    server = build_server("tests/fixtures/app-validation-corpus")
    names = [t.name for t in server._tool_manager.list_tools()]
    assert len(names) >= 20
    for name in names:
        for ok in ("true", "false"):
            value = REGISTRY.get_sample_value("mcp_tool_calls_total", {"tool": name, "ok": ok})
            assert value is not None, (name, ok)
