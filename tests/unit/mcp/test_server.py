"""MCP protocol smoke test (RFC-095 slice 1) — requires the [mcp] extra."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from podcast_scraper.mcp.cli_handlers import parse_mcp_argv
from podcast_scraper.mcp.server import build_server

pytestmark = pytest.mark.unit


def test_build_server_registers_slice1_tools(tmp_path) -> None:
    server = build_server(tmp_path)
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert names == {"resolve_entity", "search_corpus"}


def test_registered_tools_have_descriptions(tmp_path) -> None:
    server = build_server(tmp_path)
    by_name = {t.name: t for t in server._tool_manager.list_tools()}
    # Agent-facing descriptions must be non-empty (the resolve-first guidance, etc.).
    assert by_name["resolve_entity"].description
    assert by_name["search_corpus"].description


def test_parse_mcp_argv_requires_corpus() -> None:
    args = parse_mcp_argv(["--corpus", "/some/dir"])
    assert args.command == "mcp"
    assert args.corpus == "/some/dir"
    with pytest.raises(SystemExit):
        parse_mcp_argv([])
