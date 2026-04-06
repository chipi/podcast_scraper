"""CLI wiring for ``serve`` subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import podcast_scraper.cli as cli


@pytest.mark.unit
def test_parse_args_serve_delegates() -> None:
    args = cli.parse_args(["serve", "--output-dir", "/tmp/corpus", "--port", "9000"])
    assert args.command == "serve"
    assert args.output_dir == "/tmp/corpus"
    assert args.port == 9000


@pytest.mark.unit
def test_main_dispatches_serve(tmp_path: Path) -> None:
    with patch("podcast_scraper.server.cli_handlers.run_serve", return_value=0) as rs:
        code = cli.main(["serve", "--output-dir", str(tmp_path)])
    assert code == 0
    rs.assert_called_once()
    ns = rs.call_args[0][0]
    assert ns.command == "serve"
    assert ns.output_dir == str(tmp_path)
