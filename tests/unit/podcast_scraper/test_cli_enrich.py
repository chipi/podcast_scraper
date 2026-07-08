"""CLI wiring for the ``enrich`` main-CLI subcommand (#1069 consistency).

``podcast_scraper enrich <args>`` delegates to the enrichment CLI verbatim, so
enrichment invokes / schedules / runs in docker exactly like the pipeline.
"""

from __future__ import annotations

import pytest

from podcast_scraper import cli

pytestmark = [pytest.mark.unit]


def test_parse_args_routes_enrich_passthrough() -> None:
    args = cli.parse_args(["enrich", "--output-dir", "/tmp/c", "--only", "grounding_rate"])
    assert args.command == "enrich"
    assert args.enrich_argv == ["--output-dir", "/tmp/c", "--only", "grounding_rate"]


def test_enrich_subcommand_delegates_to_enrichment_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def _fake_enrich_main(argv: list[str]) -> int:
        seen.append(list(argv))
        return 0

    monkeypatch.setattr("podcast_scraper.enrichment.cli.main", _fake_enrich_main)
    rc = cli.main(["enrich", "--output-dir", "/tmp/c", "--corpus-only"])

    assert rc == 0
    assert seen == [["--output-dir", "/tmp/c", "--corpus-only"]]
