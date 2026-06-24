"""#1058 chunk 3 — CLI handler for ``cluster-corpus-topics``.

Smoke-tests the CLI scaffolding that exposes
``kg/topic_clustering.cluster_and_apply_corpus_topics`` to operators.
The clustering algorithm itself is exercised by
``tests/unit/podcast_scraper/kg/test_topic_clustering.py``; these
tests only assert that argument parsing, error codes, and the
end-to-end CLI invocation produce the expected exit code + output.
"""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.search.cli_handlers import (
    parse_cluster_corpus_topics_argv,
    run_cluster_corpus_topics_cli,
)

pytestmark = pytest.mark.unit

_LOG = logging.getLogger("test-cluster-corpus-topics")


def _write_kg(corpus: Path, podcast: str, episode: str, labels: List[str]) -> Path:
    meta = corpus / "feeds" / podcast / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    path = meta / f"{podcast}_{episode}.kg.json"
    data: Dict[str, Any] = {
        "schema_version": "2.0",
        "episode_id": f"episode:{podcast}-{episode}",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2024-01-01T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [
            {
                "id": f"episode:{podcast}-{episode}",
                "type": "Episode",
                "properties": {
                    "podcast_id": f"podcast:{podcast}",
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": f"podcast:{podcast}",
                "type": "Podcast",
                "properties": {"title": podcast},
            },
        ],
        "edges": [],
    }
    for i, label in enumerate(labels):
        data["nodes"].append(
            {
                "id": f"topic:{podcast}-{episode}-t{i}",
                "type": "Topic",
                "properties": {"label": label},
            }
        )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


class TestParseClusterCorpusTopicsArgv:
    def test_required_output_dir(self) -> None:
        with pytest.raises(SystemExit):
            parse_cluster_corpus_topics_argv([])

    def test_defaults_when_optional_args_omitted(self) -> None:
        args = parse_cluster_corpus_topics_argv(["--output-dir", "/tmp/corpus"])
        assert args.command == "cluster-corpus-topics"
        assert args.output_dir == "/tmp/corpus"
        assert args.threshold == 0.75
        assert args.min_episodes == 2
        assert args.dry_run is False

    def test_threshold_and_dry_run_parsed(self) -> None:
        args = parse_cluster_corpus_topics_argv(
            [
                "--output-dir",
                "/tmp/corpus",
                "--threshold",
                "0.5",
                "--min-episodes",
                "3",
                "--dry-run",
            ]
        )
        assert args.threshold == 0.5
        assert args.min_episodes == 3
        assert args.dry_run is True


class TestRunClusterCorpusTopicsCli:
    def test_missing_output_dir_returns_invalid_args(self) -> None:
        # SimpleNamespace without ``output_dir`` triggers the early exit.
        rc = run_cluster_corpus_topics_cli(Namespace(), _LOG)
        assert rc == 2  # EXIT_INVALID_ARGS

    def test_nonexistent_corpus_returns_no_artifacts(self, tmp_path: Path) -> None:
        rc = run_cluster_corpus_topics_cli(
            parse_cluster_corpus_topics_argv(["--output-dir", str(tmp_path / "does-not-exist")]),
            _LOG,
        )
        assert rc == 3  # EXIT_NO_ARTIFACTS

    def test_empty_corpus_returns_success_with_zero_clusters(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """No KG files → no clusters, but the CLI still exits 0 (no
        artifacts to mutate is a valid no-op outcome)."""
        rc = run_cluster_corpus_topics_cli(
            parse_cluster_corpus_topics_argv(["--output-dir", str(tmp_path)]),
            _LOG,
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "clusters=0" in out

    def test_dry_run_does_not_mutate_artifacts(self, tmp_path: Path) -> None:
        # Two episodes from different shows share a topic label → would
        # cluster, but --dry-run keeps the disk artifacts untouched.
        path_a = _write_kg(tmp_path, "show-a", "ep1", ["shared topic"])
        path_b = _write_kg(tmp_path, "show-b", "ep1", ["shared topic"])
        before_a = path_a.read_text()
        before_b = path_b.read_text()
        rc = run_cluster_corpus_topics_cli(
            parse_cluster_corpus_topics_argv(
                ["--output-dir", str(tmp_path), "--dry-run", "--threshold", "0.0"]
            ),
            _LOG,
        )
        assert rc == 0
        assert path_a.read_text() == before_a
        assert path_b.read_text() == before_b
