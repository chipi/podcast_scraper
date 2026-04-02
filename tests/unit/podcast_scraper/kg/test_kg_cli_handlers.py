"""Direct unit tests for ``kg.cli_handlers`` (exit codes and path resolution)."""

from __future__ import annotations

import json
import logging
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from podcast_scraper.kg import cli_handlers
from podcast_scraper.kg.corpus import EXIT_INVALID_ARGS, EXIT_NO_ARTIFACTS, EXIT_SUCCESS

_LOG = logging.getLogger("test_kg_cli_handlers")


@pytest.fixture
def minimal_kg_path() -> Path:
    return Path(__file__).resolve().parents[3] / "fixtures" / "kg" / "minimal.kg.json"


@pytest.mark.unit
class TestResolveKgArtifactPath:
    """resolve_kg_artifact_path."""

    def test_episode_path_file_kg_json(self, minimal_kg_path: Path) -> None:
        args = Namespace(episode_path=str(minimal_kg_path), output_dir=None, episode_id=None)
        assert cli_handlers.resolve_kg_artifact_path(args) == minimal_kg_path.resolve()

    def test_episode_path_dir_first_kg(self, tmp_path: Path, minimal_kg_path: Path) -> None:
        shutil.copy(minimal_kg_path, tmp_path / "only.kg.json")
        args = Namespace(episode_path=str(tmp_path), output_dir=None, episode_id=None)
        assert cli_handlers.resolve_kg_artifact_path(args) == (tmp_path / "only.kg.json")

    def test_output_dir_and_episode_id(self, tmp_path: Path, minimal_kg_path: Path) -> None:
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        shutil.copy(minimal_kg_path, meta / "x.kg.json")
        args = Namespace(
            episode_path=None,
            output_dir=str(tmp_path),
            episode_id="fixture:minimal-kg",
        )
        assert cli_handlers.resolve_kg_artifact_path(args) == meta / "x.kg.json"


@pytest.mark.unit
class TestRunKgValidate:
    """run_kg_validate."""

    def test_no_paths_returns_invalid_args(self) -> None:
        args = Namespace(paths=[], strict=False, quiet=True)
        assert cli_handlers.run_kg_validate(args, _LOG) == EXIT_INVALID_ARGS

    def test_no_kg_files_under_dir(self, tmp_path: Path) -> None:
        args = Namespace(paths=[str(tmp_path)], strict=False, quiet=True)
        assert cli_handlers.run_kg_validate(args, _LOG) == EXIT_NO_ARTIFACTS

    def test_fixture_passes(self, minimal_kg_path: Path) -> None:
        args = Namespace(paths=[str(minimal_kg_path)], strict=True, quiet=True)
        assert cli_handlers.run_kg_validate(args, _LOG) == EXIT_SUCCESS


@pytest.mark.unit
class TestRunKgInspect:
    """run_kg_inspect."""

    def test_missing_identifiers_returns_error(self) -> None:
        args = Namespace(
            episode_path=None,
            output_dir=None,
            episode_id=None,
            strict=False,
            format="pretty",
        )
        assert cli_handlers.run_kg_inspect(args, _LOG) == 1

    def test_output_dir_without_episode_id_returns_error(self) -> None:
        args = Namespace(
            episode_path=None,
            output_dir="/tmp",
            episode_id=None,
            strict=False,
            format="pretty",
        )
        assert cli_handlers.run_kg_inspect(args, _LOG) == 1

    def test_json_format_success(
        self, minimal_kg_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        args = Namespace(
            episode_path=str(minimal_kg_path),
            output_dir=None,
            episode_id=None,
            strict=False,
            format="json",
        )
        assert cli_handlers.run_kg_inspect(args, _LOG) == EXIT_SUCCESS
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload.get("episode_id") == "fixture:minimal-kg"


@pytest.mark.unit
class TestRunKgExportEntitiesTopics:
    """run_kg_export / entities / topics — required dirs and empty corpus."""

    def test_export_missing_output_dir(self) -> None:
        args = Namespace(output_dir=None, strict=False, format="ndjson", out=None)
        assert cli_handlers.run_kg_export(args, _LOG) == EXIT_INVALID_ARGS

    def test_export_dir_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope"
        args = Namespace(output_dir=str(missing), strict=False, format="ndjson", out=None)
        assert cli_handlers.run_kg_export(args, _LOG) == EXIT_NO_ARTIFACTS

    def test_export_no_artifacts(self, tmp_path: Path) -> None:
        args = Namespace(output_dir=str(tmp_path), strict=False, format="ndjson", out=None)
        assert cli_handlers.run_kg_export(args, _LOG) == EXIT_NO_ARTIFACTS

    def test_export_ndjson_stdout_success(
        self, tmp_path: Path, minimal_kg_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        shutil.copy(minimal_kg_path, meta / "a.kg.json")
        args = Namespace(output_dir=str(tmp_path), strict=False, format="ndjson", out=None)
        assert cli_handlers.run_kg_export(args, _LOG) == EXIT_SUCCESS
        assert "fixture:minimal-kg" in capsys.readouterr().out

    def test_entities_missing_output_dir(self) -> None:
        args = Namespace(
            output_dir=None,
            strict=False,
            format="pretty",
            min_episodes=1,
        )
        assert cli_handlers.run_kg_entities(args, _LOG) == EXIT_INVALID_ARGS

    def test_entities_json_success(self, tmp_path: Path, minimal_kg_path: Path) -> None:
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        shutil.copy(minimal_kg_path, meta / "a.kg.json")
        args = Namespace(
            output_dir=str(tmp_path),
            strict=False,
            format="json",
            min_episodes=1,
        )
        # Single episode: entity rollup may be empty list but still success
        assert cli_handlers.run_kg_entities(args, _LOG) == EXIT_SUCCESS

    def test_topics_missing_output_dir(self) -> None:
        args = Namespace(
            output_dir=None,
            strict=False,
            format="pretty",
            min_support=1,
        )
        assert cli_handlers.run_kg_topics(args, _LOG) == EXIT_INVALID_ARGS

    def test_topics_pretty_success(self, tmp_path: Path, minimal_kg_path: Path) -> None:
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        shutil.copy(minimal_kg_path, meta / "a.kg.json")
        args = Namespace(
            output_dir=str(tmp_path),
            strict=False,
            format="pretty",
            min_support=1,
        )
        assert cli_handlers.run_kg_topics(args, _LOG) == EXIT_SUCCESS


@pytest.mark.unit
class TestRunKgDispatch:
    """run_kg dispatcher."""

    def test_unknown_subcommand(self) -> None:
        args = Namespace(kg_subcommand="nope")
        assert cli_handlers.run_kg(args, _LOG) == 1
