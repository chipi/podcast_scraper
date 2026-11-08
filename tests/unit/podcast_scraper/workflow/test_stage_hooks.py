"""The pipeline's stage hooks (E04 S2): the run and the batch reach stages they do not own only
through these points, built-in hooks first, then each extension's, with no guard added."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import patch

import pytest

from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.workflow import builtin_stage_hooks, stage_hooks
from podcast_scraper.workflow.stage_hooks import PipelineHooks

pytestmark = pytest.mark.unit

_WORKFLOW = Path(stage_hooks.__file__).parent


def _recording(calls: List[Tuple[str, Tuple[Any, ...]]], who: str) -> PipelineHooks:
    def rec(point: str) -> Any:
        def hook(*args: Any) -> None:
            calls.append((f"{who}.{point}", args))

        return hook

    return PipelineHooks(
        name=who,
        run_started=(rec("run_started"),),
        corpus_derived=(rec("corpus_derived"),),
        batch_finalized=(rec("batch_finalized"),),
        run_finalized=(rec("run_finalized"),),
    )


def test_built_in_hooks_run_before_each_extension_in_load_order(tmp_path: Path) -> None:
    calls: List[Tuple[str, Tuple[Any, ...]]] = []
    exts = [
        Extension(name="a", pipeline_hooks=_recording(calls, "a")),
        Extension(name="none"),
        Extension(name="b", pipeline_hooks=_recording(calls, "b")),
    ]
    with (
        patch.object(builtin_stage_hooks, "BUILTIN", _recording(calls, "builtin")),
        use_extensions(exts),
    ):
        stage_hooks.run_started("cfg")
        stage_hooks.corpus_derived(tmp_path)
        stage_hooks.batch_finalized(tmp_path)
        stage_hooks.run_finalized("root", "run")

    assert calls == [
        ("builtin.run_started", ("cfg",)),
        ("a.run_started", ("cfg",)),
        ("b.run_started", ("cfg",)),
        ("builtin.corpus_derived", (tmp_path,)),
        ("a.corpus_derived", (tmp_path,)),
        ("b.corpus_derived", (tmp_path,)),
        ("builtin.batch_finalized", (tmp_path,)),
        ("a.batch_finalized", (tmp_path,)),
        ("b.batch_finalized", (tmp_path,)),
        ("builtin.run_finalized", ("root", "run")),
        ("a.run_finalized", ("root", "run")),
        ("b.run_finalized", ("root", "run")),
    ]


def test_a_raising_hook_stops_the_point_as_it_did_inline() -> None:
    """The runner adds no guard: a start-up check that raised before the seam still stops the run,
    and the hooks after it do not run."""
    later: List[str] = []

    def missing_dependency(cfg: Any) -> None:
        raise ImportError("grounding needs sentence-transformers")

    builtin = PipelineHooks(name="builtin", run_started=(missing_dependency,))
    ext = Extension(name="x", pipeline_hooks=PipelineHooks(name="x", run_started=(later.append,)))
    with patch.object(builtin_stage_hooks, "BUILTIN", builtin), use_extensions([ext]):
        with pytest.raises(ImportError, match="sentence-transformers"):
            stage_hooks.run_started("cfg")
    assert later == []


def test_the_built_in_hooks_call_todays_stages(tmp_path: Path) -> None:
    with (
        patch("podcast_scraper.gi.deps.validate_gil_grounding_dependencies") as deps,
        patch("podcast_scraper.providers.ml.diarization.ad_signatures.write_for_corpus") as ads,
        patch("podcast_scraper.quality.naming_coverage.measure_and_record") as coverage,
        patch("podcast_scraper.audio_sync.references.write_for_run") as refs,
        use_extensions([]),
    ):
        stage_hooks.run_started("cfg")
        stage_hooks.corpus_derived(tmp_path)
        stage_hooks.batch_finalized(tmp_path)
        stage_hooks.run_finalized("root", "run")
    deps.assert_called_once_with("cfg")
    ads.assert_called_once_with(tmp_path)
    coverage.assert_called_once_with(tmp_path)
    refs.assert_called_once_with("root", "run")


def test_a_failing_audio_reference_pass_never_fails_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch("podcast_scraper.audio_sync.references.write_for_run", side_effect=OSError("disk")),
        use_extensions([]),
    ):
        stage_hooks.run_finalized("root", "run")
    assert "audio references: end-of-run pass failed (non-fatal)" in caplog.text


@pytest.mark.parametrize(
    "module, moved",
    [
        ("orchestration.py", {"gi.deps", "audio_sync.references", "diarization.ad_signatures"}),
        ("corpus_operations.py", {"quality.naming_coverage", "diarization.ad_signatures"}),
    ],
)
def test_the_run_and_the_batch_no_longer_import_the_hooked_stages(module: str, moved: set) -> None:
    tree = ast.parse((_WORKFLOW / module).read_text(encoding="utf-8"))
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported for s in moved if m.endswith(s)}
