"""Where the pipeline calls a stage it does not own (E04 S2).

The run and the multi-feed batch call these points by name and never import the stage code
behind them. Each point runs the built-in hooks first, then each installed extension's, in load
order. A hook owns its own guard: a finalize step that must never fail the run catches its own
errors, and a start-up check that must stop the run raises. The runner adds no guard of its own,
so moving a hook behind the seam changes nothing about how it fails.

The built-in hooks are today's in-tree stages (:mod:`.builtin_stage_hooks`). When a stage's code
moves to an extension, its built-in entry goes and the extension's ``pipeline_hooks`` carries it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

#: ``(cfg)``. Before any feed is fetched; raise to stop the run (a missing dependency).
RunStartedHook = Callable[[Any], None]
#: ``(corpus_root)``. Corpus-wide derivations from what is on disk, once the run's (or the
#: batch's) index is built: single-run finalize at the run's corpus root, or batch finalize at
#: the corpus parent. Must not raise.
CorpusDerivedHook = Callable[[Path], None]
#: ``(corpus_parent)``. Once per multi-feed batch, after every feed finished, before the
#: enrichment follow-up is queued. Must not raise.
BatchFinalizedHook = Callable[[Path], None]
#: ``(corpus_root, run_dir)``. End of a run, before local audio is evicted: the last moment the
#: run's audio is on local disk. Must not raise.
RunFinalizedHook = Callable[[str, str], None]


@dataclass(frozen=True)
class PipelineHooks:
    """The stage hooks one owner contributes; every part is optional."""

    name: str
    run_started: Sequence[RunStartedHook] = field(default_factory=tuple)
    corpus_derived: Sequence[CorpusDerivedHook] = field(default_factory=tuple)
    batch_finalized: Sequence[BatchFinalizedHook] = field(default_factory=tuple)
    run_finalized: Sequence[RunFinalizedHook] = field(default_factory=tuple)


def _all() -> Iterator[PipelineHooks]:
    from podcast_scraper.extensions import load_extensions

    from .builtin_stage_hooks import BUILTIN

    yield BUILTIN
    for ext in load_extensions():
        if ext.pipeline_hooks is not None:
            yield ext.pipeline_hooks


def run_started(cfg: Any) -> None:
    for hooks in _all():
        for hook in hooks.run_started:
            hook(cfg)


def corpus_derived(corpus_root: Path) -> None:
    for hooks in _all():
        for hook in hooks.corpus_derived:
            hook(corpus_root)


def batch_finalized(corpus_parent: Path) -> None:
    for hooks in _all():
        for hook in hooks.batch_finalized:
            hook(corpus_parent)


def run_finalized(corpus_root: str, run_dir: str) -> None:
    for hooks in _all():
        for hook in hooks.run_finalized:
            hook(corpus_root, run_dir)


__all__ = [
    "BatchFinalizedHook",
    "CorpusDerivedHook",
    "PipelineHooks",
    "RunFinalizedHook",
    "RunStartedHook",
    "batch_finalized",
    "corpus_derived",
    "run_finalized",
    "run_started",
]
