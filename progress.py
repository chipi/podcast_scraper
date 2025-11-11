from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, ContextManager, Optional, Protocol


class ProgressReporter(Protocol):
    """Minimal interface for progress callbacks."""

    def update(self, advance: int) -> None:
        ...


ProgressFactory = Callable[[Optional[int], str], ContextManager[ProgressReporter]]


class _NoopProgress:
    def update(self, advance: int) -> None:  # pragma: no cover - trivial
        return None


@contextmanager
def _noop_progress(total: Optional[int], description: str) -> ContextManager[ProgressReporter]:
    yield _NoopProgress()


_progress_factory: Optional[ProgressFactory] = None


def set_progress_factory(factory: Optional[ProgressFactory]) -> None:
    """Register a global factory for progress reporters."""

    global _progress_factory
    _progress_factory = factory or _noop_progress


@contextmanager
def progress_context(total: Optional[int], description: str) -> ContextManager[ProgressReporter]:
    """Return a context manager yielding the active progress reporter."""

    factory = _progress_factory or _noop_progress
    with factory(total, description) as reporter:
        yield reporter


# Backwards compatibility: old public API name
progress = progress_context


__all__ = [
    "ProgressReporter",
    "ProgressFactory",
    "progress",
    "progress_context",
    "set_progress_factory",
]
