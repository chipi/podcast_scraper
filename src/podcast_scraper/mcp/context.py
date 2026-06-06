"""MCP server runtime context (RFC-095): the corpus directory is the read context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusContext:
    """The single corpus directory an MCP server instance reads from.

    Set once at server start by the operator (not the agent), so tool calls are confined
    to this directory by construction — there is no agent-supplied path to validate.
    """

    corpus_dir: Path

    @classmethod
    def from_path(cls, path: Path | str) -> "CorpusContext":
        """Build a context from a corpus directory; raise if it is not a directory."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"corpus path is not a directory: {resolved}")
        return cls(corpus_dir=resolved)
