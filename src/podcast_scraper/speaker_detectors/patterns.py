"""Episode pattern analysis stub (backward compatibility)."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from .constants import DEFAULT_SAMPLE_SIZE


def analyze_episode_patterns(
    episodes: List[Any],
    nlp: Any,
    cached_hosts: Set[str],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """No-op — heuristic pattern learner removed in #598 simplification."""
    _ = episodes, nlp, cached_hosts, sample_size
    return {}
