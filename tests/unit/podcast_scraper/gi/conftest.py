"""Shared GI test fixtures.

WHY THIS EXISTS
Before #1657, ``build_artifact(episode_id, transcript)`` with no provider returned an artifact
containing one Insight, one Quote and a SUPPORTED_BY edge — because the pipeline manufactured a
placeholder insight ("Summary insight (stub).") whenever extraction produced nothing, and hung a
Quote off it sliced out of the transcript head by byte offset.

That made it a very convenient fixture, and dozens of tests used it as one. It also meant those
tests were asserting against invented content: the "insight" was a fixed string and the "quote"
supported no claim. Whole test files exercised a code path that only ran when the real one had
failed.

The placeholder is gone. An episode with no insights now gets an artifact with no Insight nodes,
which is the truth. So tests that need a populated artifact have to say what should be in it —
which is what these helpers do, and it is strictly better: they drive the REAL path (prefilled
insight texts, and grounding through the evidence stack) rather than a fallback nobody wants in
production.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from gi_artifacts import artifact_with_grounded_insights, artifact_with_insights


@pytest.fixture
def gi_artifact() -> Dict[str, Any]:
    """Insight-only artifact (no quotes)."""
    artifact: Dict[str, Any] = artifact_with_insights()
    return artifact


@pytest.fixture
def gi_grounded_artifact() -> Dict[str, Any]:
    """Full artifact: Insight + Quote + SUPPORTED_BY."""
    artifact: Dict[str, Any] = artifact_with_grounded_insights()
    return artifact
