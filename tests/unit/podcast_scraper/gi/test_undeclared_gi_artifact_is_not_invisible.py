"""An artifact search SERVES must not be invisible to the gate that clears the corpus.

THE BLIND SPOT
``assess_gi_integrity`` resolves an episode's artifact from its metadata's
``grounded_insights.artifact_path``. When that block is absent it files the episode under
"episodes with no GI block" — a legal state — and moves on WITHOUT LOOKING AT DISK.

But search does not resolve GI that way. ``_episode_to_gi_path_from_discovered``
(search/cli_handlers.py) tries the declared path first and, when it is absent or does not resolve
to a file, falls back to ``_determine_gi_path`` — the sibling-name convention, ``<name>.gi.json``
next to ``<name>.metadata.json`` — and serves it if it exists.

So an episode whose metadata declares nothing while a ``gi.json`` sits beside it is SERVED BY
SEARCH and reported as "GI never ran for it" by the gate. If that artifact is a pre-#1657
placeholder, the gate returns ``VERDICT: PASS`` for a corpus whose search results contain the
placeholder — the same class of false green as the ``rm``-and-pass failure that
``check_corpus_for_placeholders`` was retired for, arrived at from the opposite direction.

THE RULE ENCODED HERE
The gate must resolve artifacts the way the SERVING path resolves them. Anything else means the
gate is clearing a corpus that is not the corpus users query.

Severity split, because an undeclared artifact is two different problems:
  * undeclared AND bad (placeholder / unreadable) -> HARD FAILURE. It is being served.
  * undeclared AND good                           -> PASS, but reported. The artifact is fine;
    the missing declaration is a provenance gap, and silently "fixing" it is not this gate's job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.gi.corpus import LEGACY_PLACEHOLDER_INSIGHT_TEXT
from podcast_scraper.gi.integrity import assess_gi_integrity, check_corpus_gi_integrity

pytestmark = [pytest.mark.unit]


def _write(
    corpus: Path,
    *,
    episode_id: str,
    name: str,
    declare: bool,
    artifact_insights: Optional[List[str]],
    feed: str = "feed_a",
    run: str = "run_20260815-120000",
    artifact_body: Optional[str] = None,
) -> Path:
    """One episode. ``declare`` and the artifact's existence are INDEPENDENT here.

    That independence is the whole point: the existing fixture helper ties them together, which
    is precisely why this combination was never tested.
    """
    meta_dir = corpus / "feeds" / feed / run / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "episode": {"episode_id": episode_id, "title": name},
        "content": {"transcript_file_path": f"transcripts/{name}.txt"},
    }
    if declare:
        meta["grounded_insights"] = {
            "artifact_path": f"metadata/{name}.gi.json",
            "insight_count": len(artifact_insights or []),
            "schema_version": "3.1",
        }
    (meta_dir / f"{name}.metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    artifact = meta_dir / f"{name}.gi.json"
    if artifact_body is not None:
        artifact.write_text(artifact_body, encoding="utf-8")
    elif artifact_insights is not None:
        nodes: List[Dict[str, Any]] = [{"id": episode_id, "type": "Episode", "properties": {}}]
        for i, text in enumerate(artifact_insights):
            nodes.append(
                {"id": f"{episode_id}:i{i}", "type": "Insight", "properties": {"text": text}}
            )
        artifact.write_text(
            json.dumps({"episode_id": episode_id, "nodes": nodes, "edges": []}), encoding="utf-8"
        )
    return artifact


def test_undeclared_placeholder_fails_the_gate(tmp_path):
    """THE false green: search serves this placeholder, the gate called the episode legal."""
    _write(
        tmp_path,
        episode_id="ep-undeclared",
        name="0001 - Undeclared",
        declare=False,
        artifact_insights=[LEGACY_PLACEHOLDER_INSIGHT_TEXT],
    )

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok is False, f"gate passed a corpus serving a placeholder:\n{report}"
    assert "ep-undeclared" in report


def test_undeclared_unreadable_artifact_fails_the_gate(tmp_path):
    """A truncated artifact is served as a parse error, not as 'no GI' — it must fail."""
    _write(
        tmp_path,
        episode_id="ep-broken",
        name="0002 - Broken",
        declare=False,
        artifact_insights=None,
        artifact_body='{"episode_id": "ep-broken", "nodes": [',
    )

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok is False, f"gate passed a corpus with an unreadable served artifact:\n{report}"
    assert "ep-broken" in report


def test_undeclared_but_healthy_artifact_passes_and_is_reported(tmp_path):
    """A good artifact that is merely undeclared is a provenance gap, not corpus damage.

    It must PASS — failing here would block a repair for a cosmetic metadata issue — while still
    appearing in the report, because an undeclared artifact is how the two cases above arise.
    """
    _write(
        tmp_path,
        episode_id="ep-ok",
        name="0003 - Fine",
        declare=False,
        artifact_insights=["A genuinely useful claim about the episode."],
    )

    ok, report = check_corpus_gi_integrity(tmp_path)
    result = assess_gi_integrity(tmp_path)

    assert ok is True, report
    assert [e["episode_id"] for e in result["undeclared_artifact"]] == ["ep-ok"]
    # Match the report LINE, not the word. ``tmp_path`` is named after the test, so a bare
    # ``"undeclared" in report`` matches the "Corpus: /…/test_undeclared_but_healthy…" header and
    # passes no matter what the gate reports — a false green this very assertion had until it was
    # checked by hand against the rendered report.
    assert "served but NOT declared   : 1" in report, report


def test_no_block_and_no_artifact_is_still_legal(tmp_path):
    """The genuinely-no-GI case must be unchanged: GI never ran, nothing on disk, PASS."""
    _write(
        tmp_path,
        episode_id="ep-none",
        name="0004 - No GI",
        declare=False,
        artifact_insights=None,
    )

    ok, report = check_corpus_gi_integrity(tmp_path)
    result = assess_gi_integrity(tmp_path)

    assert ok is True, report
    assert result["episodes_without_gi_block"] == ["ep-none"]
    assert result["undeclared_artifact"] == []


def test_declared_path_still_wins_when_both_exist(tmp_path):
    """Precedence must match search's: declared first, sibling only as fallback.

    A gate that preferred the sibling would judge a different file from the one being served.
    """
    _write(
        tmp_path,
        episode_id="ep-declared",
        name="0005 - Declared",
        declare=True,
        artifact_insights=["The declared artifact's real insight."],
    )

    result = assess_gi_integrity(tmp_path)

    assert [e["episode_id"] for e in result["healthy"]] == ["ep-declared"]
    assert result["undeclared_artifact"] == []


def test_declared_but_missing_is_still_a_hard_failure(tmp_path):
    """Regression guard: the sibling probe must not mask a broken declaration.

    The declared path is absent AND no sibling exists, so this stays a missing-artifact failure
    rather than becoming "no GI block".
    """
    meta_dir = tmp_path / "feeds" / "feed_a" / "run_20260815-120000" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "0006 - Gone.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": "ep-gone", "title": "0006 - Gone"},
                "grounded_insights": {"artifact_path": "metadata/somewhere-else.gi.json"},
            }
        ),
        encoding="utf-8",
    )

    ok, report = check_corpus_gi_integrity(tmp_path)
    result = assess_gi_integrity(tmp_path)

    assert ok is False, report
    assert [e["episode_id"] for e in result["missing_artifact"]] == ["ep-gone"]
