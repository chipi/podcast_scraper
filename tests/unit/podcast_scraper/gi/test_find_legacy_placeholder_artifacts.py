"""The work-list for replacing stub artifacts (#1657 item 9 follow-up).

#9 stopped NEW stub artifacts from being written silently. It did nothing about the ones already
in the corpus — 112 of 678 production episodes — and those cannot be found by asking "does this
episode have GI?", because they do. ``episode_complete_for_append_resume`` asks exactly that, so
an append-mode re-run skips them indefinitely.

Detection keys on the insight TEXT rather than on the node's properties, and that choice is the
whole point: #9 changed the stub's properties (``grounded`` true→false, tier CORE→FILLER,
``routing_tag`` surface→drop). A detector written against the new shape would miss every
artifact written before the fix — which is the entire population needing repair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.gi.corpus import (
    find_legacy_placeholder_artifacts,
    is_legacy_placeholder_artifact,
    LEGACY_PLACEHOLDER_INSIGHT_TEXT,
    summarize_legacy_placeholder_artifacts,
)

pytestmark = [pytest.mark.unit]


def _artifact(episode_id: str, insight_texts: List[str], **insight_props: Any) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = [
        {"id": f"episode:{episode_id}", "type": "Episode", "properties": {}}
    ]
    for i, text in enumerate(insight_texts):
        props: Dict[str, Any] = {"text": text, "episode_id": episode_id}
        props.update(insight_props)
        nodes.append({"id": f"insight:{episode_id}:{i}", "type": "Insight", "properties": props})
    return {"schema_version": "3.1", "episode_id": episode_id, "nodes": nodes, "edges": []}


def _write(root: Path, name: str, doc: Dict[str, Any]) -> Path:
    p = root / f"{name}.gi.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


class TestItRecognisesAStub:
    def test_the_old_pre_fix_shape_is_detected(self) -> None:
        """The population that actually needs repair: written when the stub still claimed to be
        a grounded, CORE-tier, surfaced insight."""
        doc = _artifact(
            "ep:1",
            [LEGACY_PLACEHOLDER_INSIGHT_TEXT],
            grounded=True,
            tier=3,
            routing_tag="surface",
            salience=1.0,
        )
        assert is_legacy_placeholder_artifact(doc) is True

    def test_the_new_post_fix_shape_is_also_detected(self) -> None:
        doc = _artifact(
            "ep:1",
            [LEGACY_PLACEHOLDER_INSIGHT_TEXT],
            grounded=False,
            tier=0,
            routing_tag="drop",
            salience=0.0,
        )
        assert is_legacy_placeholder_artifact(doc) is True

    def test_a_real_single_insight_episode_is_not_a_stub(self) -> None:
        """The false positive that would matter: an episode that genuinely produced one good
        insight must not be queued for re-derivation. Re-running it costs money and risks
        replacing a real result with a worse one."""
        doc = _artifact("ep:2", ["Ben Horowitz argues that founders should stay technical."])
        assert is_legacy_placeholder_artifact(doc) is False

    def test_a_healthy_multi_insight_episode_is_not_a_stub(self) -> None:
        doc = _artifact("ep:3", ["First real insight.", "Second real insight."])
        assert is_legacy_placeholder_artifact(doc) is False

    def test_a_stub_alongside_real_insights_is_not_a_stub_artifact(self) -> None:
        """Only a LONE stub means generation failed. If real insights exist beside it the
        episode has content, and the definition stays strict rather than greedy."""
        doc = _artifact("ep:4", [LEGACY_PLACEHOLDER_INSIGHT_TEXT, "A real insight."])
        assert is_legacy_placeholder_artifact(doc) is False

    def test_an_artifact_with_no_insights_is_not_a_stub(self) -> None:
        assert is_legacy_placeholder_artifact(_artifact("ep:5", [])) is False

    def test_whitespace_around_the_text_still_matches(self) -> None:
        assert (
            is_legacy_placeholder_artifact(
                _artifact("ep:6", [f"  {LEGACY_PLACEHOLDER_INSIGHT_TEXT}  "])
            )
            is True
        )

    def test_a_malformed_artifact_does_not_raise(self) -> None:
        """This runs across a whole corpus; one bad file must not stop the scan."""
        assert is_legacy_placeholder_artifact({}) is False
        assert is_legacy_placeholder_artifact({"nodes": None}) is False
        assert is_legacy_placeholder_artifact({"nodes": ["not-a-dict"]}) is False


class TestItScansACorpus:
    def _corpus(self, tmp_path: Path) -> Path:
        _write(
            tmp_path / "feedA" / "metadata",
            "ep1",
            _artifact("ep:1", [LEGACY_PLACEHOLDER_INSIGHT_TEXT]),
        )
        _write(tmp_path / "feedA" / "metadata", "ep2", _artifact("ep:2", ["Real one.", "Two."]))
        _write(
            tmp_path / "feedB" / "metadata",
            "ep3",
            _artifact("ep:3", [LEGACY_PLACEHOLDER_INSIGHT_TEXT]),
        )
        _write(tmp_path / "feedB" / "metadata", "ep4", _artifact("ep:4", ["Real."]))
        return tmp_path

    def test_it_finds_stubs_across_feeds(self, tmp_path: Path) -> None:
        found = find_legacy_placeholder_artifacts(self._corpus(tmp_path))
        assert [eid for _, eid in found] == ["ep:1", "ep:3"]

    def test_it_returns_paths_that_exist(self, tmp_path: Path) -> None:
        for path, _ in find_legacy_placeholder_artifacts(self._corpus(tmp_path)):
            assert path.is_file()

    def test_the_order_is_stable(self, tmp_path: Path) -> None:
        """Two scans of one corpus must produce the same work-list, or a repair run cannot be
        resumed or diffed."""
        root = self._corpus(tmp_path)
        assert find_legacy_placeholder_artifacts(root) == find_legacy_placeholder_artifacts(root)

    def test_the_summary_counts_and_shares_are_right(self, tmp_path: Path) -> None:
        s = summarize_legacy_placeholder_artifacts(self._corpus(tmp_path))
        assert s["artifacts_total"] == 4
        assert s["legacy_placeholders"] == 2
        assert s["legacy_placeholder_share"] == 0.5
        assert sorted(s["episode_ids"]) == ["ep:1", "ep:3"]

    def test_an_empty_corpus_is_zero_not_a_crash(self, tmp_path: Path) -> None:
        s = summarize_legacy_placeholder_artifacts(tmp_path)
        assert s == {
            "artifacts_total": 0,
            "legacy_placeholders": 0,
            "legacy_placeholder_share": 0.0,
            "episode_ids": [],
            "paths": [],
        }

    def test_an_unreadable_file_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        root = self._corpus(tmp_path)
        (root / "feedA" / "metadata" / "broken.gi.json").write_text("{not json", encoding="utf-8")
        found = find_legacy_placeholder_artifacts(root)
        assert [eid for _, eid in found] == ["ep:1", "ep:3"]


class TestAgainstTheRealAcceptanceCorpus:
    """Sanity check on real artifacts, not synthetic ones."""

    ROOT = Path("/Users/claude/podcast-acceptance-corpus/feeds")

    def test_the_local_acceptance_run_has_no_stubs(self) -> None:
        """Its 15 episodes carry 3-114 insights each; none should be flagged. If this ever
        fires, the detector has become greedy."""
        if not self.ROOT.is_dir():
            pytest.skip("acceptance corpus not present on this machine")
        assert find_legacy_placeholder_artifacts(self.ROOT) == []
