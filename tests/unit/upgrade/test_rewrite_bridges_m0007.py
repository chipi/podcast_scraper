"""Bridges must follow m0007's ids — and must never invent one.

`*.bridge.json` carries the same `person:` ids as the graph layers, and `cil_queries.py` walks
every bridge at request time. After m0007 those ids are stale, silently: the surfaces return empty
rather than erroring, which is the exact failure #1685 exists to remove.

Four properties, all of which could plausibly go wrong:

  * a bare id is re-pointed at the SCOPED id, in `identities` and in `fuzzy_merges`
  * a bare id whose scoped form is ABSENT from the episode's graph is left alone and reported —
    the pass must never mint a target it cannot see, because that manufactures a reference rather
    than repairing one
  * substituting onto an id the bridge already holds MERGES, never emits two entries sharing an id
  * plan writes nothing
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade.rewrite_bridges_m0007 import apply_to_payload, plan_bridge, run

pytestmark = [pytest.mark.unit]


def _write(root: Path, stem: str, *, graph_persons: list, bridge_ids: list, fuzzy=None) -> None:
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    graph = {
        "schema_version": "3.1",
        "nodes": [{"id": p, "kind": "person", "name": p.split(":")[-1]} for p in graph_persons],
        "edges": [],
    }
    (meta / f"{stem}.gi.json").write_text(json.dumps(graph), encoding="utf-8")
    (meta / f"{stem}.kg.json").write_text(json.dumps(graph), encoding="utf-8")
    payload = {
        "schema_version": "1.0",
        "episode_id": stem,
        "emitted_at": "2026-08-27T00:00:00Z",
        "identities": [
            {
                "id": i,
                "type": "person",
                "display_name": i.split(":")[-1],
                "aliases": [],
                "sources": {"gi": True, "kg": False},
            }
            for i in bridge_ids
        ],
    }
    if fuzzy:
        payload["fuzzy_merges"] = fuzzy
    (meta / f"{stem}.bridge.json").write_text(json.dumps(payload), encoding="utf-8")


def _bridge(root: Path, stem: str) -> dict:
    doc: dict = json.loads((root / "metadata" / f"{stem}.bridge.json").read_text(encoding="utf-8"))
    return doc


class TestItFollowsTheMigration:
    def test_a_bare_id_is_repointed_at_the_scoped_one(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "ep1",
            graph_persons=["person:unresolved-jensen-ep1"],
            bridge_ids=["person:jensen"],
        )
        r = run(tmp_path, dry_run=False)
        assert r["changed"] == 1 and r["rewrites"] == 1, r
        assert [i["id"] for i in _bridge(tmp_path, "ep1")["identities"]] == [
            "person:unresolved-jensen-ep1"
        ]

    def test_fuzzy_merge_endpoints_are_rewritten_too(self, tmp_path: Path) -> None:
        """They hold `person:` ids as well; missing them leaves half the file stale."""
        _write(
            tmp_path,
            "ep1",
            graph_persons=["person:unresolved-jensen-ep1"],
            bridge_ids=["person:jensen"],
            fuzzy=[{"gi_id": "person:jensen", "kg_id": "person:jensen", "similarity": 0.9}],
        )
        run(tmp_path, dry_run=False)
        row = _bridge(tmp_path, "ep1")["fuzzy_merges"][0]
        assert row["gi_id"] == "person:unresolved-jensen-ep1"
        assert row["kg_id"] == "person:unresolved-jensen-ep1"

    def test_an_already_scoped_bridge_is_untouched(self, tmp_path: Path) -> None:
        """Idempotence: re-running after a partial pass must be a no-op."""
        _write(
            tmp_path,
            "ep1",
            graph_persons=["person:unresolved-jensen-ep1"],
            bridge_ids=["person:unresolved-jensen-ep1"],
        )
        assert run(tmp_path, dry_run=False)["changed"] == 0


class TestItNeverInventsATarget:
    def test_a_missing_scoped_id_is_left_alone_and_reported(self, tmp_path: Path) -> None:
        """The graph does not hold the id this WOULD become — so the two disagree for a reason
        this pass cannot explain, and guessing would manufacture a reference rather than repair
        one."""
        _write(
            tmp_path,
            "ep1",
            graph_persons=["person:someone-else"],
            bridge_ids=["person:jensen"],
        )
        r = run(tmp_path, dry_run=False)
        assert r["changed"] == 0, r
        assert r["unresolved"] == ["person:jensen"], r
        assert [i["id"] for i in _bridge(tmp_path, "ep1")["identities"]] == ["person:jensen"]


class TestItMergesRatherThanDuplicating:
    def test_a_collision_produces_one_entry(self, tmp_path: Path) -> None:
        """The bridge already holds the scoped id AND its bare form. Two entries sharing one id
        is a corrupt bridge — same hazard `rewrite_ids` handles for the graph layers."""
        payload = {
            "episode_id": "ep1",
            "identities": [
                {"id": "person:jensen", "aliases": ["J"], "sources": {"gi": True}},
                {
                    "id": "person:unresolved-jensen-ep1",
                    "aliases": ["Jensen"],
                    "sources": {"kg": True},
                },
            ],
        }
        out, changes = apply_to_payload(payload, {"person:jensen": "person:unresolved-jensen-ep1"})
        ids = [i["id"] for i in out["identities"]]
        assert ids == ["person:unresolved-jensen-ep1"], ids
        assert changes == 1
        merged = out["identities"][0]
        assert merged["aliases"] == ["J", "Jensen"]
        assert merged["sources"] == {"gi": True, "kg": True}


class TestPlanNeverWrites:
    def test_the_corpus_is_byte_identical(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "ep1",
            graph_persons=["person:unresolved-jensen-ep1"],
            bridge_ids=["person:jensen"],
        )
        before = {
            str(p.relative_to(tmp_path)): p.read_bytes()
            for p in sorted(tmp_path.rglob("*"))
            if p.is_file()
        }
        r = run(tmp_path, dry_run=True)
        assert r["changed"] == 1, "plan must still REPORT what it would do"
        after = {
            str(p.relative_to(tmp_path)): p.read_bytes()
            for p in sorted(tmp_path.rglob("*"))
            if p.is_file()
        }
        assert after == before


class TestABridgeWithoutAnEpisodeIdIsSkipped:
    def test_no_episode_id_means_no_target_can_be_computed(self) -> None:
        mapping, unresolved = plan_bridge(
            {"identities": [{"id": "person:jensen"}]}, {"person:unresolved-jensen-ep1"}
        )
        assert mapping == {} and unresolved == []
