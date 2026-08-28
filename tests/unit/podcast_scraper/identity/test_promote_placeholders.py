"""Promoting a placeholder is the irreversible branch — it must refuse everything uncertain.

`person:unresolved-brandon-{ep}` beside `person:brandon-anderson` is a question the graph already
answers. Promoting it merges the placeholder's content onto a REAL person's global id, and there is
no cheap undo — get it wrong and one person's words are attributed to another, permanently.

So the rules are deliberately narrow, and each is pinned here:

  * exactly ONE node-backed candidate promotes. Zero is the ordinary case; two or more is genuine
    ambiguity and is REFUSED, never guessed — that is #1801's enricher problem, not this one's.
  * candidates must be NODE-BACKED (#1868). An id present only as an edge endpoint or a quote's
    `speaker_id` is a dangling reference, not evidence a person is in the episode.
  * `plan` writes nothing, so all 12 production cases are readable before any of them is applied.
  * both graph layers move together, or the two disagree about who this person is (#1862).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.identity.promote_placeholders import main, plan_promotions, run

pytestmark = [pytest.mark.unit]

EP = "substack-post-212802973"
PH = f"person:unresolved-brandon-{EP}"
REAL = "person:brandon-anderson"


def _nodes(*ids):
    return [{"id": i, "kind": "person", "properties": {"name": i.split(":")[-1]}} for i in ids]


def _episode(root: Path, name: str, *, gi_ids, kg_ids=None, gi_edges=()):
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"{name}.gi.json").write_text(
        json.dumps({"schema_version": "3.1", "nodes": _nodes(*gi_ids), "edges": list(gi_edges)}),
        encoding="utf-8",
    )
    if kg_ids is not None:
        (meta / f"{name}.kg.json").write_text(
            json.dumps({"schema_version": "2.0", "nodes": _nodes(*kg_ids), "edges": []}),
            encoding="utf-8",
        )


def _ids(root: Path, name: str, layer: str) -> set:
    doc = json.loads((root / "metadata" / f"{name}.{layer}.json").read_text(encoding="utf-8"))
    return {n["id"] for n in doc["nodes"]}


class TestItPromotesTheUnambiguousCase:
    def test_the_production_shape(self, tmp_path: Path) -> None:
        """`unresolved-brandon-… -> brandon-anderson`, one of the real 12."""
        _episode(tmp_path, "ep1", gi_ids=[PH, REAL], kg_ids=[PH, REAL])
        r = run(tmp_path, dry_run=False)
        assert r["promotions"] == 1, r
        assert _ids(tmp_path, "ep1", "gi") == {REAL}
        assert _ids(tmp_path, "ep1", "kg") == {REAL}, "both layers, or they disagree (#1862)"

    def test_the_placeholders_content_survives_the_merge(self, tmp_path: Path) -> None:
        """The promotion MERGES — the placeholder's properties must not be dropped on the floor."""
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        (meta / "ep1.gi.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"id": PH, "properties": {"name": "Brandon", "role": "guest"}},
                        {"id": REAL, "properties": {"name": "Brandon Anderson"}},
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        run(tmp_path, dry_run=False)
        doc = json.loads((meta / "ep1.gi.json").read_text(encoding="utf-8"))
        assert len(doc["nodes"]) == 1
        assert doc["nodes"][0]["properties"].get("role") == "guest", doc["nodes"][0]


class TestItRefusesEverythingUncertain:
    def test_two_candidates_are_refused_not_guessed(self, tmp_path: Path) -> None:
        _episode(
            tmp_path,
            "ep1",
            gi_ids=[f"person:unresolved-trump-{EP}", "person:donald-trump", "person:eric-trump"],
        )
        r = run(tmp_path, dry_run=False)
        assert r["promotions"] == 0
        assert any("unresolved-trump" in x for x in r["refused"]), r["refused"]

    def test_no_candidate_is_left_alone(self, tmp_path: Path) -> None:
        """The ~192 orphans. Nothing in the episode to promote to — genuinely #1801's job."""
        _episode(tmp_path, "ep1", gi_ids=[f"person:unresolved-jensen-{EP}"])
        r = run(tmp_path, dry_run=False)
        assert r["promotions"] == 0 and r["refused"] == []

    def test_an_edge_only_candidate_is_NOT_evidence(self, tmp_path: Path) -> None:
        """#1868's rule, applied here too: a dangling id must not license an irreversible write."""
        _episode(
            tmp_path,
            "ep1",
            gi_ids=[PH],
            gi_edges=[{"source": REAL, "target": "episode:e1"}],
        )
        assert run(tmp_path, dry_run=False)["promotions"] == 0

    def test_a_bare_id_is_not_a_placeholder_and_is_untouched(self, tmp_path: Path) -> None:
        """This operation starts from placeholders only; scoping bare names is m0007's job."""
        _episode(tmp_path, "ep1", gi_ids=["person:brandon", REAL])
        assert run(tmp_path, dry_run=False)["promotions"] == 0
        assert _ids(tmp_path, "ep1", "gi") == {"person:brandon", REAL}


class TestPlanNeverWrites:
    def test_bytes_identical_but_still_reported(self, tmp_path: Path) -> None:
        _episode(tmp_path, "ep1", gi_ids=[PH, REAL], kg_ids=[PH, REAL])
        before = {
            str(p.relative_to(tmp_path)): p.read_bytes()
            for p in sorted(tmp_path.rglob("*"))
            if p.is_file()
        }
        r = run(tmp_path, dry_run=True)
        assert r["promotions"] == 1, "plan must still say what it would do"
        after = {
            str(p.relative_to(tmp_path)): p.read_bytes()
            for p in sorted(tmp_path.rglob("*"))
            if p.is_file()
        }
        assert after == before


class TestItIsIdempotent:
    def test_a_second_apply_changes_nothing(self, tmp_path: Path) -> None:
        _episode(tmp_path, "ep1", gi_ids=[PH, REAL], kg_ids=[PH, REAL])
        run(tmp_path, dry_run=False)
        assert run(tmp_path, dry_run=False)["promotions"] == 0


class TestAWrongPathIsLoud:
    def test_missing_corpus_exits_nonzero(self, tmp_path: Path, capsys) -> None:
        assert main(["--corpus-root", str(tmp_path / "nope"), "--mode", "apply"]) == 1
        assert "does not exist" in capsys.readouterr().err


class TestThePlanIsReadable:
    def test_every_promotion_is_named_in_the_output(self, tmp_path: Path, capsys) -> None:
        """All 12 must be reviewable BEFORE applying — that is what makes this safe now when the
        blanket heal=false decision said not to."""
        _episode(tmp_path, "ep1", gi_ids=[PH, REAL])
        assert main(["--corpus-root", str(tmp_path), "--mode", "plan"]) == 0
        out = capsys.readouterr().out
        assert "unresolved-brandon" in out and "brandon-anderson" in out


class TestThePureFunction:
    """`plan_promotions` decides; `run` does I/O. Testing the decision without the filesystem
    keeps the rule readable and is how a caller other than the CLI would use it."""

    def test_it_returns_the_mapping_and_refuses_separately(self) -> None:
        gi = {"nodes": _nodes(PH, REAL), "edges": []}
        mapping, refused = plan_promotions(gi, {})
        assert mapping == {PH: REAL}
        assert refused == []

    def test_ambiguity_lands_in_refused_not_in_the_mapping(self) -> None:
        ph = f"person:unresolved-trump-{EP}"
        gi = {"nodes": _nodes(ph, "person:donald-trump", "person:eric-trump"), "edges": []}
        mapping, refused = plan_promotions(gi, {})
        assert mapping == {}
        assert len(refused) == 1 and "donald-trump" in refused[0] and "eric-trump" in refused[0]

    def test_it_unions_both_layers_before_deciding(self) -> None:
        """The placeholder in GI, the real person in KG — one episode, one decision."""
        mapping, _ = plan_promotions(
            {"nodes": _nodes(PH), "edges": []}, {"nodes": _nodes(REAL), "edges": []}
        )
        assert mapping == {PH: REAL}


class TestTheVoiceConflictRefusal:
    """The failure the one-candidate rule is blind to, and the signal that catches it.

    Two people named Brandon in one episode — a guest `brandon-anderson` with his own quotes, and
    a second diarized voice that scoping turned into `unresolved-brandon-{ep}`. Exactly one
    candidate, so the rule promotes, and the OTHER Brandon's words are silently reattributed to
    the guest. Constructed against the real code before this guard existed; it merged.

    Both carrying voice attribution means the extractor separated two speaker identities. That is
    either two people (worst case) or one voice labelled inconsistently (merge is right), and the
    artifacts cannot distinguish them — so it refuses. Real shapes: `SPOKEN_BY` edges
    (`{"from": "quote:…", "to": "person:…"}`) and quote `properties.speaker_id`, both present at
    scale (232 of each across 36 prod-shaped GI files).
    """

    @staticmethod
    def _two_brandons():
        return {
            "nodes": [
                {"id": PH, "kind": "person", "properties": {"name": "Brandon"}},
                {"id": REAL, "kind": "person", "properties": {"name": "Brandon Anderson"}},
                {"id": "quote:1", "type": "Quote", "properties": {"speaker_id": REAL, "text": "a"}},
                {"id": "quote:2", "type": "Quote", "properties": {"speaker_id": PH, "text": "b"}},
            ],
            "edges": [],
        }

    def test_both_speaking_is_refused(self) -> None:
        mapping, refused = plan_promotions(self._two_brandons(), {})
        assert mapping == {}, "must not merge two voices"
        assert any("VOICE CONFLICT" in r for r in refused), refused

    def test_the_other_brandons_quote_is_left_alone(self, tmp_path: Path) -> None:
        """The concrete harm: without the guard, quote:2 is reattributed to the guest."""
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        (meta / "ep1.gi.json").write_text(json.dumps(self._two_brandons()), encoding="utf-8")
        run(tmp_path, dry_run=False)
        doc = json.loads((meta / "ep1.gi.json").read_text(encoding="utf-8"))
        q2 = next(n for n in doc["nodes"] if n.get("id") == "quote:2")
        assert q2["properties"]["speaker_id"] == PH, q2

    def test_spoken_by_edges_count_as_voice_too(self) -> None:
        """The other of the two real forms — an edge, not a node property."""
        art = {
            "nodes": _nodes(PH, REAL),
            "edges": [
                {"from": "quote:1", "to": REAL, "type": "SPOKEN_BY"},
                {"from": "quote:2", "to": PH, "type": "SPOKEN_BY"},
            ],
        }
        mapping, refused = plan_promotions(art, {})
        assert mapping == {} and any("VOICE CONFLICT" in r for r in refused)

    def test_placeholder_voice_plus_mention_only_target_STILL_promotes(self) -> None:
        """The classic correct shape — a diarized first name plus a guest named in show notes.
        The guard must not refuse this, or it removes the value it exists to protect."""
        art = {
            "nodes": [
                {"id": PH, "properties": {"name": "Brandon"}},
                {"id": REAL, "properties": {"name": "Brandon Anderson"}},
                {"id": "quote:1", "type": "Quote", "properties": {"speaker_id": PH, "text": "a"}},
            ],
            "edges": [],
        }
        mapping, refused = plan_promotions(art, {})
        assert mapping == {PH: REAL}, (mapping, refused)

    def test_neither_speaking_still_promotes(self) -> None:
        """Both mention-only — low stakes either way, and no reason to refuse."""
        mapping, _ = plan_promotions({"nodes": _nodes(PH, REAL), "edges": []}, {})
        assert mapping == {PH: REAL}

    def test_voice_evidence_is_unioned_across_layers(self) -> None:
        """A voice asserted in KG must count when the placeholder speaks in GI."""
        gi = {
            "nodes": [
                {"id": PH, "properties": {"name": "Brandon"}},
                {"id": "quote:1", "type": "Quote", "properties": {"speaker_id": PH}},
            ],
            "edges": [],
        }
        kg = {
            "nodes": _nodes(REAL),
            "edges": [{"from": "quote:9", "to": REAL, "type": "SPOKEN_BY"}],
        }
        mapping, refused = plan_promotions(gi, kg)
        assert mapping == {} and any("VOICE CONFLICT" in r for r in refused), (mapping, refused)
