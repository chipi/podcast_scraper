"""`measure_placeholder_health` must fire on the damage it exists to find (#1685/#1801).

The committed fixture carries ZERO placeholders, so running the audit against it renders this
section not at all — a clean pass that proves nothing. That is the failure mode this file exists
to close: a measure that silently reports nothing is indistinguishable from a measure that found
nothing, and only one of those is good news.

The three things it must count, each built here as the smallest corpus that exhibits it:

  * CONTAMINATED — the same `unresolved-…` id in two episodes. A placeholder carries its own
    episode, so sharing one is proof that an episode imported another episode's scope. This is
    what the un-fixed `resolve_candidates` wrote into production between 2026-08-21 and the fix.
  * BLOCKED HEAL — a placeholder in an episode that also holds its real person. Under the old
    rule the placeholder was its own rival candidate, so the rule declined to guess and scoped
    instead of healing.
  * RECURRENCE — single-token names appearing in 2+ episodes vs exactly once. This is the number
    #1801 is decided on: a recurring name is a person whose mentions are being lost, a one-off is
    an incidental reference worth nothing to resolve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.capability_audit import measure_placeholder_health

pytestmark = [pytest.mark.unit]


class _Row:
    """The catalog-row surface the measure's call chain touches.

    `has_kg` / `has_gi` are load-bearing, not decoration: `_episode_features` short-circuits on
    them (`app_discover_view.py:170`), so a row that omits them makes the measure read an empty
    person set and report a cheerful zero — the exact silent-pass this file exists to prevent.
    """

    def __init__(self, relpath: str, feed: str) -> None:
        self.metadata_relative_path = relpath
        self.feed_id = feed
        self.feed_title = feed
        self.has_kg = True
        self.has_gi = False
        self.kg_relative_path = relpath.replace(".metadata.json", ".kg.json")
        self.gi_relative_path = relpath.replace(".metadata.json", ".gi.json")


def _corpus(tmp_path: Path, episodes: Dict[str, List[str]]) -> tuple[Path, List[_Row]]:
    """`{episode_name: [person ids]}` -> a corpus on disk plus its rows.

    Persons are written into the KG layer, which `_episode_person_ids` unions with GI.
    """
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, persons in episodes.items():
        (meta / f"{name}.kg.json").write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "nodes": [
                        {"id": p, "kind": "person", "name": p.split(":")[-1]} for p in persons
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        (meta / f"{name}.metadata.json").write_text(
            json.dumps({"episode_id": name, "title": name, "summary": {"bullets": ["x"]}}),
            encoding="utf-8",
        )
        rows.append(_Row(f"metadata/{name}.metadata.json", f"feed-{name}"))
    return tmp_path, rows


def _corpus_layers(tmp_path: Path, episodes: Dict[str, Dict[str, Any]]):
    """`{episode: {"kg": [...], "gi": [...], "kg_edges": [(src,tgt)]}}` -> corpus + rows.

    Separate from `_corpus` because the location measure needs the layers to DIFFER; a helper
    that writes the same ids to both cannot exercise it.
    """
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, spec in episodes.items():
        for suffix, key in ((".kg.json", "kg"), (".gi.json", "gi")):
            ids = spec.get(key) or []
            edges = [
                {"source": a, "target": b, "type": "MENTIONS"}
                for a, b in (spec.get(f"{key}_edges") or [])
            ]
            (meta / f"{name}{suffix}").write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "nodes": [
                            {"id": i, "kind": "person", "name": i.split(":")[-1]} for i in ids
                        ],
                        "edges": edges,
                    }
                ),
                encoding="utf-8",
            )
        (meta / f"{name}.metadata.json").write_text(
            json.dumps({"episode_id": name, "title": name}), encoding="utf-8"
        )
        row = _Row(f"metadata/{name}.metadata.json", f"feed-{name}")
        row.has_gi = True
        rows.append(row)
    return tmp_path, rows


class TestTheLocationActuallyDiscriminates:
    """The coexistence count says "impossible happened"; the location says HOW.

    Without this the extra field is decoration — it would render `?` or a constant and nobody
    would notice until it was relied on to diagnose production.
    """

    def test_a_half_written_pair_names_the_two_layers(self, tmp_path: Path) -> None:
        """KG scoped, GI still bare — the "only one write landed" shape."""
        root, rows = _corpus_layers(
            tmp_path,
            {"ep1": {"kg": ["person:unresolved-dario-ep1"], "gi": ["person:dario"]}},
        )
        r = measure_placeholder_health(root, rows)
        ex = r["coexist_examples"][0]
        assert ex["bare_at"] == "gi:nodes", ex
        assert ex["placeholder_at"] == "kg:nodes", ex

    def test_an_edge_only_id_names_the_EDGE_ENDPOINT_specifically(self, tmp_path: Path) -> None:
        """Not just "elsewhere" — the exact structure.

        `elsewhere` lumped edge endpoints, edge `speaker_id` and quote-node
        `properties.speaker_id` under one label, and I then read that label as though it meant
        the first. That is how "every one of the 23 is an edge endpoint" became a claim the
        measurement could not support. Each needs a different fix, so each gets its own name.
        """
        root, rows = _corpus_layers(
            tmp_path,
            {
                "ep1": {
                    "kg": ["person:unresolved-dario-ep1"],
                    "gi": [],
                    "gi_edges": [("person:dario", "person:x")],
                }
            },
        )
        r = measure_placeholder_health(root, rows)
        ex = r["coexist_examples"][0]
        assert ex["bare_at"] == "gi:edge_endpoint", ex
        assert ex["placeholder_at"] == "kg:nodes", ex

    def test_both_layers_is_reported_as_both(self, tmp_path: Path) -> None:
        root, rows = _corpus_layers(
            tmp_path,
            {
                "ep1": {
                    "kg": ["person:dario", "person:unresolved-dario-ep1"],
                    "gi": ["person:dario", "person:unresolved-dario-ep1"],
                }
            },
        )
        ex = measure_placeholder_health(root, rows)["coexist_examples"][0]
        assert ex["bare_at"] == "gi:nodes, kg:nodes", ex


class TestItCountsCrossEpisodeContamination:
    def test_a_placeholder_shared_by_two_episodes_is_flagged(self, tmp_path: Path) -> None:
        root, rows = _corpus(
            tmp_path,
            {
                "ep1": ["person:unresolved-dario-ep1"],
                # ep2 wrongly carries ep1's placeholder — the cross-episode heal.
                "ep2": ["person:unresolved-dario-ep1"],
            },
        )
        r = measure_placeholder_health(root, rows)
        assert r["contaminated_ids"] == 1, r
        assert r["contaminated_examples"][0]["placeholder"] == "unresolved-dario-ep1"
        assert r["contaminated_examples"][0]["episodes"] == 2

    def test_correctly_scoped_placeholders_are_not_flagged(self, tmp_path: Path) -> None:
        """Each episode owning its own placeholder is the CORRECT state and must read as clean."""
        root, rows = _corpus(
            tmp_path,
            {
                "ep1": ["person:unresolved-dario-ep1"],
                "ep2": ["person:unresolved-dario-ep2"],
            },
        )
        r = measure_placeholder_health(root, rows)
        assert r["contaminated_ids"] == 0, r
        assert r["placeholders_total"] == 2


class TestItCountsBlockedHeals:
    def test_a_placeholder_beside_its_real_person_is_the_repair_list(self, tmp_path: Path) -> None:
        root, rows = _corpus(
            tmp_path,
            {"ep1": ["person:unresolved-dario-ep1", "person:dario-amodei"]},
        )
        r = measure_placeholder_health(root, rows)
        assert r["blocked_heals"] == 1, r
        ex = r["blocked_examples"][0]
        assert ex["placeholder"] == "unresolved-dario-ep1"
        assert ex["should_be"] == "dario-amodei"

    def test_a_placeholders_own_bare_form_is_not_a_heal_target(self, tmp_path: Path) -> None:
        """The self-match that shipped in the first cut, caught against production.

        `unresolved-dario-ep1` and `person:dario` in one episode made the measure report
        `unresolved-dario-… should be dario` — "healing" a placeholder into the very bare name it
        stands for. A resolution target must be a real FULL name; excluding placeholders from the
        target pool is not enough, because a bare token trivially contains itself. Same self-match
        shape as the pipeline bug this whole measure exists to find, one layer up.
        """
        root, rows = _corpus(tmp_path, {"ep1": ["person:dario", "person:unresolved-dario-ep1"]})
        r = measure_placeholder_health(root, rows)
        assert r["blocked_heals"] == 0, r["blocked_examples"]

    def test_that_case_is_reported_as_coexistence_instead(self, tmp_path: Path) -> None:
        """It is not nothing — it means the scoping did not stick, which the backfill must know."""
        root, rows = _corpus(tmp_path, {"ep1": ["person:dario", "person:unresolved-dario-ep1"]})
        r = measure_placeholder_health(root, rows)
        assert r["bare_coexists_with_placeholder"] == 1, r
        assert r["coexist_examples"][0]["bare"] == "dario"
        assert r["coexist_examples"][0]["placeholder"] == "unresolved-dario-ep1"

    def test_a_real_full_name_still_wins_when_the_bare_form_is_also_present(
        self, tmp_path: Path
    ) -> None:
        """Excluding bare ids must not suppress a GENUINE blocked heal sitting beside them."""
        root, rows = _corpus(
            tmp_path,
            {"ep1": ["person:dario", "person:unresolved-dario-ep1", "person:dario-amodei"]},
        )
        r = measure_placeholder_health(root, rows)
        assert r["blocked_heals"] == 1, r
        assert r["blocked_examples"][0]["should_be"] == "dario-amodei"
        assert r["bare_coexists_with_placeholder"] == 1, "both facts are true and both reported"

    def test_two_candidates_is_not_a_blocked_heal(self, tmp_path: Path) -> None:
        """Genuinely ambiguous stays ambiguous — the repair list must not invent a verdict."""
        root, rows = _corpus(
            tmp_path,
            {"ep1": ["person:unresolved-trump-ep1", "person:donald-trump", "person:eric-trump"]},
        )
        assert measure_placeholder_health(root, rows)["blocked_heals"] == 0

    def test_an_orphan_placeholder_is_not_a_blocked_heal(self, tmp_path: Path) -> None:
        root, rows = _corpus(tmp_path, {"ep1": ["person:unresolved-jensen-ep1"]})
        assert measure_placeholder_health(root, rows)["blocked_heals"] == 0


class TestItSplitsRecurringFromOneOff:
    """The #1801 decision number: only recurring names represent a person worth resolving."""

    def test_recurring_and_once_only_are_counted_separately(self, tmp_path: Path) -> None:
        root, rows = _corpus(
            tmp_path,
            {
                "ep1": ["person:unresolved-jensen-ep1", "person:unresolved-nandini-ep1"],
                "ep2": ["person:unresolved-jensen-ep2"],
                "ep3": ["person:unresolved-jensen-ep3"],
            },
        )
        r = measure_placeholder_health(root, rows)
        assert r["names_recurring"] == 1, r  # jensen, in 3 episodes
        assert r["names_once_only"] == 1, r  # nandini, in 1
        assert r["recurring_examples"][0] == {"name": "jensen", "episodes": 3}

    def test_a_bare_id_that_was_never_scoped_still_counts_toward_recurrence(
        self, tmp_path: Path
    ) -> None:
        """Pre-migration corpora hold BARE ids, not placeholders. The enricher decision covers
        both populations, so a corpus that has not been migrated yet must still produce a
        meaningful recurrence answer — otherwise the number only works after the migration that
        it is supposed to inform."""
        root, rows = _corpus(
            tmp_path, {"ep1": ["person:jensen"], "ep2": ["person:jensen"], "ep3": ["person:sam"]}
        )
        r = measure_placeholder_health(root, rows)
        assert r["names_recurring"] == 1, r
        assert r["names_once_only"] == 1, r


class TestTheSectionIsNotSilentlyEmpty:
    def test_a_corpus_with_no_placeholders_reports_zero_not_absence(self, tmp_path: Path) -> None:
        """Distinguishes "measured, found none" from "did not measure"."""
        root, rows = _corpus(tmp_path, {"ep1": ["person:dario-amodei"]})
        r = measure_placeholder_health(root, rows)
        assert r["placeholders_total"] == 0
        assert r["contaminated_ids"] == 0
        assert r["blocked_heals"] == 0


class TestTheConvergenceCheckIsHonest:
    """`converges` answers: would re-running the migration land on the placeholder ALREADY there?

    If the audit derives the episode id differently from m0007, the check is worse than useless —
    it would report "converges" while the re-run mints a SECOND placeholder, giving one person
    three ids. So the derivation is pinned against the migration's own, not merely written to
    look similar.
    """

    def test_it_derives_the_same_episode_id_as_m0007(self, tmp_path: Path) -> None:
        from podcast_scraper.capability_audit import _m0007_episode_id
        from podcast_scraper.upgrade.migrations.m0007_scope_bare_person_names import (
            _episode_id_of,
        )

        root, rows = _corpus_layers(
            tmp_path, {"ep1": {"kg": ["person:jensen"], "gi": ["person:jensen"]}}
        )
        # _corpus_layers writes no episode: node, so both must agree on the empty fallback —
        # which scoped_person_id turns into "unknown", a greppable marker rather than a silent
        # mis-scoping.
        assert _m0007_episode_id(root, rows[0]) == _episode_id_of({}, {}) == ""

    def test_it_reads_the_episode_node_when_present(self, tmp_path: Path) -> None:
        from podcast_scraper.capability_audit import _m0007_episode_id
        from podcast_scraper.upgrade.migrations.m0007_scope_bare_person_names import (
            _episode_id_of,
        )

        meta = tmp_path / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "3.1",
            "nodes": [
                {"id": "episode:substack-post-42", "kind": "episode"},
                {"id": "person:jensen", "kind": "person"},
            ],
            "edges": [],
        }
        for suffix in (".gi.json", ".kg.json"):
            (meta / f"ep1{suffix}").write_text(json.dumps(payload), encoding="utf-8")
        (meta / "ep1.metadata.json").write_text(json.dumps({"episode_id": "ep1"}), encoding="utf-8")
        row = _Row("metadata/ep1.metadata.json", "feed-1")
        row.has_gi = True

        assert _m0007_episode_id(tmp_path, row) == "substack-post-42"
        assert _episode_id_of(payload) == "substack-post-42"

    def test_a_matching_placeholder_reports_converges_true(self, tmp_path: Path) -> None:
        """End to end: bare id + the placeholder m0007 would produce -> converges."""
        root, rows = _corpus_layers(
            tmp_path,
            {"ep1": {"kg": ["person:jensen", "person:unresolved-jensen-unknown"], "gi": []}},
        )
        r = measure_placeholder_health(root, rows)
        assert r["coexist_examples"][0]["converges"] is True, r["coexist_examples"]
