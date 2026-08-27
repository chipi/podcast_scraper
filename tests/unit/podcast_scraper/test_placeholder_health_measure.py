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
from typing import Dict, List

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
