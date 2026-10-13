"""RFC-128's replay (scripts/eval/replay_storylines.py) on the fixture corpus."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "tests" / "fixtures" / "app-validation-corpus" / "v3"
_SPEC = importlib.util.spec_from_file_location(
    "replay_storylines_under_test", ROOT / "scripts" / "eval" / "replay_storylines.py"
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)


@pytest.fixture(scope="module")
def replayed() -> dict:
    result: dict = _mod.replay(CORPUS, min_episodes=5)
    return result


def test_the_old_rules_last_month_is_a_fresh_production_run(replayed: dict) -> None:
    """Faithfulness: with every episode in, the old rule is the enricher run from scratch."""
    import tempfile

    from podcast_scraper.enrichment.enrichers import topic_theme_clusters as ttc
    from podcast_scraper.enrichment.paths import discover_episode_bundles

    ctx = type("Ctx", (), {"run_id": "t", "enricher_id": "topic_theme_clusters"})()
    with tempfile.TemporaryDirectory() as d:
        fresh = ttc._compute(None, Path(d), discover_episode_bundles(CORPUS), {}, ctx)
    month, last = replayed["old"][-1]
    assert set(last) == {c["graph_compound_parent_id"] for c in fresh["clusters"]}


def test_the_new_rule_never_drops_a_storyline_it_published(replayed: dict) -> None:
    seen: set[str] = set()
    for _month, snap in replayed["new"]:
        assert seen <= set(snap), "a published storyline disappeared"
        seen |= set(snap)
    summary = _mod.summarise(replayed["new"])
    kept, total = summary["follows_kept"].split("/")
    assert kept == total


def test_the_new_rule_keeps_one_storyline_per_topic(replayed: dict) -> None:
    for month, snap in replayed["new"]:
        assert _mod.summarise([(month, snap)])["topics_in_two_storylines"] == 0, month
