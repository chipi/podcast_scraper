"""Following an interest widens "Mine" (ADR-162 amendment, ADR-163 §6; operator 2026-10-11).

Over the COMMITTED fixture corpus (``tests/fixtures/app-validation-corpus/v3``, read only): a
followed topic or person brings its episodes into the listener's world, a followed theme or
storyline its member topics' episodes — and an umbrella never does, so the interests picker can
seed Mine without turning it into Everyone. Deterministic, disk-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_discover_view import interest_episode_index
from podcast_scraper.server.app_slugs import _slug_index
from podcast_scraper.server.app_user_corpus import (
    followed_interest_episodes,
    UMBRELLA_TOPIC_SHARE,
    world_episode_set,
)

# The rails, reels and Mine read the search sidecar (`interest_episode_index`), which is gitignored:
# this builds it once per session, rebuilds it when the corpus changes, and skips without a model.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("app_validation_search_index")]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def _following(tmp_path: Path, *tokens: str) -> Path:
    for t in tokens:
        app_user_state.add_interest(tmp_path, "u1", t)
    return tmp_path


def _episodes_of(token: str) -> set[str]:
    paths = interest_episode_index(ROOT)[token]
    return {s for s, row in _slug_index(ROOT).items() if row.metadata_relative_path in paths}


def test_a_followed_topic_or_person_brings_in_its_episodes(tmp_path: Path) -> None:
    d = _following(tmp_path, "topic:systems-thinking", "person:dr-elena-fischer")
    want = _episodes_of("topic:systems-thinking") | _episodes_of("person:dr-elena-fischer")
    assert want and world_episode_set(ROOT, d, "u1") == want


def test_a_followed_theme_brings_in_its_member_topics_episodes(tmp_path: Path) -> None:
    import json

    clusters = json.loads((ROOT / "search" / "topic_clusters.json").read_text("utf-8"))["clusters"]
    members = next(c for c in clusters if c["graph_compound_parent_id"] == "tc:dive-planning")
    want = set().union(*(_episodes_of(m["topic_id"]) for m in members["members"]))
    got = followed_interest_episodes(ROOT, _following(tmp_path, "tc:dive-planning"), "u1")
    assert len(members["members"]) > 1 and got == want


#: The natural corpus (2026-10-12) has no topic on 90% of episodes — the stamped umbrellas that
#: were are gone, as real data has none (prod: 1.0% at most). The guard still needs proving, so
#: these lower the share for this module: at 0.55, `risk management` (31 of 55 episodes) is an
#: umbrella, `thc:second-order-effects` (34 across its members, none alone) is a storyline that
#: covers too much, and `tc:risk-management` is a theme with an umbrella member and narrow ones.
_TEST_SHARE = 0.55


@pytest.fixture
def umbrella_share(monkeypatch: pytest.MonkeyPatch) -> float:
    from podcast_scraper.server import app_user_corpus

    monkeypatch.setattr(app_user_corpus, "UMBRELLA_TOPIC_SHARE", _TEST_SHARE)
    return _TEST_SHARE


def test_an_umbrella_never_widens_mine(tmp_path: Path, umbrella_share: float) -> None:
    """A topic on most episodes, and a storyline whose members together cover most: following
    either would make Mine everyone's."""
    total = len(_slug_index(ROOT))
    assert len(_episodes_of("topic:risk-management")) >= umbrella_share * total
    d = _following(tmp_path, "topic:risk-management", "thc:second-order-effects")
    assert followed_interest_episodes(ROOT, d, "u1") == set()


def test_a_theme_keeps_its_narrow_members_when_one_member_is_an_umbrella(
    tmp_path: Path, umbrella_share: float
) -> None:
    """tc:risk-management contains the umbrella topic of the same name; its narrower members still
    count."""
    got = followed_interest_episodes(ROOT, _following(tmp_path, "tc:risk-management"), "u1")
    assert 0 < len(got) < len(_episodes_of("topic:risk-management"))


def test_the_packaged_share_still_leaves_real_subjects_followable(tmp_path: Path) -> None:
    """At the packaged 0.9 nothing on the natural corpus is an umbrella: following risk management
    widens Mine by its episodes, as following a subject should."""
    assert UMBRELLA_TOPIC_SHARE == 0.9
    got = followed_interest_episodes(ROOT, _following(tmp_path, "topic:risk-management"), "u1")
    assert got == _episodes_of("topic:risk-management")


def test_following_nothing_adds_nothing(tmp_path: Path) -> None:
    assert followed_interest_episodes(ROOT, tmp_path, "u1") == set()
    assert world_episode_set(ROOT, tmp_path, "u1") == set()
