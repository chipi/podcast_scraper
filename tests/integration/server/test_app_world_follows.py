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

pytestmark = [pytest.mark.integration]

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
    assert (
        len(followed_interest_episodes(ROOT, _following(tmp_path, "tc:dive-planning"), "u1")) == 2
    )


def test_an_umbrella_never_widens_mine(tmp_path: Path) -> None:
    """Two fixture topics are in all 55 episodes, and the one storyline's members together cover all
    55: following any of them would make Mine everyone's."""
    total = len(_slug_index(ROOT))
    assert len(_episodes_of("topic:lifelong-learning")) >= UMBRELLA_TOPIC_SHARE * total
    d = _following(tmp_path, "topic:lifelong-learning", "thc:managing-risk")
    assert followed_interest_episodes(ROOT, d, "u1") == set()


def test_a_theme_keeps_its_narrow_members_when_one_member_is_an_umbrella(tmp_path: Path) -> None:
    """tc:lifelong-learning contains the umbrella topic of the same name; its narrower members still
    count."""
    got = followed_interest_episodes(ROOT, _following(tmp_path, "tc:lifelong-learning"), "u1")
    assert 0 < len(got) < len(_slug_index(ROOT))


def test_following_nothing_adds_nothing(tmp_path: Path) -> None:
    assert followed_interest_episodes(ROOT, tmp_path, "u1") == set()
    assert world_episode_set(ROOT, tmp_path, "u1") == set()
