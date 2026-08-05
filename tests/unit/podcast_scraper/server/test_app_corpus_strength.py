"""Unit tests for corpus strength (RFC-114 Phase 2, #1470)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_corpus_strength as st

pytestmark = pytest.mark.unit


def _s(
    *,
    heard_fraction: float = 0.0,
    captures: int = 0,
    favorited: bool = False,
    relistens: int = 0,
) -> float:
    return st.strength(
        heard_fraction=heard_fraction,
        captures=captures,
        favorited=favorited,
        relistens=relistens,
    )


def test_bounds_and_empty() -> None:
    assert _s() == 0.0
    assert _s(heard_fraction=1.0, captures=99, favorited=True, relistens=99) == 1.0


def test_clamps_out_of_range() -> None:
    assert _s(heard_fraction=5.0) == round(st.DEFAULT.heard, 4)  # heard_fraction clamped to 1
    assert _s(heard_fraction=-1.0) == 0.0


def test_monotonic_in_each_signal() -> None:
    base = _s(heard_fraction=0.3)
    assert _s(heard_fraction=0.3, captures=1) > base
    assert _s(heard_fraction=0.3, favorited=True) > base
    assert _s(heard_fraction=0.3, relistens=1) > base
    assert _s(heard_fraction=0.6) > base


def test_rfc_ordering_engaged_outranks_bare_play() -> None:
    # RFC-114 test: a highlighted + re-heard episode outranks a bare 30% play.
    bare = _s(heard_fraction=0.3)
    engaged = _s(heard_fraction=1.0, captures=3, relistens=2)
    assert engaged > bare


def test_capture_saturation() -> None:
    # Beyond the cap, more captures add nothing.
    at_cap = _s(captures=st.DEFAULT.captures_cap)
    over_cap = _s(captures=st.DEFAULT.captures_cap + 10)
    assert at_cap == over_cap


def test_episode_strengths_over_signals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_scraper.server import app_user_state

    uid = "u_0123456789abcdef01234567"
    monkeypatch.setattr(
        st.app_user_corpus, "experienced_episode_set", lambda r, d, u: {"ep-a", "ep-b"}
    )
    monkeypatch.setattr(
        st.app_user_corpus, "slug_durations", lambda r: {"ep-a": 1000.0, "ep-b": 1000.0}
    )
    monkeypatch.setattr(st.app_user_corpus, "saved_episode_set", lambda d, u: set())
    # ep-a: fully heard + 2 highlights; ep-b: 30% heard, nothing else
    monkeypatch.setattr(
        app_user_state,
        "list_playback",
        lambda d, u: [
            {"slug": "ep-a", "position_seconds": 1000.0},
            {"slug": "ep-b", "position_seconds": 300.0},
        ],
    )
    monkeypatch.setattr(
        app_user_state,
        "get_highlights",
        lambda d, u: [{"episode_slug": "ep-a"}, {"episode_slug": "ep-a"}],
    )
    monkeypatch.setattr(app_user_state, "get_notes", lambda d, u, target=None: [])
    monkeypatch.setattr(app_user_state, "list_listen_events", lambda d, u: [])

    scores = st.episode_strengths(tmp_path, tmp_path, uid)
    assert scores["ep-a"] > scores["ep-b"]  # engaged episode ranks higher
