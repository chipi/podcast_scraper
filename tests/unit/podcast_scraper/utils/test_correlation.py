"""Run/episode correlation ids — the join key across o11y signals (#1053)."""

from __future__ import annotations

import logging
import threading

import pytest

from podcast_scraper.utils import correlation as corr


@pytest.fixture(autouse=True)
def _reset():
    corr._reset_for_tests()
    yield
    corr._reset_for_tests()


# ── resolve_run_id ──────────────────────────────────────────────────────────


def test_resolve_keeps_a_real_id() -> None:
    assert corr.resolve_run_id("my-run-7") == "my-run-7"
    assert corr.resolve_run_id("  spaced  ") == "spaced"


@pytest.mark.parametrize("raw", [None, "", "   ", "auto", "AUTO", "Auto"])
def test_resolve_synthesises_for_auto_or_empty(raw) -> None:
    rid = corr.resolve_run_id(raw)
    assert rid.startswith("run-")
    assert len(rid) > len("run-")  # carries a timestamp


def test_resolve_is_stable_within_a_call_but_set_once_in_practice() -> None:
    # two resolves of "auto" differ (timestamped) — which is exactly why the
    # pipeline resolves ONCE at run start and shares the value.
    assert corr.resolve_run_id("auto") != corr.resolve_run_id("auto")


# ── run id (process global) ─────────────────────────────────────────────────


def test_run_id_unset_is_none() -> None:
    assert corr.get_run_id() is None
    assert corr.correlation_fields() == {}


def test_set_get_run_id() -> None:
    corr.set_run_id("run-42")
    assert corr.get_run_id() == "run-42"
    assert corr.correlation_fields() == {"run_id": "run-42"}


def test_set_run_id_blank_clears() -> None:
    corr.set_run_id("run-42")
    corr.set_run_id("   ")
    assert corr.get_run_id() is None


def test_run_id_is_visible_across_threads() -> None:
    # run_id is a process global precisely so the summarisation worker pool sees it.
    corr.set_run_id("run-threaded")
    seen: list = []
    t = threading.Thread(target=lambda: seen.append(corr.get_run_id()))
    t.start()
    t.join()
    assert seen == ["run-threaded"]


# ── episode id (context-local) ──────────────────────────────────────────────


def test_set_get_episode_id() -> None:
    corr.set_episode_id("ep:1")
    assert corr.get_episode_id() == "ep:1"


def test_correlation_fields_includes_both() -> None:
    corr.set_run_id("run-9")
    corr.set_episode_id("ep:9")
    assert corr.correlation_fields() == {"run_id": "run-9", "episode_id": "ep:9"}


# ── CorrelationFormatter ────────────────────────────────────────────────────


def _record(msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("n", logging.INFO, "p", 1, msg, None, None)


def test_formatter_stamps_run_id_when_set() -> None:
    corr.set_run_id("run-LOG")
    fmt = corr.CorrelationFormatter("%(levelname)s [run=%(run_id)s]: %(message)s")
    out = fmt.format(_record("boom"))
    assert out == "INFO [run=run-LOG]: boom"  # Loki can `|= \"run=run-LOG\"`


def test_formatter_defaults_to_dash_outside_a_run() -> None:
    # never KeyErrors on %(run_id)s even when no run is active.
    fmt = corr.CorrelationFormatter("[run=%(run_id)s ep=%(episode_id)s] %(message)s")
    assert fmt.format(_record("x")) == "[run=- ep=-] x"
