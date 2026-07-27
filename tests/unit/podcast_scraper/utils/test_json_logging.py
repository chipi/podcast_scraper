"""Tests for the structured JSON log formatter (o11y P2: was a dead import path)."""

from __future__ import annotations

import json
import logging

import pytest

from podcast_scraper.utils import correlation as corr
from podcast_scraper.utils.json_logging import JSONFormatter


@pytest.fixture(autouse=True)
def _reset():
    corr._reset_for_tests()
    yield
    corr._reset_for_tests()


def _record(msg="hi", level=logging.INFO, **extra):
    r = logging.LogRecord("mylogger", level, "p", 1, msg, None, None)
    for k, v in extra.items():
        setattr(r, k, v)
    return r


def test_emits_valid_json_with_core_fields():
    out = JSONFormatter().format(_record("hello"))
    obj = json.loads(out)  # must parse
    assert obj["message"] == "hello"
    assert obj["level"] == "INFO"
    assert obj["logger"] == "mylogger"
    assert "ts" in obj


def test_stamps_correlation_ids_when_set():
    corr.set_run_id("run-J")
    corr.set_episode_id("ep-J")
    obj = json.loads(JSONFormatter().format(_record("x")))
    assert obj["run_id"] == "run-J"
    assert obj["episode_id"] == "ep-J"


def test_omits_ids_when_unset():
    obj = json.loads(JSONFormatter().format(_record("x")))
    assert "run_id" not in obj
    assert "episode_id" not in obj


def test_includes_exception_and_extras():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = logging.LogRecord("l", logging.ERROR, "p", 1, "failed", None, sys.exc_info())
    rec.custom_field = "abc"  # a caller extra
    obj = json.loads(JSONFormatter().format(rec))
    assert "ValueError: boom" in obj["exc"]
    assert obj["custom_field"] == "abc"


def test_non_serializable_extra_is_stringified():
    obj = json.loads(JSONFormatter().format(_record("x", weird=object())))
    assert isinstance(obj["weird"], str)
