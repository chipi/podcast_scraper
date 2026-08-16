"""task-#14: topic-cluster merge threshold precedence (override > config > 0.75 default).

The 0.35 value is a viewer-validation *small-fixture* override; it must NOT leak into a prod
rebuild. The rebuild endpoints resolve the threshold through ``resolve_topic_cluster_threshold``,
so this pins the fall-through so a regression can't silently ship 0.35 (or None) to prod.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request

from podcast_scraper.server.routes.index_rebuild import resolve_topic_cluster_threshold

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def _request(config: object) -> Request:
    # resolve_topic_cluster_threshold only touches request.app.state.config — a namespace stands in.
    return cast(Request, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=config))))


def test_explicit_override_wins_over_config() -> None:
    req = _request(SimpleNamespace(topic_cluster_threshold=0.5))
    assert resolve_topic_cluster_threshold(req, 0.9) == 0.9


def test_config_value_used_when_no_override() -> None:
    req = _request(SimpleNamespace(topic_cluster_threshold=0.6))
    assert resolve_topic_cluster_threshold(req, None) == 0.6


def test_defaults_to_profile_075_when_config_absent() -> None:
    req = _request(SimpleNamespace(topic_cluster_threshold=None))
    assert resolve_topic_cluster_threshold(req, None) == 0.75


def test_defaults_to_075_when_no_config_object() -> None:
    req = _request(None)
    assert resolve_topic_cluster_threshold(req, None) == 0.75


def test_override_zero_is_honored_not_treated_as_falsy() -> None:
    """0.0 is an explicit (if degenerate) override, not 'unset' — must not fall through to 0.75."""
    req = _request(SimpleNamespace(topic_cluster_threshold=0.6))
    assert resolve_topic_cluster_threshold(req, 0.0) == 0.0
