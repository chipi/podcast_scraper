"""The app-release reads cost the database one statement per few seconds, not one per request.

Prod (2026-11-06, after the kernel stores moved to Postgres): ``/auth/status`` and ``/health`` read
``shared.app_releases`` twice per call and ``pg_stat_statements`` counted ~3,200 reads in 5.4 h,
each one a pre-ping, BEGIN, SELECT and ROLLBACK. The rows change when an admin sets a release.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from podcast_scraper.server import app_release_store
from podcast_scraper.server.routes import health
from tests._database import captured_statements

pytestmark = pytest.mark.unit


def _reads(seen: list[tuple[str, bool]]) -> int:
    return sum(1 for sql, _ in seen if "app_releases" in sql)


def test_reads_hit_the_database_once_within_the_ttl(tmp_path: Path) -> None:
    app_release_store.save_released_version(tmp_path, "1.0.1")
    with captured_statements() as seen:
        for _ in range(5):
            assert app_release_store.load_released_versions(tmp_path) == {"player": "1.0.1"}
    assert _reads(seen) == 1


def test_a_write_in_this_process_shows_on_the_next_read(tmp_path: Path) -> None:
    app_release_store.save_released_version(tmp_path, "1.0.1")
    assert app_release_store.load_released_version(tmp_path) == "1.0.1"
    app_release_store.save_released_version(tmp_path, "1.0.2")
    assert app_release_store.load_released_version(tmp_path) == "1.0.2"
    app_release_store.save_released_version(tmp_path, None)
    assert app_release_store.load_released_version(tmp_path) is None


def test_the_cache_hands_out_copies(tmp_path: Path) -> None:
    app_release_store.save_released_version(tmp_path, "1.0.1")
    app_release_store.load_released_versions(tmp_path)["player"] = "9.9.9"
    assert app_release_store.load_released_version(tmp_path) == "1.0.1"


def test_the_cache_expires(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(app_release_store.time, "monotonic", lambda: clock["now"])
    app_release_store.save_released_version(tmp_path, "1.0.1")
    app_release_store.load_released_versions(tmp_path)
    with captured_statements() as seen:
        clock["now"] += app_release_store.CACHE_SECONDS - 0.1
        app_release_store.load_released_versions(tmp_path)
        assert _reads(seen) == 0
        clock["now"] += 0.2
        app_release_store.load_released_versions(tmp_path)
    assert _reads(seen) == 1


def test_a_read_that_raced_a_write_does_not_cache_the_old_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_release_store.save_released_version(tmp_path, "1.0.1")
    real_read = app_release_store.read_with_retry

    def read_then_write(fn: Callable[..., Any]) -> Any:
        stale = real_read(fn)
        app_release_store.save_released_version(tmp_path, "1.0.2")  # lands while the read returns
        return stale

    app_release_store.clear_cache()
    monkeypatch.setattr(app_release_store, "read_with_retry", read_then_write)
    assert app_release_store.load_released_versions(tmp_path) == {"player": "1.0.1"}
    monkeypatch.setattr(app_release_store, "read_with_retry", real_read)
    assert app_release_store.load_released_version(tmp_path) == "1.0.2"


def test_client_health_reads_the_releases_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_release_store, "CACHE_SECONDS", 0.0)  # no cache to hide a second read
    st = SimpleNamespace(
        app_data_dir=tmp_path, player_version="1.0.0", app_version_defaults={"news": "2.0"}
    )
    with captured_statements() as seen:
        facts = health.player_client_health(st)
    assert _reads(seen) == 1
    assert facts["player_version"] == "1.0.0"
    assert facts["app_versions"] == {"player": "1.0.0", "news": "2.0"}
