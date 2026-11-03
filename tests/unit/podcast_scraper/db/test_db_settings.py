"""Database settings from the environment (ADR-169)."""

from __future__ import annotations

import pytest

from podcast_scraper.db.settings import DatabaseNotConfigured, DatabaseSettings, normalize_url


def test_no_url_is_a_clear_error_naming_the_test_database() -> None:
    with pytest.raises(DatabaseNotConfigured) as err:
        DatabaseSettings.from_env({})
    message = str(err.value)
    assert "PODCAST_DATABASE_URL" in message
    assert "make test-db-up" in message


def test_ro_defaults_to_rw_and_plane_to_player() -> None:
    s = DatabaseSettings.from_env({"PODCAST_DATABASE_URL": "postgresql://u:p@h/db"})
    assert s.rw_url == s.ro_url == "postgresql+psycopg://u:p@h/db"
    assert s.plane == "player"


def test_ro_url_and_operator_plane() -> None:
    s = DatabaseSettings.from_env(
        {
            "PODCAST_DATABASE_URL": "postgresql://u:p@h/db",
            "PODCAST_DATABASE_RO_URL": "postgres://r:p@replica/db",
            "PODCAST_DB_PLANE": "operator",
        }
    )
    assert s.ro_url == "postgresql+psycopg://r:p@replica/db"
    assert s.plane == "operator"


def test_an_unknown_plane_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown plane"):
        DatabaseSettings.from_env(
            {"PODCAST_DATABASE_URL": "postgresql://u:p@h/db", "PODCAST_DB_PLANE": "corpus"}
        )


def test_a_driver_qualified_url_is_left_alone() -> None:
    assert normalize_url("postgresql+psycopg://u@h/db") == "postgresql+psycopg://u@h/db"
