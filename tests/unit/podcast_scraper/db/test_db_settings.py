"""Database settings from the environment (ADR-169)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.db.settings import (
    DatabaseNotConfigured,
    DatabaseSettings,
    is_configured,
    normalize_url,
)


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


def test_docker_exec_without_the_shim_reads_the_secret_files(tmp_path: Path) -> None:
    """`docker exec` skips the image's secrets shim; the files are there, the variables are not."""
    (tmp_path / "podcast_database_url").write_text("postgresql://u:p@h/db\n", encoding="utf-8")
    (tmp_path / "podcast_database_ro_url").write_text("postgresql://r:p@h/db", encoding="utf-8")
    s = DatabaseSettings.from_env({"SECRETS_DIR": str(tmp_path)})
    assert s.rw_url == "postgresql+psycopg://u:p@h/db"
    assert s.ro_url == "postgresql+psycopg://r:p@h/db"


def test_the_variable_wins_over_the_secret_file(tmp_path: Path) -> None:
    (tmp_path / "podcast_database_url").write_text("postgresql://file@h/db", encoding="utf-8")
    s = DatabaseSettings.from_env(
        {"SECRETS_DIR": str(tmp_path), "PODCAST_DATABASE_URL": "postgresql://env@h/db"}
    )
    assert s.rw_url == "postgresql+psycopg://env@h/db"


def test_no_variable_and_no_secret_file_is_still_the_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DatabaseNotConfigured):
        DatabaseSettings.from_env({"SECRETS_DIR": str(tmp_path)})


def test_is_configured_sees_the_secret_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PODCAST_DATABASE_URL", raising=False)
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path))
    assert not is_configured()
    (tmp_path / "podcast_database_url").write_text("postgresql://u@h/db", encoding="utf-8")
    assert is_configured()
