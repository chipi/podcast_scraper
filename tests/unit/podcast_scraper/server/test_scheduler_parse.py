"""Unit tests for ``scheduler.parse_scheduled_jobs`` (#708).

Pure YAML parsing + Pydantic validation — no APScheduler dependency, so
these run under the ``[dev]``-only test extra.
"""

from __future__ import annotations

import pytest

from podcast_scraper.server.scheduler import (
    parse_scheduled_jobs,
    ScheduledJobConfig,
    ScheduledJobsParseError,
)


class TestParseScheduledJobs:
    def test_empty_yaml_returns_empty_list(self) -> None:
        assert parse_scheduled_jobs("") == []
        assert parse_scheduled_jobs("   \n   ") == []

    def test_yaml_without_key_returns_empty_list(self) -> None:
        assert parse_scheduled_jobs("max_episodes: 1\nworkers: 4\n") == []

    def test_yaml_with_null_key_returns_empty_list(self) -> None:
        assert parse_scheduled_jobs("scheduled_jobs:\n") == []

    def test_yaml_root_must_be_mapping(self) -> None:
        # A list at root is invalid for operator YAML; treat as no schedule.
        assert parse_scheduled_jobs("- a\n- b\n") == []

    def test_invalid_yaml_raises(self) -> None:
        with pytest.raises(ScheduledJobsParseError, match="invalid YAML"):
            parse_scheduled_jobs("scheduled_jobs:\n  -\n   - [unclosed")

    def test_non_list_value_raises(self) -> None:
        with pytest.raises(ScheduledJobsParseError, match="must be a list"):
            parse_scheduled_jobs("scheduled_jobs: not-a-list\n")

    def test_non_mapping_entry_raises(self) -> None:
        yaml_text = "scheduled_jobs:\n  - just-a-string\n"
        with pytest.raises(ScheduledJobsParseError, match="must be a mapping"):
            parse_scheduled_jobs(yaml_text)

    def test_single_valid_job(self) -> None:
        yaml_text = (
            "scheduled_jobs:\n"
            "  - name: morning-sweep\n"
            "    cron: '0 4 * * *'\n"
            "    enabled: true\n"
        )
        out = parse_scheduled_jobs(yaml_text)
        assert len(out) == 1
        assert out[0].name == "morning-sweep"
        assert out[0].cron == "0 4 * * *"
        assert out[0].enabled is True

    def test_enabled_defaults_to_true(self) -> None:
        yaml_text = "scheduled_jobs:\n  - name: x\n    cron: '0 * * * *'\n"
        out = parse_scheduled_jobs(yaml_text)
        assert out[0].enabled is True

    def test_disabled_job_is_loaded(self) -> None:
        yaml_text = (
            "scheduled_jobs:\n" "  - name: x\n" "    cron: '0 * * * *'\n" "    enabled: false\n"
        )
        out = parse_scheduled_jobs(yaml_text)
        assert out[0].enabled is False

    def test_multiple_jobs(self) -> None:
        yaml_text = (
            "scheduled_jobs:\n"
            "  - name: morning\n"
            "    cron: '0 4 * * *'\n"
            "  - name: evening\n"
            "    cron: '0 20 * * *'\n"
            "    enabled: false\n"
        )
        out = parse_scheduled_jobs(yaml_text)
        assert [j.name for j in out] == ["morning", "evening"]
        assert [j.enabled for j in out] == [True, False]

    def test_duplicate_names_raise(self) -> None:
        yaml_text = (
            "scheduled_jobs:\n"
            "  - name: same\n"
            "    cron: '0 4 * * *'\n"
            "  - name: same\n"
            "    cron: '0 5 * * *'\n"
        )
        with pytest.raises(ScheduledJobsParseError, match="duplicate name"):
            parse_scheduled_jobs(yaml_text)

    def test_missing_required_field_raises(self) -> None:
        yaml_text = "scheduled_jobs:\n  - name: x\n"
        with pytest.raises(ScheduledJobsParseError):
            parse_scheduled_jobs(yaml_text)

    def test_empty_name_raises(self) -> None:
        yaml_text = "scheduled_jobs:\n  - name: ''\n    cron: '0 4 * * *'\n"
        with pytest.raises(ScheduledJobsParseError):
            parse_scheduled_jobs(yaml_text)

    def test_empty_cron_raises(self) -> None:
        yaml_text = "scheduled_jobs:\n  - name: x\n    cron: ''\n"
        with pytest.raises(ScheduledJobsParseError):
            parse_scheduled_jobs(yaml_text)

    def test_invalid_name_chars_rejected(self) -> None:
        yaml_text = "scheduled_jobs:\n" "  - name: 'has spaces'\n" "    cron: '0 * * * *'\n"
        with pytest.raises(ScheduledJobsParseError, match="letters, digits"):
            parse_scheduled_jobs(yaml_text)

    def test_name_with_underscores_and_hyphens_ok(self) -> None:
        yaml_text = "scheduled_jobs:\n" "  - name: morning_sweep-v2\n" "    cron: '0 4 * * *'\n"
        out = parse_scheduled_jobs(yaml_text)
        assert out[0].name == "morning_sweep-v2"

    def test_name_too_long_rejected(self) -> None:
        long_name = "a" * 65
        yaml_text = f"scheduled_jobs:\n  - name: {long_name}\n    cron: '0 * * * *'\n"
        with pytest.raises(ScheduledJobsParseError):
            parse_scheduled_jobs(yaml_text)

    def test_cron_field_strips_whitespace(self) -> None:
        cfg = ScheduledJobConfig(name="x", cron="  0 4 * * *  ")
        assert cfg.cron == "0 4 * * *"
