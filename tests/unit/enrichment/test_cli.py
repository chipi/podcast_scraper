"""Unit tests for ``enrichment.cli``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.cli import (
    build_arg_parser,
    build_enricher_set_from_yaml,
    parse_id_list,
    re_enable_enricher,
    run_cli,
)
from podcast_scraper.enrichment.health import HealthRegistry
from podcast_scraper.enrichment.paths import enrichment_health_path

# ---------------------------------------------------------------------------
# parse_id_list
# ---------------------------------------------------------------------------


def test_parse_id_list_none_returns_none() -> None:
    assert parse_id_list(None) is None
    assert parse_id_list("") is None


def test_parse_id_list_strips_whitespace_and_drops_empties() -> None:
    assert parse_id_list("a , b ,,c") == ["a", "b", "c"]


def test_parse_id_list_single_entry() -> None:
    assert parse_id_list("a") == ["a"]


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------


def test_arg_parser_requires_output_dir() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_accepts_all_documented_flags(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--only",
            "a,b",
            "--skip",
            "c",
            "--corpus-only",
            "--re-enable",
            "x",
            "--re-enable-reason",
            "manual",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.output_dir == tmp_path
    assert args.only == "a,b"
    assert args.skip == "c"
    assert args.corpus_only is True
    assert args.re_enable == "x"
    assert args.re_enable_reason == "manual"
    assert args.log_level == "DEBUG"


def test_arg_parser_accepts_chunk_7_profile_flags(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--profile",
            "airgapped_thin",
            "--enrichers",
            "topic_cooccurrence",
            "--no-enrichers",
            "--opt-in",
            "nli_contradiction",
        ]
    )
    assert args.profile == "airgapped_thin"
    assert args.enrichers == "topic_cooccurrence"
    assert args.no_enrichers is True
    assert args.opt_in == "nli_contradiction"


def test_arg_parser_profile_flags_default_to_none(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    assert args.profile is None
    assert args.enrichers is None
    assert args.no_enrichers is False
    assert args.opt_in is None


def test_profile_plus_only_narrows_to_subset(tmp_path: Path) -> None:
    """--profile gives the base set; --only narrows it."""
    from podcast_scraper.enrichment.profile_sets import (
        apply_cli_overrides,
        enricher_set_for_profile,
    )

    base = enricher_set_for_profile("airgapped_thin")
    out = apply_cli_overrides(base, only=["topic_cooccurrence"])
    assert out.enabled_enrichers == ["topic_cooccurrence"]


def test_no_enrichers_beats_profile_and_only(tmp_path: Path) -> None:
    """--no-enrichers wins over --profile (chunk-7 documented precedence)."""
    from podcast_scraper.enrichment.profile_sets import (
        apply_cli_overrides,
        enricher_set_for_profile,
    )

    base = enricher_set_for_profile("cloud_balanced")
    out = apply_cli_overrides(base, only=["topic_similarity"], no_enrichers=True)
    assert out.enabled_enrichers == []


# ---------------------------------------------------------------------------
# build_enricher_set_from_yaml
# ---------------------------------------------------------------------------


def test_build_enricher_set_from_yaml_missing_file_returns_empty() -> None:
    s = build_enricher_set_from_yaml(Path("/does/not/exist.yaml"))
    assert s.enabled_enrichers == []


def test_build_enricher_set_from_yaml_none_returns_empty() -> None:
    s = build_enricher_set_from_yaml(None)
    assert s.enabled_enrichers == []


def test_build_enricher_set_from_yaml_reads_enabled_list(tmp_path: Path) -> None:
    config = tmp_path / "operator.yaml"
    config.write_text(
        "enrichment:\n"
        "  enabled: true\n"
        "  enrichers:\n"
        "    topic_cooccurrence:\n"
        "      enabled: true\n"
        "    nli_contradiction:\n"
        "      enabled: false\n"
        "      opt_in: false\n",
        encoding="utf-8",
    )
    s = build_enricher_set_from_yaml(config)
    assert s.enabled_enrichers == ["topic_cooccurrence"]
    assert s.has_opt_in("nli_contradiction") is False


def test_build_enricher_set_from_yaml_per_enricher_config(tmp_path: Path) -> None:
    config = tmp_path / "operator.yaml"
    config.write_text(
        "enrichment:\n"
        "  enrichers:\n"
        "    topic_similarity:\n"
        "      enabled: true\n"
        "      threshold: 0.7\n",
        encoding="utf-8",
    )
    s = build_enricher_set_from_yaml(config)
    cfg = s.get_config("topic_similarity")
    assert cfg["threshold"] == 0.7


def test_build_enricher_set_rejects_invalid_block(tmp_path: Path) -> None:
    """Malformed enrichment block exits with SystemExit (per chunk-1 lock audit §B8)."""
    config = tmp_path / "operator.yaml"
    # ``enabled`` must be a boolean.
    config.write_text(
        "enrichment:\n  enabled: 'yes please'\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        build_enricher_set_from_yaml(config)


def test_build_enricher_set_no_enrichment_block_returns_empty(tmp_path: Path) -> None:
    config = tmp_path / "operator.yaml"
    config.write_text("other_block:\n  foo: bar\n", encoding="utf-8")
    s = build_enricher_set_from_yaml(config)
    assert s.enabled_enrichers == []


# ---------------------------------------------------------------------------
# Shape B — implicit-enabled-default semantics
# ---------------------------------------------------------------------------


def test_shape_b_block_present_without_enabled_key_is_on(tmp_path: Path) -> None:
    """Operator mental model: an enricher block in the YAML means 'configured to
    run'. ``enabled: true`` is the implicit default — operators don't need to
    write it on every block."""
    config = tmp_path / "operator.yaml"
    config.write_text(
        "enrichment:\n"
        "  enrichers:\n"
        "    temporal_velocity:\n"
        "      alpha: 0.7\n"
        "    topic_cooccurrence: {}\n"
        "    grounding_rate:\n",
        encoding="utf-8",
    )
    s = build_enricher_set_from_yaml(config)
    assert sorted(s.enabled_enrichers) == sorted(
        ["temporal_velocity", "topic_cooccurrence", "grounding_rate"]
    )
    assert s.get_config("temporal_velocity")["alpha"] == 0.7


def test_shape_b_explicit_enabled_false_opts_out(tmp_path: Path) -> None:
    """Profile may enable an enricher; operator override can disable JUST that
    one by setting ``enabled: false`` on its block. Profile's other enrichers
    untouched. Deep-merge plays nicely with this shape."""
    config = tmp_path / "operator.yaml"
    config.write_text(
        "enrichment:\n"
        "  enrichers:\n"
        "    temporal_velocity:\n"
        "      alpha: 0.7\n"
        "    topic_similarity:\n"
        "      enabled: false\n",
        encoding="utf-8",
    )
    s = build_enricher_set_from_yaml(config)
    assert "temporal_velocity" in s.enabled_enrichers
    assert "topic_similarity" not in s.enabled_enrichers
    # The config block is preserved even when disabled — the operator
    # may re-enable later without losing knobs.
    assert s.get_config("topic_similarity") == {"enabled": False}


# ---------------------------------------------------------------------------
# re_enable_enricher
# ---------------------------------------------------------------------------


def test_re_enable_clears_auto_disabled_and_persists(tmp_path: Path) -> None:
    # Seed an auto-disabled enricher.
    h = HealthRegistry(tmp_path)
    rec = h.get("x")
    rec.auto_disabled = True
    rec.auto_disabled_reason = "burned in production"
    rec.consecutive_failures = 5
    rec.cooldown_until = "2999-01-01T00:00:00Z"
    h.save()

    result = re_enable_enricher(
        corpus_root=tmp_path,
        enricher_id="x",
        reason="operator confirmed transient HF outage",
    )
    assert result["auto_disabled"] is False
    assert result["consecutive_failures"] == 0
    assert "re_enabled" in (result["auto_disabled_reason"] or "")
    assert result["cooldown_until"] is None

    # Persisted on disk.
    persisted = json.loads(enrichment_health_path(tmp_path).read_text())
    assert persisted["enrichers"]["x"]["auto_disabled"] is False


# ---------------------------------------------------------------------------
# run_cli end-to-end (no enrichers configured → clean no-op)
# ---------------------------------------------------------------------------


def test_run_cli_with_re_enable_returns_zero(tmp_path: Path) -> None:
    HealthRegistry(tmp_path).save()  # seed empty file
    parser = build_arg_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path), "--re-enable", "x"])
    exit_code = asyncio.run(run_cli(args))
    assert exit_code == 0


def test_run_cli_empty_enricher_set_returns_zero(tmp_path: Path) -> None:
    """No config + empty registry → run completes ok with exit_code 0."""
    parser = build_arg_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    exit_code = asyncio.run(run_cli(args))
    assert exit_code == 0


def test_run_cli_rejects_missing_output_dir() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--output-dir", "/does/not/exist"])
    exit_code = asyncio.run(run_cli(args))
    assert exit_code == 2
