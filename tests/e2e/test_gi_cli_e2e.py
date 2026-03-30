"""E2E tests for GIL CLI: all gi commands, with and without full pipeline.

Two styles of E2E:

1. **Pre-built artifact (no mock server):** Fixture dir with a .gi.json written in-process
   (build_artifact + write_artifact). Runs gi validate, export, inspect, show-insight, explore,
   query via subprocess. Simulates: "user already has pipeline output and runs gi commands."
   Does NOT use the E2E HTTP server; does NOT generate GIs from pipeline content.

2. **Full pipeline + mock server:** Uses e2e_server (RSS + transcripts from mock server).
   Runs the main pipeline with generate_gi=true via config -> produces .gi.json from
   transcript content -> then runs gi inspect, show-insight, explore, and query on that output.
   Simulates real user flow and generates GIs from content served by the mock server.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from podcast_scraper.gi import build_artifact, write_artifact
from podcast_scraper.gi.io import read_artifact
from podcast_scraper.gi.schema import validate_artifact


def _validate_gi_artifacts_strict(gi_paths: list[Path]) -> None:
    """Validate each gi.json with strict schema; raise on first failure."""
    for path in gi_paths:
        data = read_artifact(path)
        validate_artifact(data, strict=True)


def _run_gi(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run podcast_scraper gi subcommand as subprocess."""
    cmd = [sys.executable, "-m", "podcast_scraper.cli"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=timeout,
    )


@pytest.fixture
def gi_fixture_output_dir(tmp_path: Path) -> Path:
    """Create output dir with metadata/ep1.gi.json and optional transcript."""
    (tmp_path / "metadata").mkdir(parents=True)
    (tmp_path / "transcripts").mkdir(parents=True)
    artifact = build_artifact(
        "ep:fixture-1",
        "Evidence for insight. Regulation will lag.",
        prompt_version="v1",
    )
    write_artifact(tmp_path / "metadata" / "ep1.gi.json", artifact, validate=True)
    (tmp_path / "transcripts" / "ep1.txt").write_text(
        "Evidence for insight. Regulation will lag.", encoding="utf-8"
    )
    return tmp_path


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_inspect_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi inspect on fixture artifact exits 0 and prints stats."""
    gi_path = gi_fixture_output_dir / "metadata" / "ep1.gi.json"
    proc = _run_gi(
        ["gi", "inspect", "--episode-path", str(gi_path), "--format", "json"],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "ep:fixture-1" in proc.stdout
    assert "insight_count" in proc.stdout or "insights" in proc.stdout


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_show_insight_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi show-insight by id on fixture dir exits 0."""
    gi_path = gi_fixture_output_dir / "metadata" / "ep1.gi.json"
    art = json.loads(gi_path.read_text(encoding="utf-8"))
    insight_id = next(n["id"] for n in art["nodes"] if n.get("type") == "Insight")
    proc = _run_gi(
        [
            "gi",
            "show-insight",
            "--id",
            insight_id,
            "--output-dir",
            str(gi_fixture_output_dir),
            "--format",
            "json",
        ],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert insight_id in proc.stdout or "Summary insight" in proc.stdout


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_explore_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi explore on fixture dir exits 0 and returns insights."""
    proc = _run_gi(
        [
            "gi",
            "explore",
            "--output-dir",
            str(gi_fixture_output_dir),
            "--format",
            "json",
        ],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "episodes_searched" in proc.stdout or "insight_count" in proc.stdout
    # Topic filter (substring in insight text; stub text contains "insight")
    proc2 = _run_gi(
        [
            "gi",
            "explore",
            "--topic",
            "insight",
            "--output-dir",
            str(gi_fixture_output_dir),
            "--format",
            "json",
        ],
        cwd=gi_fixture_output_dir,
    )
    assert proc2.returncode == 0, (proc2.stdout, proc2.stderr)


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_query_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi query with UC4 topic pattern exits 0 and prints answer envelope JSON."""
    proc = _run_gi(
        [
            "gi",
            "query",
            "--output-dir",
            str(gi_fixture_output_dir),
            "--question",
            "What insights about stub?",
        ],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert '"question"' in proc.stdout and '"answer"' in proc.stdout


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_query_unmatched_pattern_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi query with no UC4 pattern match exits 2."""
    proc = _run_gi(
        [
            "gi",
            "query",
            "--output-dir",
            str(gi_fixture_output_dir),
            "--question",
            "Random phrase with no uc4 pattern xyz789.",
        ],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 2, (proc.stdout, proc.stderr)


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_validate_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi validate --strict on metadata dir exits 0."""
    proc = _run_gi(
        ["gi", "validate", "--strict", str(gi_fixture_output_dir / "metadata")],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "OK" in proc.stdout or "passed validation" in proc.stdout


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_export_ndjson_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi export --format ndjson writes lines with episode_id."""
    out_file = gi_fixture_output_dir / "gi_export.ndjson"
    proc = _run_gi(
        [
            "gi",
            "export",
            "--output-dir",
            str(gi_fixture_output_dir),
            "--format",
            "ndjson",
            "--out",
            str(out_file),
        ],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    text = out_file.read_text(encoding="utf-8")
    assert "ep:fixture-1" in text
    assert "_artifact_path" in text


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_export_merged_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi export --format merged writes gi_corpus_bundle."""
    out_file = gi_fixture_output_dir / "gi_bundle.json"
    proc = _run_gi(
        [
            "gi",
            "export",
            "--output-dir",
            str(gi_fixture_output_dir),
            "--format",
            "merged",
            "--out",
            str(out_file),
        ],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    text = out_file.read_text(encoding="utf-8")
    assert "gi_corpus_bundle" in text
    assert "artifacts" in text


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_query_top_topics_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi query topic-leaderboard pattern exits 0 with topics in JSON."""
    proc = _run_gi(
        [
            "gi",
            "query",
            "--output-dir",
            str(gi_fixture_output_dir),
            "--question",
            "Which topics have the most insights?",
        ],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert '"topics"' in proc.stdout


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gi_query_compound_speaker_topic_e2e(gi_fixture_output_dir: Path) -> None:
    """E2E: gi query 'What did X say about Y?' exits 0."""
    proc = _run_gi(
        [
            "gi",
            "query",
            "--output-dir",
            str(gi_fixture_output_dir),
            "--question",
            "What did nobody say about insight?",
        ],
        cwd=gi_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert '"answer"' in proc.stdout


@pytest.mark.e2e
@pytest.mark.critical_path
def test_gil_quality_metrics_script_e2e() -> None:
    """E2E: gil_quality_metrics.py runs on committed CI fixture (subprocess)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    fixture = project_root / "tests" / "fixtures" / "gil_kg_ci_enforce"
    script = project_root / "scripts" / "tools" / "gil_quality_metrics.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(fixture), "--json"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "artifact_paths" in proc.stdout


# --- Full pipeline + mock server: generate GIs from E2E server content ---


def _run_cli(args: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run main CLI (not gi subcommand) as subprocess."""
    cmd = [sys.executable, "-m", "podcast_scraper.cli"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=timeout,
    )


@pytest.mark.e2e
@pytest.mark.slow
def test_gi_commands_after_pipeline_with_generate_gi(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: Run pipeline with generate_gi using mock server content, then all gi commands.

    Uses E2E HTTP server (RSS + transcript). Runs full pipeline with generate_gi=true
    so .gi.json is produced from transcript content. Then runs gi inspect, show-insight,
    explore, and query on that output. Simulates what a user would do.
    """
    import json

    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    rss_url = e2e_server.urls.feed("podcast1_with_transcript")

    config = {
        "rss": rss_url,
        "output_dir": str(output_dir),
        "max_episodes": 1,
        "generate_metadata": True,
        "generate_gi": True,
        "quote_extraction_provider": "transformers",
        "entailment_provider": "transformers",
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    # Run pipeline (download transcript from mock server, write metadata + gi.json)
    proc = _run_cli(["--config", str(config_path)], cwd=project_root)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    # Find produced .gi.json
    gi_files = list(output_dir.rglob("*.gi.json"))
    assert len(gi_files) >= 1, (
        f"Expected at least one .gi.json under {output_dir}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    _validate_gi_artifacts_strict(gi_files)
    gi_path = gi_files[0]
    artifact = json.loads(gi_path.read_text(encoding="utf-8"))
    episode_id = artifact.get("episode_id", "")
    insight_id = None
    for node in artifact.get("nodes", []):
        if node.get("type") == "Insight":
            insight_id = node.get("id")
            break
    assert insight_id, "Artifact should have at least one Insight node"
    # Provider evidence path: artifact has schema and edges (SUPPORTED_BY when grounded)
    assert "schema_version" in artifact and "nodes" in artifact
    assert isinstance(artifact.get("edges"), list), "Artifact should have edges list"

    # Run gi commands (same as a user would)
    inspect_proc = _run_gi(
        ["gi", "inspect", "--episode-path", str(gi_path), "--format", "json"],
        cwd=project_root,
    )
    assert inspect_proc.returncode == 0, (inspect_proc.stdout, inspect_proc.stderr)
    assert episode_id in inspect_proc.stdout or "insight" in inspect_proc.stdout.lower()

    show_proc = _run_gi(
        [
            "gi",
            "show-insight",
            "--id",
            insight_id,
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
        cwd=project_root,
    )
    assert show_proc.returncode == 0, (show_proc.stdout, show_proc.stderr)
    assert insight_id in show_proc.stdout

    explore_proc = _run_gi(
        ["gi", "explore", "--output-dir", str(output_dir), "--format", "json"],
        cwd=project_root,
    )
    assert explore_proc.returncode == 0, (explore_proc.stdout, explore_proc.stderr)
    assert "episodes_searched" in explore_proc.stdout or "insight_count" in explore_proc.stdout

    query_proc = _run_gi(
        [
            "gi",
            "query",
            "--output-dir",
            str(output_dir),
            "--question",
            "What insights about insight?",
        ],
        cwd=project_root,
    )
    assert query_proc.returncode == 0, (query_proc.stdout, query_proc.stderr)
    assert '"answer"' in query_proc.stdout


@pytest.mark.e2e
@pytest.mark.openai
@pytest.mark.slow
def test_gi_provider_path_openai_evidence_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: GIL with quote_extraction_provider=openai, entailment_provider=openai via mock server.

    Uses the same E2E mock server pattern as other OpenAI E2E tests: openai_api_base
    points to the server, which now handles /v1/chat/completions for GIL extract_quotes
    and score_entailment (returns JSON quote_text and 0.85). Asserts gi.json is produced
    with nodes and edges.
    """
    import json

    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    rss_url = e2e_server.urls.feed("podcast1_with_transcript")

    config = {
        "rss": rss_url,
        "output_dir": str(output_dir),
        "max_episodes": 1,
        "generate_metadata": True,
        "generate_gi": True,
        "generate_summaries": True,
        "summary_provider": "openai",
        "quote_extraction_provider": "openai",
        "entailment_provider": "openai",
        "openai_api_key": "sk-test123",
        "openai_api_base": e2e_server.urls.openai_api_base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    gi_files = list(output_dir.rglob("*.gi.json"))
    assert len(gi_files) >= 1, (
        f"Expected at least one .gi.json under {output_dir}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    _validate_gi_artifacts_strict(gi_files)
    artifact = json.loads(gi_files[0].read_text(encoding="utf-8"))
    assert "schema_version" in artifact and "nodes" in artifact
    assert isinstance(artifact.get("edges"), list), "Artifact should have edges list"
    insight_nodes = [n for n in artifact.get("nodes", []) if n.get("type") == "Insight"]
    assert len(insight_nodes) >= 1, "Artifact should have at least one Insight node"


@pytest.mark.e2e
@pytest.mark.anthropic
@pytest.mark.slow
def test_gi_provider_path_anthropic_evidence_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: GIL with quote_extraction_provider=anthropic, entailment_provider=anthropic.

    Uses E2E mock server: anthropic_api_base points to the server, which handles
    /v1/messages for GIL extract_quotes and score_entailment. Asserts gi.json is produced.
    """
    import json

    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    rss_url = e2e_server.urls.feed("podcast1_with_transcript")

    config = {
        "rss": rss_url,
        "output_dir": str(output_dir),
        "max_episodes": 1,
        "generate_metadata": True,
        "generate_gi": True,
        "generate_summaries": True,
        "summary_provider": "anthropic",
        "quote_extraction_provider": "anthropic",
        "entailment_provider": "anthropic",
        "anthropic_api_key": "sk-test-anthropic",
        "anthropic_api_base": e2e_server.urls.anthropic_api_base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    gi_files = list(output_dir.rglob("*.gi.json"))
    assert len(gi_files) >= 1, (
        f"Expected at least one .gi.json under {output_dir}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    _validate_gi_artifacts_strict(gi_files)
    artifact = json.loads(gi_files[0].read_text(encoding="utf-8"))
    assert "schema_version" in artifact and "nodes" in artifact
    assert isinstance(artifact.get("edges"), list), "Artifact should have edges list"
    insight_nodes = [n for n in artifact.get("nodes", []) if n.get("type") == "Insight"]
    assert len(insight_nodes) >= 1, "Artifact should have at least one Insight node"


@pytest.mark.e2e
@pytest.mark.gemini
@pytest.mark.slow
def test_gi_provider_path_gemini_evidence_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: GIL with quote_extraction_provider=gemini, entailment_provider=gemini.

    Uses E2E mock server only: gemini_api_base points to the server (provider
    passes it via http_options.base_url). No real API key needed; gi.json produced.
    """
    import json

    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    rss_url = e2e_server.urls.feed("podcast1_with_transcript")

    config = {
        "rss": rss_url,
        "output_dir": str(output_dir),
        "max_episodes": 1,
        "generate_metadata": True,
        "generate_gi": True,
        "generate_summaries": True,
        "summary_provider": "gemini",
        "quote_extraction_provider": "gemini",
        "entailment_provider": "gemini",
        "gemini_api_key": "test-gemini-key",
        "gemini_api_base": e2e_server.urls.base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    gi_files = list(output_dir.rglob("*.gi.json"))
    assert len(gi_files) >= 1, (
        f"Expected at least one .gi.json under {output_dir}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    _validate_gi_artifacts_strict(gi_files)
    artifact = json.loads(gi_files[0].read_text(encoding="utf-8"))
    assert "schema_version" in artifact and "nodes" in artifact
    assert isinstance(artifact.get("edges"), list), "Artifact should have edges list"
    insight_nodes = [n for n in artifact.get("nodes", []) if n.get("type") == "Insight"]
    assert len(insight_nodes) >= 1, "Artifact should have at least one Insight node"


@pytest.mark.e2e
@pytest.mark.mistral
@pytest.mark.slow
def test_gi_provider_path_mistral_evidence_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: GIL with quote_extraction_provider=mistral, entailment_provider=mistral.

    Uses E2E mock server: mistral_api_base is server root so the Mistral SDK
    builds .../v1/chat/completions (avoids double /v1). No real API key needed.
    """
    import json

    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    rss_url = e2e_server.urls.feed("podcast1_with_transcript")

    config = {
        "rss": rss_url,
        "output_dir": str(output_dir),
        "max_episodes": 1,
        "generate_metadata": True,
        "generate_gi": True,
        "generate_summaries": True,
        "summary_provider": "mistral",
        "quote_extraction_provider": "mistral",
        "entailment_provider": "mistral",
        "mistral_api_key": "test-mistral-key",
        "mistral_api_base": e2e_server.urls.base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    gi_files = list(output_dir.rglob("*.gi.json"))
    assert len(gi_files) >= 1, (
        f"Expected at least one .gi.json under {output_dir}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    _validate_gi_artifacts_strict(gi_files)
    artifact = json.loads(gi_files[0].read_text(encoding="utf-8"))
    assert "schema_version" in artifact and "nodes" in artifact
    assert isinstance(artifact.get("edges"), list), "Artifact should have edges list"
    insight_nodes = [n for n in artifact.get("nodes", []) if n.get("type") == "Insight"]
    assert len(insight_nodes) >= 1, "Artifact should have at least one Insight node"


@pytest.mark.e2e
@pytest.mark.grok
@pytest.mark.slow
def test_gi_provider_path_grok_evidence_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: GIL with quote_extraction_provider=grok, entailment_provider=grok.

    Uses E2E mock server: grok_api_base points to /v1/chat/completions which
    handles GIL extract_quotes and score_entailment. Asserts gi.json is produced.
    """
    import json

    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    rss_url = e2e_server.urls.feed("podcast1_with_transcript")

    config = {
        "rss": rss_url,
        "output_dir": str(output_dir),
        "max_episodes": 1,
        "generate_metadata": True,
        "generate_gi": True,
        "generate_summaries": True,
        "summary_provider": "grok",
        "quote_extraction_provider": "grok",
        "entailment_provider": "grok",
        "grok_api_key": "test-grok-key",
        "grok_api_base": e2e_server.urls.grok_api_base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    gi_files = list(output_dir.rglob("*.gi.json"))
    assert len(gi_files) >= 1, (
        f"Expected at least one .gi.json under {output_dir}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    _validate_gi_artifacts_strict(gi_files)
    artifact = json.loads(gi_files[0].read_text(encoding="utf-8"))
    assert "schema_version" in artifact and "nodes" in artifact
    assert isinstance(artifact.get("edges"), list), "Artifact should have edges list"
    insight_nodes = [n for n in artifact.get("nodes", []) if n.get("type") == "Insight"]
    assert len(insight_nodes) >= 1, "Artifact should have at least one Insight node"


@pytest.mark.e2e
@pytest.mark.deepseek
@pytest.mark.slow
def test_gi_provider_path_deepseek_evidence_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: GIL with quote_extraction_provider=deepseek, entailment_provider=deepseek.

    Uses E2E mock server: deepseek_api_base points to /v1/chat/completions which
    handles GIL extract_quotes and score_entailment. Asserts gi.json is produced.
    """
    import json

    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    rss_url = e2e_server.urls.feed("podcast1_with_transcript")

    config = {
        "rss": rss_url,
        "output_dir": str(output_dir),
        "max_episodes": 1,
        "generate_metadata": True,
        "generate_gi": True,
        "generate_summaries": True,
        "summary_provider": "deepseek",
        "quote_extraction_provider": "deepseek",
        "entailment_provider": "deepseek",
        "deepseek_api_key": "test-deepseek-key",
        "deepseek_api_base": e2e_server.urls.deepseek_api_base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    gi_files = list(output_dir.rglob("*.gi.json"))
    assert len(gi_files) >= 1, (
        f"Expected at least one .gi.json under {output_dir}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    _validate_gi_artifacts_strict(gi_files)
    artifact = json.loads(gi_files[0].read_text(encoding="utf-8"))
    assert "schema_version" in artifact and "nodes" in artifact
    assert isinstance(artifact.get("edges"), list), "Artifact should have edges list"
    insight_nodes = [n for n in artifact.get("nodes", []) if n.get("type") == "Insight"]
    assert len(insight_nodes) >= 1, "Artifact should have at least one Insight node"


@pytest.mark.e2e
@pytest.mark.ollama
@pytest.mark.slow
def test_gi_provider_path_ollama_evidence_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: GIL with quote_extraction_provider=ollama, entailment_provider=ollama.

    Uses E2E mock server: ollama_api_base points to /v1/chat/completions which
    handles GIL extract_quotes and score_entailment. Asserts gi.json is produced.
    """
    import json

    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    rss_url = e2e_server.urls.feed("podcast1_with_transcript")

    config = {
        "rss": rss_url,
        "output_dir": str(output_dir),
        "max_episodes": 1,
        "generate_metadata": True,
        "generate_gi": True,
        "generate_summaries": True,
        "summary_provider": "ollama",
        "quote_extraction_provider": "ollama",
        "entailment_provider": "ollama",
        "ollama_api_base": e2e_server.urls.ollama_api_base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    gi_files = list(output_dir.rglob("*.gi.json"))
    assert len(gi_files) >= 1, (
        f"Expected at least one .gi.json under {output_dir}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    _validate_gi_artifacts_strict(gi_files)
    artifact = json.loads(gi_files[0].read_text(encoding="utf-8"))
    assert "schema_version" in artifact and "nodes" in artifact
    assert isinstance(artifact.get("edges"), list), "Artifact should have edges list"
    insight_nodes = [n for n in artifact.get("nodes", []) if n.get("type") == "Insight"]
    assert len(insight_nodes) >= 1, "Artifact should have at least one Insight node"


@pytest.mark.e2e
def test_gi_explore_without_output_dir_exits_2() -> None:
    """E2E: gi explore without required --output-dir exits with 2 (invalid args)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    proc = _run_gi(["gi", "explore", "--topic", "x"], cwd=project_root)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)


@pytest.mark.e2e
def test_gi_show_insight_without_id_exits_2() -> None:
    """E2E: gi show-insight without required --id exits with 2 (invalid args)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    proc = _run_gi(["gi", "show-insight", "--output-dir", "/tmp"], cwd=project_root)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
