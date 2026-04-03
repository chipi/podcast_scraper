"""E2E tests for KG CLI: validate, inspect, export on fixture artifacts.

Includes fixture-only tests and full-pipeline tests with the E2E HTTP server (like GI).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from podcast_scraper.kg import build_artifact, write_artifact
from podcast_scraper.kg.io import read_artifact
from podcast_scraper.kg.schema import validate_artifact

# Transitive typing from optional deps (e.g. google.genai) can trigger this once during
# Pydantic schema generation when the KG package pulls in a large import graph.
pytestmark = pytest.mark.filterwarnings(
    "ignore:<built-in function any>:pydantic.warnings.ArbitraryTypeWarning"
)


def _validate_kg_artifacts_strict(kg_paths: list[Path]) -> None:
    """Validate each kg.json with strict schema; raise on first failure."""
    for path in kg_paths:
        data = read_artifact(path)
        validate_artifact(data, strict=True)


def _run_kg(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run podcast_scraper kg subcommand as subprocess."""
    cmd = [sys.executable, "-m", "podcast_scraper.cli"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=timeout,
    )


@pytest.fixture
def kg_fixture_output_dir(tmp_path: Path) -> Path:
    """Output dir with metadata/*.kg.json for CLI tests."""
    (tmp_path / "metadata").mkdir(parents=True)
    art = build_artifact(
        "ep:fixture-kg-1",
        "Sample transcript about markets.",
        podcast_id="podcast:test",
        episode_title="Fixture Episode",
        topic_label="Markets overview",
        detected_hosts=["Host A"],
    )
    write_artifact(tmp_path / "metadata" / "ep1.kg.json", art, validate=True)
    return tmp_path


@pytest.mark.e2e
@pytest.mark.critical_path
def test_kg_validate_e2e(kg_fixture_output_dir: Path) -> None:
    """E2E: kg validate on fixture directory exits 0."""
    proc = _run_kg(
        ["kg", "validate", str(kg_fixture_output_dir / "metadata"), "--strict"],
        cwd=kg_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)


@pytest.mark.e2e
@pytest.mark.critical_path
def test_kg_inspect_e2e(kg_fixture_output_dir: Path) -> None:
    """E2E: kg inspect --episode-path prints JSON stats."""
    kg_path = kg_fixture_output_dir / "metadata" / "ep1.kg.json"
    proc = _run_kg(
        ["kg", "inspect", "--episode-path", str(kg_path), "--format", "json"],
        cwd=kg_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "ep:fixture-kg-1" in proc.stdout


@pytest.mark.e2e
@pytest.mark.critical_path
def test_kg_export_ndjson_e2e(kg_fixture_output_dir: Path) -> None:
    """E2E: kg export ndjson writes lines."""
    out = kg_fixture_output_dir / "out.ndjson"
    proc = _run_kg(
        [
            "kg",
            "export",
            "--output-dir",
            str(kg_fixture_output_dir),
            "--format",
            "ndjson",
            "--out",
            str(out),
        ],
        cwd=kg_fixture_output_dir,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert out.is_file()
    text = out.read_text(encoding="utf-8").strip()
    assert text
    read_artifact(
        kg_fixture_output_dir / "metadata" / "ep1.kg.json",
        validate=True,
        strict=True,
    )


@pytest.mark.e2e
@pytest.mark.critical_path
def test_kg_quality_metrics_script_e2e() -> None:
    """E2E: kg_quality_metrics.py runs on committed CI fixture (subprocess)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    fixture = project_root / "tests" / "fixtures" / "gil_kg_ci_enforce"
    script = project_root / "scripts" / "tools" / "kg_quality_metrics.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(fixture), "--json"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "artifact_paths" in proc.stdout


def _run_cli(args: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run main CLI (not kg subcommand) as subprocess."""
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
def test_kg_commands_after_pipeline_with_generate_kg(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: Run pipeline with generate_kg (stub extraction), then kg validate + inspect."""
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
        "generate_kg": True,
        "kg_extraction_source": "stub",
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    kg_files = list(output_dir.rglob("*.kg.json"))
    assert len(kg_files) >= 1, (
        f"Expected at least one .kg.json under {output_dir}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    _validate_kg_artifacts_strict(kg_files)
    kg_path = kg_files[0]
    artifact = json.loads(kg_path.read_text(encoding="utf-8"))
    episode_id = str(artifact.get("episode_id", ""))
    assert episode_id

    val_proc = _run_kg(
        ["kg", "validate", str(kg_path.parent), "--strict"],
        cwd=project_root,
        timeout=60,
    )
    assert val_proc.returncode == 0, (val_proc.stdout, val_proc.stderr)

    insp_proc = _run_kg(
        ["kg", "inspect", "--episode-path", str(kg_path), "--format", "json"],
        cwd=project_root,
        timeout=60,
    )
    assert insp_proc.returncode == 0, (insp_proc.stdout, insp_proc.stderr)
    assert episode_id in insp_proc.stdout


@pytest.mark.e2e
@pytest.mark.openai
@pytest.mark.slow
def test_kg_provider_path_openai_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: KG with kg_extraction_source=provider and OpenAI via E2E mock chat completions."""
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
        "generate_kg": True,
        "kg_extraction_source": "provider",
        "generate_summaries": True,
        "summary_provider": "openai",
        "openai_api_key": "sk-test123",
        "openai_api_base": e2e_server.urls.openai_api_base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    kg_files = list(output_dir.rglob("*.kg.json"))
    assert len(kg_files) >= 1, (proc.stdout, proc.stderr)
    _validate_kg_artifacts_strict(kg_files)
    artifact = json.loads(kg_files[0].read_text(encoding="utf-8"))
    mv = (artifact.get("extraction") or {}).get("model_version", "")
    assert str(mv).startswith("provider:"), f"expected provider model_version, got {mv!r}"
    topic_labels = [
        n.get("properties", {}).get("label")
        for n in artifact.get("nodes", [])
        if n.get("type") == "Topic"
    ]
    assert any("E2E mock topic" in str(lab) for lab in topic_labels), topic_labels


@pytest.mark.e2e
@pytest.mark.anthropic
@pytest.mark.slow
def test_kg_provider_path_anthropic_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: KG with kg_extraction_source=provider and Anthropic via E2E /v1/messages."""
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
        "generate_kg": True,
        "kg_extraction_source": "provider",
        "generate_summaries": True,
        "summary_provider": "anthropic",
        "anthropic_api_key": "sk-test-anthropic",
        "anthropic_api_base": e2e_server.urls.anthropic_api_base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    kg_files = list(output_dir.rglob("*.kg.json"))
    assert len(kg_files) >= 1, (proc.stdout, proc.stderr)
    _validate_kg_artifacts_strict(kg_files)
    artifact = json.loads(kg_files[0].read_text(encoding="utf-8"))
    mv = (artifact.get("extraction") or {}).get("model_version", "")
    assert str(mv).startswith("provider:"), f"expected provider model_version, got {mv!r}"
    topic_labels = [
        n.get("properties", {}).get("label")
        for n in artifact.get("nodes", [])
        if n.get("type") == "Topic"
    ]
    assert any("E2E mock topic" in str(lab) for lab in topic_labels), topic_labels


@pytest.mark.e2e
@pytest.mark.gemini
@pytest.mark.slow
def test_kg_provider_path_gemini_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: KG with kg_extraction_source=provider and Gemini via E2E generateContent."""
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
        "generate_kg": True,
        "kg_extraction_source": "provider",
        "generate_summaries": True,
        "summary_provider": "gemini",
        "gemini_api_key": "test-gemini-key",
        "gemini_api_base": e2e_server.urls.base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    kg_files = list(output_dir.rglob("*.kg.json"))
    assert len(kg_files) >= 1, (proc.stdout, proc.stderr)
    _validate_kg_artifacts_strict(kg_files)
    artifact = json.loads(kg_files[0].read_text(encoding="utf-8"))
    mv = (artifact.get("extraction") or {}).get("model_version", "")
    assert str(mv).startswith("provider:"), f"expected provider model_version, got {mv!r}"
    topic_labels = [
        n.get("properties", {}).get("label")
        for n in artifact.get("nodes", [])
        if n.get("type") == "Topic"
    ]
    assert any("E2E mock topic" in str(lab) for lab in topic_labels), topic_labels


@pytest.mark.e2e
@pytest.mark.grok
@pytest.mark.slow
def test_kg_provider_path_grok_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: KG with kg_extraction_source=provider and Grok (OpenAI-compatible mock)."""
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
        "generate_kg": True,
        "kg_extraction_source": "provider",
        "generate_summaries": True,
        "summary_provider": "grok",
        "grok_api_key": "test-grok-key",
        "grok_api_base": e2e_server.urls.grok_api_base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    kg_files = list(output_dir.rglob("*.kg.json"))
    assert len(kg_files) >= 1, (proc.stdout, proc.stderr)
    _validate_kg_artifacts_strict(kg_files)
    artifact = json.loads(kg_files[0].read_text(encoding="utf-8"))
    mv = (artifact.get("extraction") or {}).get("model_version", "")
    assert str(mv).startswith("provider:"), f"expected provider model_version, got {mv!r}"
    topic_labels = [
        n.get("properties", {}).get("label")
        for n in artifact.get("nodes", [])
        if n.get("type") == "Topic"
    ]
    assert any("E2E mock topic" in str(lab) for lab in topic_labels), topic_labels


@pytest.mark.e2e
@pytest.mark.deepseek
@pytest.mark.slow
def test_kg_provider_path_deepseek_e2e(
    e2e_server,
    tmp_path: Path,
) -> None:
    """E2E: KG with kg_extraction_source=provider and DeepSeek (OpenAI-compatible mock)."""
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
        "generate_kg": True,
        "kg_extraction_source": "provider",
        "generate_summaries": True,
        "summary_provider": "deepseek",
        "deepseek_api_key": "test-deepseek-key",
        "deepseek_api_base": e2e_server.urls.deepseek_api_base(),
    }
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    proc = _run_cli(["--config", str(config_path)], cwd=project_root, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    kg_files = list(output_dir.rglob("*.kg.json"))
    assert len(kg_files) >= 1, (proc.stdout, proc.stderr)
    _validate_kg_artifacts_strict(kg_files)
    artifact = json.loads(kg_files[0].read_text(encoding="utf-8"))
    mv = (artifact.get("extraction") or {}).get("model_version", "")
    assert str(mv).startswith("provider:"), f"expected provider model_version, got {mv!r}"
    topic_labels = [
        n.get("properties", {}).get("label")
        for n in artifact.get("nodes", [])
        if n.get("type") == "Topic"
    ]
    assert any("E2E mock topic" in str(lab) for lab in topic_labels), topic_labels
