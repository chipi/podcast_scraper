"""Integration test: real CLI subprocess invocation honours --profile.

Unit tests (``tests/unit/podcast_scraper/test_cli_profile_routing.py``)
cover the argparse → _build_config → Config chain in-process. This
integration test spawns the real CLI as a subprocess and inspects the
startup log output to confirm the profile values actually reach the
orchestrator — catching failures that only manifest when ``python -m
podcast_scraper.cli`` runs with its own module resolution, SDK imports,
and log configuration.

Uses ``--max-episodes 1 --no-transcribe-missing --no-generate-metadata``
pointed at a deliberately invalid RSS path so the pipeline reaches the
fetch stage (proving the full chain wired) and then exits quickly.
No LLM / transcription cost.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
# CI and local dev may not use ``.venv`` at this path; use the active interpreter.
_PY = sys.executable


def _fake_keys_env() -> dict:
    env = dict(os.environ)
    for name in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROK_API_KEY",
    ):
        env.setdefault(name, "test-" + name.lower().replace("_", "-") + "-dummy-key")
    return env


def _run_cli(profile: str, output_dir: Path) -> subprocess.CompletedProcess:
    """Spawn the CLI with the profile + an invalid RSS URL. The CLI should
    load the profile, enter orchestration, then exit on RSS fetch failure.
    Returns the completed process for inspection."""
    cmd = [
        _PY,
        "-m",
        "podcast_scraper.cli",
        "--profile",
        profile,
        "--output-dir",
        str(output_dir),
        "--max-episodes",
        "1",
        "--no-transcribe-missing",
        "--no-auto-speakers",
        # Unreachable but URL-validator-safe; scraper logs config, then fails to fetch.
        "http://127.0.0.1:1/profile_test.xml",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=_fake_keys_env())


@pytest.mark.parametrize(
    "profile,markers",
    [
        # Each profile's two-line startup log has a distinctive combination of
        # summary=on:<provider> and metadata=<format>. If the profile loaded
        # correctly the markers all appear.
        (
            "cloud_balanced",
            ["summary=on:gemini", "metadata=on:json"],
        ),
        (
            "cloud_quality",
            ["summary=on:anthropic", "metadata=on:json"],
        ),
        (
            "local",
            ["summary=on:ollama", "metadata=on:json"],
        ),
        (
            "airgapped",
            ["summary=on:summllama", "metadata=on:json"],
        ),
        (
            "dev",
            ["summary=on:transformers", "metadata=on:json"],
        ),
    ],
)
def test_profile_reaches_orchestrator_with_expected_config(
    tmp_path: Path, profile: str, markers: list[str]
) -> None:
    result = _run_cli(profile, tmp_path)
    combined = (result.stdout or "") + (result.stderr or "")
    for m in markers:
        assert m in combined, (
            f"{profile}: marker {m!r} missing from CLI output.\n"
            f"exit={result.returncode}\n"
            f"tail: {combined[-800:]}"
        )


def test_profile_loads_before_rss_fetch_failure(tmp_path: Path) -> None:
    """Smoke: profile-load happens early enough that orchestration logs
    appear even though RSS fetch ultimately fails."""
    result = _run_cli("cloud_balanced", tmp_path)
    combined = (result.stdout or "") + (result.stderr or "")
    # "config: rss=" is the first startup log line; proves args parsed.
    assert "config: rss=" in combined, combined[-800:]
    # Profile-specific provider name should appear in the config summary.
    assert "gemini" in combined.lower(), combined[-800:]
