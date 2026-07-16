"""Behavioral tests for docker/secrets-shim.sh (ADR-115).

The shim is the ENTRYPOINT of the api + pipeline images: it exports every file in
``$SECRETS_DIR`` as an UPPERCASE env var, then ``exec``s the real entrypoint. A
regression here silently boots containers without their secrets, so lock the
three contract points: exports-then-exec, tolerant of an absent/empty dir, and
honors ``WRAPPED_ENTRYPOINT`` + passes args through.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SHIM = REPO / "docker" / "secrets-shim.sh"

# A fake "real entrypoint" that dumps the env var the shim should have exported,
# plus its own argv — so the test can assert both the export and the exec-through.
FAKE_ENTRYPOINT = (
    "#!/bin/sh\n" 'echo "OPENAI_API_KEY=[${OPENAI_API_KEY:-<unset>}]"\n' 'echo "ARGV=[$*]"\n'
)


def _run(secrets_dir: str | None, entrypoint: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["WRAPPED_ENTRYPOINT"] = str(entrypoint)
    if secrets_dir is not None:
        env["SECRETS_DIR"] = secrets_dir
    return subprocess.run(
        ["/bin/sh", str(SHIM), *argv],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture()
def entrypoint(tmp_path: Path) -> Path:
    ep = tmp_path / "entrypoint.sh"
    ep.write_text(FAKE_ENTRYPOINT)
    ep.chmod(0o755)
    return ep


def test_exports_secret_files_as_uppercase_env_then_execs(tmp_path: Path, entrypoint: Path) -> None:
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "openai_api_key").write_text("sk-test-123")
    proc = _run(str(secrets), entrypoint, "serve", "--port", "8000")
    assert proc.returncode == 0, proc.stderr
    assert "OPENAI_API_KEY=[sk-test-123]" in proc.stdout, proc.stdout
    # exec-through preserves argv
    assert "ARGV=[serve --port 8000]" in proc.stdout, proc.stdout


def test_absent_secrets_dir_is_tolerated(tmp_path: Path, entrypoint: Path) -> None:
    proc = _run(str(tmp_path / "does-not-exist"), entrypoint)
    assert proc.returncode == 0, proc.stderr
    assert "OPENAI_API_KEY=[<unset>]" in proc.stdout
    assert "ARGV=[]" in proc.stdout


def test_empty_secrets_dir_is_tolerated(tmp_path: Path, entrypoint: Path) -> None:
    secrets = tmp_path / "empty"
    secrets.mkdir()
    proc = _run(str(secrets), entrypoint)
    assert proc.returncode == 0, proc.stderr
    assert "OPENAI_API_KEY=[<unset>]" in proc.stdout
