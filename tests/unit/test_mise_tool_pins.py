"""The dev tools ``mise.toml`` pins are the versions the pipeline image runs.

rclone is pinned twice: ``PIN_RCLONE_VERSION`` in ``docker/pipeline/Dockerfile`` (the image) and
``mise.toml`` (local checkouts, for the archive tests that need the binary). A local run on another
version proves nothing about the image's.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]


def test_mise_rclone_matches_the_pipeline_image() -> None:
    dockerfile = (_ROOT / "docker" / "pipeline" / "Dockerfile").read_text(encoding="utf-8")
    pin = re.search(r"^ARG PIN_RCLONE_VERSION=v?(\S+)$", dockerfile, re.MULTILINE)
    assert pin, "docker/pipeline/Dockerfile no longer declares ARG PIN_RCLONE_VERSION"
    tools = tomllib.loads((_ROOT / "mise.toml").read_text(encoding="utf-8"))["tools"]
    assert str(tools["rclone"]) == pin.group(
        1
    ), f"mise.toml pins rclone {tools['rclone']}, the pipeline image pins {pin.group(1)}"
