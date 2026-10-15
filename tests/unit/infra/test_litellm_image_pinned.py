"""The prod LiteLLM gateway runs a pinned image, and the deploy can still move it on purpose.

2026-10-15: the compose file said `litellm-database:main-stable`. That tag moves (it already
pointed past the running v1.95.0), so any pull upgraded the gateway silently, and the deploy's
`override_image_tag` (exported as LITELLM_IMAGE_TAG) never reached the compose file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "infra" / "litellm" / "docker-compose.litellm.yml"
MOVING = {"main-stable", "main-latest", "latest", "main", "stable"}


def _image() -> str:
    return str(yaml.safe_load(COMPOSE.read_text())["services"]["litellm"]["image"])


def test_the_gateway_image_is_overridable_with_a_pinned_default() -> None:
    m = re.fullmatch(
        r"ghcr\.io/berriai/litellm-database:\$\{LITELLM_IMAGE_TAG:-([^}]+)\}", _image()
    )
    assert m, f"expected ...:${{LITELLM_IMAGE_TAG:-<version>}}, got {_image()!r}"
    default = m.group(1)
    assert default not in MOVING, f"default tag {default!r} moves"
    assert re.fullmatch(r"v\d+\.\d+\.\d+.*", default), f"default tag {default!r} is not a version"
