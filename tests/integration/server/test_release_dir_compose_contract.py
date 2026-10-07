#!/usr/bin/env python3
"""The released app version is set in the operator viewer and served by the player api (#2296).

Both public apis must mount the SAME host dir and point ``APP_RELEASE_DIR`` at it; otherwise the
Admin field writes a copy no phone ever reads — which is what shipped first. Only that dir is
shared: each stack's ``/app/appdata`` keeps its own host path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

pytestmark = [
    pytest.mark.integration,
    pytest.mark.critical_path,
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker CLI not on PATH; compose-config contract test requires docker",
    ),
]

COMPOSE = Path(__file__).resolve().parents[3] / "compose"


def _api(name: str) -> Dict[str, Any]:
    env = {**os.environ, "PODCAST_CORPUS_VOLUME": "compose_corpus_data"}
    cmd = ["docker", "compose", "-f", str(COMPOSE / name), "config", "--format", "yaml"]
    proc = subprocess.run(  # noqa: S603 - hardcoded cmd
        cmd, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise AssertionError(f"`docker compose config` exited {proc.returncode}\n{proc.stderr}")
    svc: Dict[str, Any] = yaml.safe_load(proc.stdout)["services"]["api"]
    return svc


def _source(svc: Dict[str, Any], target: str) -> str | None:
    for vol in svc.get("volumes") or []:
        if isinstance(vol, dict) and vol.get("target") == target:
            return str(vol.get("source"))
    return None


def test_both_apis_share_one_release_dir_and_nothing_else() -> None:
    player = _api("docker-compose.player-public.yml")
    operator = _api("docker-compose.operator-public.yml")
    for svc in (player, operator):
        env = svc.get("environment") or {}
        assert env.get("APP_RELEASE_DIR") == "/app/release"
        # The Admin view reads "served now" from the operator api: it needs the player's default.
        assert "APP_PLAYER_VERSION" in env
    shared = _source(player, "/app/release")
    assert shared and shared == _source(operator, "/app/release")
    assert _source(player, "/app/appdata") != _source(operator, "/app/appdata")
    assert shared not in (_source(player, "/app/appdata"), _source(operator, "/app/appdata"))
