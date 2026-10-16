"""Contract: APP_OPERATOR_API_KEY is wired end-to-end for prod (incremental-add prereq).

The operator write-API (POST /api/jobs, PUT /api/feeds, DELETE /api/corpus/*) authenticates
headless automation via X-Operator-Key == APP_OPERATOR_API_KEY (app_operator_guard.py). Wiring it
into prod spans four files; a future edit dropping any one leg leaves prod with an EMPTY key and no
way for automation to authenticate (silent — reads stay open). These text-contract assertions keep
all legs in lockstep. The shim's uppercase-export behaviour itself is covered by test_secrets_shim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]


def test_prod_compose_api_env_exposes_the_key() -> None:
    text = (REPO / "compose" / "docker-compose.prod.yml").read_text(encoding="utf-8")
    assert "APP_OPERATOR_API_KEY: ${APP_OPERATOR_API_KEY:-}" in text


def test_secrets_overlay_mounts_the_key_for_api_only() -> None:
    text = (REPO / "compose" / "docker-compose.secrets.yml").read_text(encoding="utf-8")
    # top-level secret definition + the tmpfs source path
    assert "app_operator_api_key:" in text
    assert "file: /dev/shm/podcast-secrets/app_operator_api_key" in text
    # api service references it; pipeline-llm (no HTTP) must NOT.
    api_block = text.split("pipeline-llm:")[0]
    assert "- app_operator_api_key" in api_block
    # pipeline-llm SERVICE secrets = between "pipeline-llm:" and the top-level "secrets:" block.
    pipeline_service_block = text.split("pipeline-llm:")[1].split("\nsecrets:")[0]
    assert "app_operator_api_key" not in pipeline_service_block


def test_no_literal_key_value_committed() -> None:
    """The wiring must reference the secret, never embed a value (defense against a paste slip)."""
    # deploy-prod.yml used to be checked here too. It is no longer in this repository, and
    # the deploy path that consumes these files asserts the same thing on its own side.
    for rel in (
        "compose/docker-compose.prod.yml",
        "compose/docker-compose.secrets.yml",
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        # a 64-hex openssl-rand key literal would show up as a long hex run assigned to the var
        assert "APP_OPERATOR_API_KEY=" not in text or "$PROD_APP_OPERATOR_API_KEY" in text
