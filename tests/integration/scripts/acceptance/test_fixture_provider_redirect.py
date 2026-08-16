"""Guard the acceptance-fixture provider-base redirect (the #1527 litellm-class bug).

``modify_config_for_fixtures`` must point EVERY provider base URL at the E2E mock so a
``--use-fixtures`` run never dials real infrastructure. Two mechanisms exist:

* **env-fallback** providers (openai/gemini/anthropic/... ) read ``<X>_API_BASE`` from the
  environment, so the harness overrides ``os.environ``.
* **config-only** providers (litellm/qwen/vllm) have NO env-var fallback in
  :mod:`podcast_scraper.config` — ``OpenAICompatibleProvider`` reads ``cfg.<ns>_api_base``
  directly. The harness MUST rewrite the CONFIG field; an env override is silently ignored.

The original bug: #1527 switched ``cloud_balanced`` to ``litellm`` (config-only) but the harness
only overrode env vars, so the fixture run dialed the real ``homelab:4001`` and CI went red on
main-push (never on the PR). These tests make that class of failure a fast, on-PR check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from podcast_scraper.config import Config

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

# scripts/acceptance is not a package; add it to the path (same shim as the unit tests).
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_ACCEPTANCE = _PROJECT_ROOT / "scripts" / "acceptance"
if str(_SCRIPTS_ACCEPTANCE) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ACCEPTANCE))

from run_acceptance_tests import modify_config_for_fixtures  # noqa: E402

# Provider ``*_api_base`` fields that resolve from an env var (``_load_string_env_var`` /
# ``_load_*_api_base_from_env`` in config.py). For these the harness overrides os.environ.
_ENV_FALLBACK_API_BASES = frozenset(
    {
        "openai_api_base",
        "gemini_api_base",
        "anthropic_api_base",
        "deepgram_api_base",
        "mistral_api_base",
        "deepseek_api_base",
        "grok_api_base",
        "groq_api_base",
        "ollama_api_base",
    }
)

# Provider ``*_api_base`` fields with NO env fallback — the harness MUST rewrite the config field.
# Keep in lockstep with the config_dict[...] overrides in modify_config_for_fixtures.
_CONFIG_ONLY_API_BASES = frozenset({"litellm_api_base", "qwen_api_base", "vllm_api_base"})


class _FakeURLs:
    """The subset of ``E2EServerURLs`` that ``modify_config_for_fixtures`` calls, all pointing at
    a fixed 127.0.0.1 mock so the assertion is "did the harness redirect off the real host?"."""

    _BASE = "http://127.0.0.1:59999"

    def feed(self, name: str) -> str:
        return f"{self._BASE}/feeds/{name}/feed.xml"

    def _v1(self) -> str:
        return f"{self._BASE}/v1"

    # env-fallback providers (also set as env vars by the harness; harmless to expose here)
    openai_api_base = gemini_api_base = mistral_api_base = _v1
    grok_api_base = deepseek_api_base = ollama_api_base = _v1
    groq_api_base = _v1

    def anthropic_api_base(self) -> str:
        return self._BASE

    def deepgram_api_base(self) -> str:
        return self._BASE

    # config-only providers — the ones this test exists to guard
    litellm_api_base = qwen_api_base = vllm_api_base = _v1


class _FakeE2EServer:
    def __init__(self) -> None:
        self.urls = _FakeURLs()
        self.base_url = _FakeURLs._BASE


def test_all_api_base_fields_are_classified() -> None:
    """Every ``*_api_base`` Config field is either env-fallback or config-only. A NEW one that is
    unclassified fails here — forcing the author to decide whether the acceptance harness needs a
    config-field redirect (the litellm bug) or an env override."""
    all_api_bases = {name for name in Config.model_fields if name.endswith("_api_base")}
    classified = _ENV_FALLBACK_API_BASES | _CONFIG_ONLY_API_BASES
    unclassified = all_api_bases - classified
    assert not unclassified, (
        f"Unclassified *_api_base Config fields: {sorted(unclassified)}. Add each to "
        "_ENV_FALLBACK_API_BASES (env override) or _CONFIG_ONLY_API_BASES (and add a "
        "config_dict[...] redirect in modify_config_for_fixtures), else a fixture run may dial "
        "real infra (the #1527 litellm/homelab:4001 bug)."
    )
    # The classification must not name fields that no longer exist.
    assert (
        classified <= all_api_bases
    ), f"Stale classified fields no longer on Config: {sorted(classified - all_api_bases)}"


def test_config_only_api_bases_are_redirected_to_the_mock(tmp_path, monkeypatch) -> None:
    """The core guard: a config that pins litellm/qwen/vllm at a real host must come out of
    ``modify_config_for_fixtures`` pointing at the 127.0.0.1 mock."""
    # The harness mutates os.environ directly (base URLs + dummy keys). Swap in a copy so
    # monkeypatch restores the real environment at teardown — no cross-test pollution.
    monkeypatch.setattr(os, "environ", dict(os.environ))

    src = tmp_path / "profile_config.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "rss": "https://real.example.com/feed.xml",
                "feeds": ["https://real.example.com/feed.xml"],
                "litellm_api_base": "http://homelab:4001/v1",
                "qwen_api_base": "http://homelab:4002/v1",
                "vllm_api_base": "http://homelab:4003/v1",
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    out_path = modify_config_for_fixtures(
        src, _FakeE2EServer(), session_dir=tmp_path, run_output_dir=run_dir, use_fixtures=True
    )
    out = yaml.safe_load(out_path.read_text(encoding="utf-8"))

    for field in sorted(_CONFIG_ONLY_API_BASES):
        val = str(out.get(field, ""))
        assert val.startswith("http://127.0.0.1"), (
            f"{field} was not redirected to the E2E mock (got {val!r}). "
            "Add a config_dict[...] override in modify_config_for_fixtures."
        )
