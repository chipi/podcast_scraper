"""Unit tests for the autoresearch sweep driver's per-model prompt wiring.

The sweep used to override only ``backend.model`` in the materialized config
(every candidate scored on the same shared bullet-JSON prompt — see the W27
ledger). With per-model tuned prompts now wired in, each candidate is scored
on the templates we'd ship it with in production. A candidate that lacks
tuned prompts must fail fast rather than silently degrade on a foreign prompt.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

sweep = importlib.import_module("scripts.baselines.autoresearch_sweep")


def _seed_prompt_dir(root: Path, model_safe: str) -> Path:
    d = root / "src/podcast_scraper/prompts/ollama" / model_safe / "summarization"
    d.mkdir(parents=True, exist_ok=True)
    (d / "system_v1.j2").write_text("system stub", encoding="utf-8")
    (d / "long_v1.j2").write_text("long stub", encoding="utf-8")
    return d


def test_resolve_per_model_prompts_present(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    _seed_prompt_dir(tmp_path, "llama3.1_8b")
    resolved = sweep._resolve_per_model_prompts("llama3.1:8b")
    assert resolved == (
        "ollama/llama3.1_8b/summarization/system_v1",
        "ollama/llama3.1_8b/summarization/long_v1",
    )


def test_resolve_per_model_prompts_keeps_dots(monkeypatch, tmp_path):
    """``qwen3.5:9b`` must resolve to ``qwen3.5_9b`` — dots preserved, only
    the ``:`` replaced. Earlier sweep code stripped dots too (for filenames),
    which would silently miss the existing per-model dir on disk."""
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    _seed_prompt_dir(tmp_path, "qwen3.5_9b")
    resolved = sweep._resolve_per_model_prompts("qwen3.5:9b")
    assert resolved is not None
    assert "qwen3.5_9b" in resolved[0]


def test_resolve_per_model_prompts_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    # No prompt dir seeded.
    assert sweep._resolve_per_model_prompts("brand_new_model:1b") is None


def test_resolve_per_model_prompts_partial_missing_returns_none(monkeypatch, tmp_path):
    """If only ``system_v1.j2`` exists (no ``long_v1.j2``), treat as missing —
    don't silently mix a tuned system prompt with a missing user prompt."""
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    d = tmp_path / "src/podcast_scraper/prompts/ollama/halfway_3b/summarization"
    d.mkdir(parents=True)
    (d / "system_v1.j2").write_text("only system", encoding="utf-8")
    assert sweep._resolve_per_model_prompts("halfway:3b") is None


def test_materialize_candidate_config_writes_tuned_prompts(monkeypatch, tmp_path):
    """Materialized config must override BOTH backend.model AND prompts.{system,user}
    with the per-model tuned templates — not just backend.model (the W27 bug)."""
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    _seed_prompt_dir(tmp_path, "hermes3_8b")

    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "id": "baseline",
                "backend": {"type": "ollama", "model": "PLACEHOLDER"},
                "prompts": {"system": "shared/bullets_v1", "user": "shared/bullets_user_v1"},
                "data": {"dataset_id": "curated_5feeds_smoke_v2"},
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "tmp"
    out_dir.mkdir()
    cfg_path, source = sweep._materialize_candidate_config(base_config, "hermes3:8b", out_dir)
    assert source == "tuned"
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["backend"]["model"] == "hermes3:8b"
    assert written["prompts"]["system"] == "ollama/hermes3_8b/summarization/system_v1"
    assert written["prompts"]["user"] == "ollama/hermes3_8b/summarization/long_v1"


def test_materialize_candidate_config_returns_none_when_prompts_missing(monkeypatch, tmp_path):
    """Missing prompts → (None, 'missing_prompts'); the caller records a
    fail-fast row rather than producing a degraded score."""
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    # No prompt dir seeded for "ghost:1b".
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "id": "baseline",
                "backend": {"type": "ollama", "model": "PLACEHOLDER"},
                "prompts": {"system": "shared/x", "user": "shared/y"},
                "data": {"dataset_id": "anything"},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "tmp"
    out_dir.mkdir()
    cfg_path, source = sweep._materialize_candidate_config(base_config, "ghost:1b", out_dir)
    assert cfg_path is None
    assert source == "missing_prompts"
