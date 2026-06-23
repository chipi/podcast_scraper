"""RFC-097 fingerprint gap closure (FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md §4):
runtime.inference_target classifies WHERE inference happens. Operator-flagged:
two ``provider_type: openai`` runs may be against local Ollama, DGX vLLM, or
the cloud OpenAI API — the today-fingerprint can't tell them apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.data.materialize_baseline import (  # noqa: E402
    _classify_inference_target,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "base_url, backend_type, expected",
    [
        # Local Ollama (port 11434)
        ("http://localhost:11434/v1", "openai", "local-ollama"),
        ("http://127.0.0.1:11434/v1", "openai", "local-ollama"),
        ("http://localhost:11434/v1", "ollama", "local-ollama"),
        # Local vLLM (any other localhost port)
        ("http://localhost:8003/v1", "openai", "local-vllm"),
        ("http://0.0.0.0:8000/v1", "openai", "local-vllm"),
        # DGX (Tailscale / dgx- hostname)
        ("http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1", "openai", "dgx-vllm"),
        ("http://dgx-llm-1:8003/v1", "openai", "dgx-vllm"),
        # In-process HF
        (None, "hf_local", "local-hf"),
        # Cloud APIs (no base_url)
        (None, "openai", "cloud-openai"),
        (None, "anthropic", "cloud-anthropic"),
        (None, "gemini", "cloud-gemini"),
        (None, "deepseek", "cloud-deepseek"),
        (None, "mistral", "cloud-mistral"),
        (None, "grok", "cloud-grok"),
        # Ollama on no base_url (rare; defaults to local socket)
        (None, "ollama", "local-ollama"),
        # Remote vllm (custom hostname, neither localhost nor DGX pattern)
        ("http://my-vllm.example.com/v1", "openai", "remote-vllm"),
        # Unknown
        (None, None, "unknown"),
        ("", "", "unknown"),
    ],
)
def test_classify_inference_target(base_url, backend_type, expected) -> None:
    assert _classify_inference_target(base_url, backend_type) == expected


def test_ollama_vs_dgx_distinction_drives_different_fingerprints() -> None:
    """The headline regression this gap-closure prevents: same
    ``provider_type: openai``, same model alias, but inference target differs
    materially. Distinct classifications → distinct fingerprint values."""
    local = _classify_inference_target("http://localhost:11434/v1", "openai")
    dgx = _classify_inference_target("http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1", "openai")
    assert local == "local-ollama"
    assert dgx == "dgx-vllm"
    assert local != dgx
