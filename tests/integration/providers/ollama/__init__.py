"""Integration tests for model-specific Ollama providers.

These tests verify model-specific prompt loading and behavior for:
- Llama 3.1 8B (Issue #394)
- Mistral 7B (Issue #395)
- Phi-3 Mini (Issue #396)
- Gemma 2 9B (Issue #397)
- Qwen 2.5 7B (Issue #430)
- Qwen 2.5 32B (same prompt layout as 7B; larger Ollama tag)
- Qwen 3.5 9B / 27B / 35B / 35B-A3B (OLLAMA_PROVIDER_GUIDE three-tier checklist)
"""
