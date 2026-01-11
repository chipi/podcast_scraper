# ADR-037: Local LLM Backend Abstraction

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md)

## Context & Problem Statement

Local LLM inference requires different backends depending on the hardware (PyTorch for NVIDIA GPUs, `llama.cpp` for Apple Silicon). Hardcoding one library would limit the project's portability.

## Decision

We implement a **Local LLM Backend Abstraction**.

- The `hybrid_ml` provider interacts with backends via a simple `InferenceBackend` protocol.
- Supported backends include:
  - **`transformers`**: For standard PyTorch/GPU execution.
  - **`llama_cpp`**: Optimized for CPU and Apple Silicon (Metal).
  - **`ollama`**: For users who prefer a managed local inference server.

## Rationale

- **Hardware Portability**: Allows the scraper to run with peak performance on both Mac laptops and Linux servers.
- **Optimization**: We can leverage GGUF/quantized models via `llama.cpp` to run 7B-14B models in <16GB of RAM.
- **Future-Proofing**: Easy to add new backends (e.g., vLLM, TensorRT-LLM) without touching the summarization logic.

## Alternatives Considered

1. **PyTorch-Only**: Rejected as it is inefficient for LLM inference on consumer-grade Apple hardware.

## Consequences

- **Positive**: Broad hardware support; minimal memory footprint; flexible deployment options.
- **Negative**: Requires managing multiple optional system dependencies.

## References

- [RFC-042: Hybrid Podcast Summarization Pipeline](../rfc/RFC-042-hybrid-summarization-pipeline.md)
