# DGX serving — how this pipeline reaches its local inference (bridge)

**This is a bridge, not the source of truth.** The DGX GPU box and its services are owned,
deployed, and documented in the **`agentic-ai-homelab`** repo — that is where DGX serving is
declared and changed. This page is the short consumer-side view: what *this* pipeline depends
on, how to reach it, and where the full docs live. Keep DGX serving detail in the homelab repo;
update only this thin bridge here.

## Source of truth (the full docs)

| What | Where (in `agentic-ai-homelab`) |
| --- | --- |
| DGX service map + host tooling | `infra/dgx/README.md` |
| GPU-mode coordination (the bring-up) | `infra/dgx/bin/README.md` + `docs/recipes/gpu-mode-swap.md` |
| vLLM composes (coder / autoresearch) | `infra/vllm/README.md` |
| Telemetry / health-scrape | `infra/dgx-scrape/README.md` |

## Access (verified 2026-08-02)

- The DGX runs as **`ops`** from `/home/ops/agentic-ai-homelab` (a **deploy-only** checkout —
  never commit on the box). Reach it over Tailscale: **`ssh ops@dgx-llm-1`** (`100.69.49.126`).
- `gpu-mode-swap.sh` is on PATH at **`/usr/local/bin/gpu-mode-swap.sh`** — always call by that
  absolute path in non-interactive shells (the `gpu-mode` alias only loads interactively).

## Services this pipeline consumes

A full run is a **cascade**: ASR → diarization → (naming + summarization) LLM. **Two roles:**

**Fixed supporting services** — already in the pipeline, unchanged; must be up for every full run,
but NOT what v2.5 researches:

| Pipeline stage | DGX service | Port | Notes |
| --- | --- | --- | --- |
| ASR (v2.3, faster-whisper turbo) | faster-whisper | 8000 | `openai-whisper` on 8002 is the failover peer |
| ASR coverage-failover (ADR-123, #1273) | MOSS | 8004 | transcription fallback; saturates under load |
| Diarization (v2.2, community-1) | pyannote | 8001 | exposes Prometheus `/metrics` |
| Text LLM (local, misc) | ollama | 11434 | e.g. `llama3.1:8b` |

**The v2.5 research target** — the ONE thing being swapped (Gemini → DGX-local):

| Pipeline stage | DGX service | Port | Notes |
| --- | --- | --- | --- |
| Speaker naming + summarization LLM | vLLM (autoresearch) | 8003 | **GPU-mode-gated** — see below. Phase D bakes off local LLM candidates here to match the Gemini baseline. |

Only the naming/summarization **LLM** changes in v2.5. ASR, diarization, and the ASR failover stay
exactly as they are — they support the run, they are not under evaluation.

## The one gotcha: the autoresearch vLLM is GPU-mode-gated

A single GB10 GPU can't host the coder-next vLLM and the autoresearch vLLM at once. When the
mode is `free`/`idle`, `:8003` serves nothing — that is **idle, not gone**. Before the v2.5
bake-off (Phase D), bring it up:

```bash
ssh ops@dgx-llm-1 /usr/local/bin/gpu-mode-swap.sh --mode-only   # code | research | idle | free
ssh ops@dgx-llm-1 /usr/local/bin/gpu-mode-swap.sh research       # bring autoresearch vLLM up
```

Never use `code`/coder-next (the operator's IDE backend). Health is **TCP-open, not HTTP** —
inference servers stop answering `/health` under load while still serving.

## Related permanent docs in this repo

- [ADR-122](../adr/ADR-122-self-hosted-model-resilience-policy.md) — self-hosted model resilience (circuit-breaker on DGX contention).
- [ADR-123](../adr/ADR-123-quality-gate-transcription-failover.md) — transcription failover (faster-whisper → MOSS).
- [ADR-142](../adr/ADR-142-litellm-prod-gateway.md) — the prod-VPS LiteLLM gateway (a *different* failure domain from the DGX).
- [ADR-143](../adr/ADR-146-corpus-reprocess-methodology.md) — the corpus arc; the v2.5 bake-off runs on the autoresearch vLLM above.
- Profiles that target DGX services: `config/profiles/prod_dgx_*.yaml`, `cloud_with_dgx_primary.yaml`.
