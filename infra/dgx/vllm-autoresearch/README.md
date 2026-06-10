# DGX vllm-autoresearch — vLLM serving for autoresearch evaluations (#928)

vLLM service running NVIDIA's pre-built container on the DGX GB10, exposing
an OpenAI-compatible API for autoresearch evals. Listens on port **8003**.

| Verb | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness — 200 once vLLM has loaded the model |
| `GET` | `/v1/models` | OpenAI-style model listing |
| `POST` | `/v1/chat/completions` | OpenAI-style chat (used by autoresearch summary/GI/KG scoring) |
| `POST` | `/v1/completions` | OpenAI-style completion |

## Why this exists

`cloud_balanced` + `cloud_thin` profiles currently route summary + GI + KG
to `gemini-2.5-flash-lite`. `cloud_with_dgx_*` profiles route to Ollama
(per the #932 finale). vLLM is the **third local-serving option** — same
DGX hardware as Ollama but a different serving framework.

The #928 championship asks: across all three LLM stages (summary, GI,
KG), does a vLLM-served open-weight model produce production-quality
results, and how does it compare on quality + latency + reliability to:

- The current cloud incumbent (`gemini-2.5-flash-lite`)
- The current local champion (`qwen3.5:35b` on Ollama)
- Each other (model architecture variations on vLLM)

This service is the vLLM endpoint that autoresearch eval scripts hit.
It coexists with the Ollama daemon (port 11434) and the whisper services
(8000 = faster-whisper, 8002 = openai-whisper) on different ports.

## Why no Dockerfile

Unlike `pyannote-server` and `whisper-server` which wrap a Python
library in our own FastAPI app, **vLLM ships its own OpenAI-compatible
server**. We just deploy NVIDIA's pre-built `nvcr.io/nvidia/vllm:25.11-py3`
image directly. The compose passes the model id + serving args to the
in-image `vllm serve` entrypoint.

This mirrors how the operator's other vLLM containers
(`~/docker-compose/vllm-Qwen3-Coder-Next/`, `~/docker-compose/vllm-openwebui/`)
work — the well-trodden NVIDIA-recommended pattern for vLLM on GB10.

## Default model + how to switch

The `deploy.py` recipe ships a compose configured for
**`Qwen/Qwen3-Coder-Next-FP8`** by default — the model already cached on
the operator's DGX HF cache from prior agentic-coding work (~30B params
in FP8, fits in GB10 at `gpu-memory-utilization=0.92`).

This is **NOT** the final autoresearch panel. Qwen3-Coder-Next is code-
tuned and may be a poor fit for prose summarization. The first autoresearch
sweep is partly to find out — see #928 panel design — and partly to
validate the vLLM serving path works for our shape of request.

To swap in a different model:

1. Edit `VLLM_MODEL` near the top of `infra/dgx/converge/deploy.py`.
2. If the model isn't already cached on the DGX, ensure HF_TOKEN is set
   in `/home/markodragoljevic/.env` and let vLLM download on first boot
   (~30 GB for a typical 30B FP8; slower for FP16/BF16 variants).
3. Adjust `--max-model-len` and `--gpu-memory-utilization` to the model's
   actual size and the GB10's 128 GB unified memory budget. The operator's
   working composes give the calibration: a 30B FP8 model wants
   `gpu-memory-utilization=0.92` with `max-model-len=131072`; a 7B model
   wants `~0.60` with `max-model-len=8192`.
4. Re-run `make dgx-deploy`. Docker compose recreates the container only
   when the compose YAML changes; the model load on first boot will be
   slow (~5-15 min for cold cache, ~2-5 min when cached).

## Configuration

Environment variables (defaults match deploy.py compose):

- `HF_HOME` (`/root/.cache/huggingface` inside container, mounted from
  `/opt/llm-models/huggingface` outside) — shared with all other DGX
  services so weights aren't duplicated.
- `VLLM_DISABLE_TORCH_COMPILE=1` — disables vLLM's torch.compile path.
  Per the operator's working composes, this is the GB10 hot fix:
  torch.compile has historically been the sore spot for the
  Blackwell-PTX-vs-CUDA-runtime versioning issue. Disabling it loses a
  ~10-20% steady-state throughput optimization but is the reliable
  start path.
- `NVIDIA_VISIBLE_DEVICES=all` — passes all GPUs (single GB10 today).

vLLM serve args (in the compose `command:` block):

- `--host 0.0.0.0` + `--port 8003` — bind to all interfaces (host
  networking). Tailnet ACL gates real external access.
- `--dtype bfloat16` — Blackwell-native.
- `--gpu-memory-utilization 0.92` — high because Qwen3-Coder-Next is
  large. Adjust per model.
- `--max-model-len 131072` — Qwen3's full context. Smaller for smaller
  models.
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder` — only
  relevant for tool-using deployments (the operator's coding flow).
  Autoresearch chat completions don't use these; harmless to leave on.

## Validating

After `make dgx-deploy`:

```bash
# 1. Health — only 200 after model is fully loaded (5-15 min on cold cache)
curl -s http://dgx-llm-1.tail6d0ed4.ts.net:8003/health
# Expected: empty body, HTTP 200

# 2. Models endpoint
curl -s http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1/models | jq
# Expected: data list including the configured model id

# 3. End-to-end chat completion smoke
curl -s http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-Coder-Next-FP8", "messages": [{"role":"user","content":"Reply with the single word OK"}], "max_tokens": 8}'
# Expected: choices[0].message.content == "OK" (or close)
```

## Operational notes

- **Coexists with other DGX services**: port 8003 is non-conflicting
  (8000=speaches, 8001=pyannote, 8002=openai-whisper, 11434=ollama).
- **GPU memory budget**: at `gpu-memory-utilization=0.92`, vLLM
  reserves ~118 GB of the GB10's 128 GB unified memory. **Don't run
  vLLM concurrently with heavy Ollama models** — there isn't enough
  headroom. For autoresearch sweeps, stop Ollama (or scope to small
  models) when running vLLM; or use the operator's coding compose with
  `gpu-memory-utilization=0.60` for ~76 GB instead.
- **First request is slow**: vLLM warms its CUDA graphs on the first
  call (~5-30 s of one-time setup). Subsequent calls are fast.
- **Healthcheck patience**: the compose's healthcheck allows 10 min for
  model load (30 retries × 30s). Cold cache + large model can hit that.

## References

- This issue: #928 (summary/GI/KG championship)
- Operator's reference composes: `~/docker-compose/vllm-Qwen3-Coder-Next/`
  and `~/docker-compose/vllm-openwebui/` on the DGX
- Sibling DGX services: `infra/dgx/pyannote-server/` (#926),
  `infra/dgx/whisper-server/` (#953), `infra/dgx/speaches-gb10/` (#948)
- Finale framework reused for scoring: `src/podcast_scraper/evaluation/finale_runner.py`
  plus `g_eval.py` and judges (from #949)
