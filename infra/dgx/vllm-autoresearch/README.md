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
server**. We just deploy NVIDIA's pre-built `nvcr.io/nvidia/vllm:*-py3`
image directly. The compose passes the model id + serving args to the
in-image `vllm serve` entrypoint.

This mirrors how the operator's other vLLM containers
(`~/docker-compose/vllm-Qwen3-Coder-Next/`, `~/docker-compose/vllm-openwebui/`)
work — the well-trodden NVIDIA-recommended pattern for vLLM on GB10.

## Default image + model (post-#928 Cell C)

Defaults shipped by `deploy.py`:

- **Image**: `nvcr.io/nvidia/vllm:26.05-py3`
- **Model**: `Qwen/Qwen3.6-35B-A3B` (bf16, ~67 GB on disk)
- **GPU memory utilization**: 0.60 (coexists with whisper / pyannote)
- **Max model len**: 32768
- **Max num seqs**: 128 (Mamba-cache-block limit specific to qwen3_5_moe)

**Why these specifics — important constraints discovered during #928 Cell C**:

- `qwen3_5_moe` architecture requires transformers 5.x. NVIDIA's vLLM
  images 25.11–26.04 all ship transformers 4.57.x and reject the model.
  Only 26.05-py3 (vLLM 0.20.1 / transformers 5.6.0) supports it and
  boots cleanly on this operator's GB10. The post-release hotfix
  `26.05.post1-py3` was verified broken on this DGX — do not use.
- Qwen3.6-35B-A3B uses Mamba layers; vLLM 0.20 enforces
  `max_num_seqs ≤ Mamba cache blocks`. The default 256 fails CUDA-graph
  capture (only 190 cache blocks available at `gpu_memory_utilization=0.60`).
  128 is well under the limit with headroom.
- **Consumer contract**: the model emits a "thinking process" reasoning
  preamble by default (same pattern as DeepSeek R1-Distill). Production
  consumers MUST pass `chat_template_kwargs={"enable_thinking": False}`
  in the chat completions request to suppress it. The
  `summary_vllm_predict_v1.py` harness has a `--disable-thinking` flag
  that does exactly this — copy that pattern for any new caller.

### To swap to a different model

1. Edit `VLLM_MODEL` (and possibly `VLLM_IMAGE`, `VLLM_GPU_MEM_UTIL`,
   `VLLM_MAX_NUM_SEQS`) near the top of `infra/dgx/converge/deploy.py`.
2. If the model isn't already cached on the DGX, ensure HF_TOKEN is set
   in `/home/markodragoljevic/.env` and let vLLM download on first boot
   (~67 GB for Qwen3.6-35B-A3B; slower for FP16/BF16 variants).
3. Adjust `--max-model-len` to the model's actual context budget and
   `--gpu-memory-utilization` to the GB10's 128 GB unified memory budget
   minus other services. The operator's reference composes give the
   calibration: a 30B FP8 model wants `0.92` with 131072; a 35B bf16 MoE
   model wants `0.60` with 32768.
4. Re-run `make dgx-deploy`. Docker compose recreates the container only
   when the compose YAML changes; the model load on first boot will be
   slow (~5-15 min for cold cache, ~2-5 min when cached).

### To revert to R1-Distill on the older image

The R1-Distill-32B alternative remains in `/opt/llm-models/huggingface/`
on the DGX:

1. Set `VLLM_IMAGE = "nvcr.io/nvidia/vllm:25.11-py3"`
2. Set `VLLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"`
3. Remove the `--max-num-seqs` block from the compose `command` list
   (R1-Distill isn't Mamba-based; the constraint doesn't apply).
4. Re-run `make dgx-deploy`.

Or for an ad-hoc revert without touching the committed defaults, on the
DGX directly: `sudo cp /opt/vllm-autoresearch/docker-compose.yml.r1-distill.bak
/opt/vllm-autoresearch/docker-compose.yml` then
`sudo docker compose -f /opt/vllm-autoresearch/docker-compose.yml up -d`.

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
- `--gpu-memory-utilization 0.60` — leaves headroom for whisper /
  pyannote / Ollama. Bump higher only if those are stopped.
- `--max-model-len 32768` — enough for podcast transcripts; smaller
  than Qwen3.6's native max but fits the GPU memory budget at 0.60.
- `--max-num-seqs 128` — Mamba-cache-block limit. Required for
  Qwen3.5/3.6 family at gpu_mem_util=0.60. Drop if you switch to a
  non-Mamba model.

## Validating

After `make dgx-deploy`:

```bash
# 1. Health — only 200 after model is fully loaded (5-15 min on cold cache)
curl -s http://your-dgx.tailnet.ts.net:8003/health
# Expected: empty body, HTTP 200

# 2. Models endpoint
curl -s http://your-dgx.tailnet.ts.net:8003/v1/models | jq
# Expected: data list including the configured model id

# 3. End-to-end chat completion smoke (with enable_thinking=False)
curl -s http://your-dgx.tailnet.ts.net:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.6-35B-A3B", "messages": [{"role":"user","content":"Reply with the single word OK"}], "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": false}}'
# Expected: choices[0].message.content == "OK" (or close)
# Without chat_template_kwargs, Qwen3.6 emits a "thinking process" preamble
# instead of the answer — see Default model section above.
```

## Operational notes

- **Coexists with other DGX services**: port 8003 is non-conflicting
  (8000=speaches, 8001=pyannote, 8002=openai-whisper, 11434=ollama).
- **GPU memory budget**: at `gpu-memory-utilization=0.60`, vLLM
  reserves ~73 GB of the GB10's 128 GB unified memory. That coexists
  with whisper-openai (~3 GB), pyannote (~3-4 GB), faster-whisper
  (~3-4 GB), and an Ollama-loaded model (~10 GB). Concurrent heavy
  use of all four is borderline — for autoresearch sweeps focused on
  one stage (e.g., transcription), stop the others.
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
