# LLM Model Playbook — vLLM-on-GB10 (autoresearch)

Per-model serving notes accumulated from #1016 (LLM landscape Phase 2). Living
doc — add new models / new tweaks as they're discovered. Companion to
[`JUDGING.md`](JUDGING.md) (methodology rules).

For each model we capture:

- **HF model id + size on disk + arch shape** (total / active params, MoE / dense)
- **Required `vllm serve` flags** (max-model-len, max-num-batched-tokens,
  trust-remote-code, reasoning-parser, etc.)
- **Recommended sampling** (temperature, top_p — vendor's documented values
  or our empirical findings)
- **Prompt convention** (specific system / user template patterns the model
  responds well to, like `/no_think` for Nemotron or `enable_thinking: false`
  for Qwen3)
- **Required postprocessor** in the eval harness (e.g. byte-level decode for
  DeepSeek-V2, R1-Distill family)
- **Known failure modes + workarounds**
- **Citation** (HF card, vendor blog, vLLM doc — what's authoritative)

Use this as the FIRST stop before launching a new autoresearch sweep with an
unfamiliar model. It will save you 30-60 min of debugging per model that has
non-default requirements.

---

## Model Onboarding Checklist (mandatory before any cohort run)

This is the hard-won lesson of #1016 Phase 2c: a model is NOT ready for an
autoresearch sweep just because its weights downloaded and `/health` returned
200. Every model has its own prompt convention, sampling regime, and quirks.
Onboarding is the dedicated phase where you discover them — BEFORE the model
joins the cohort.

**Rule**: every new model goes through this checklist before it produces any
data that lands in an eval scorecard. Document everything in this playbook's
per-vendor section as you learn it.

### Step 1 — Documentation pass

Read the HF model card end-to-end. Don't skim. Extract verbatim quotes for:

- **Sampling parameters** the vendor recommends (separate values for reasoning
  on / reasoning off if applicable)
- **System prompt conventions** for reasoning control, tool calling, or output
  shape (e.g. `/no_think` for Nemotron, `enable_thinking: false` chat-template
  kwarg for Qwen3)
- **Stop tokens / EOS** behavior
- **Chat template** specifics (does the model want `<|im_start|>`, `[INST]`, or
  `<|begin_of_text|>`?)
- **Any vLLM-specific flags** the vendor lists

If the model card is sparse, also check:

- The model's GitHub repo (if open-weight) for an inference example
- The vendor's blog post for any deployment guidance
- vLLM's supported-models docs page for arch + flag notes
- The DGX-Spark blog (`vllm.ai/blog/2026-06-01-vllm-dgx-spark`) for GB10
  recommendations

### Step 2 — Architecture + quantization fit check

Verify before downloading 100 GB:

- **Architecture name** (`config.json` → `architectures[0]`) — is it in vLLM's
  supported models list? `LlamaForCausalLM`, `Gemma4ForConditionalGeneration`,
  `DeepseekV4ForCausalLM`, etc.
- **Quantization** (`config.json` → `quantization_config.quant_method` or HF
  tag) — does our vLLM image's kernel support it on Blackwell? NVFP4 / FP8
  preferred; INT4 (`gptq`, `awq`, `bnb`) is risky on Blackwell.
- **Memory cost** estimate: `total_params × bytes_per_param + ~10 GB KV/CUDA
  budget` — does it fit GB10's 128 GB unified pool at our default
  `gpu_memory_utilization`?

### Step 3 — Boot test (smoke for the serving stack)

After swap + container restart:

1. Wait for `/health` 200
2. `curl /v1/models` — verify the model id is correct
3. `curl /v1/chat/completions` with a 5-token user message + `max_tokens=20`.
   Example body:

   ```json
   {"model":"autoresearch","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":20,"temperature":0}
   ```

4. **Verify the response**:
   - 200 status, well-formed JSON
   - `choices[0].message.content` is non-empty
   - **NO byte-level BPE artifacts** (`Ġ`, `Ċ`, `âĢĻ` in plain output) — if
     present, you need a tokenizer-decode postprocessor (e.g.
     `decode_r1_byte_level`)
   - **NO unexpected `<think>` block** — if present, the reasoning control
     isn't working
   - `finish_reason` is `stop` (not `length`)

### Step 4 — Smoke summary test

Single full episode summary:

1. Pick a small representative episode (`p01_e01` works — it's our canonical
   smoke episode)
2. Fire the request through the harness with the planned config.
   Example command:

   ```bash
   python scripts/eval/experiment/run_experiment.py \
     data/eval/configs/summarization/<your_config>.yaml \
     --vllm-base-url "$VLLM_API_BASE" --dry-run --force
   ```

3. **Verify on the resulting prediction**:
   - `finish_reason=stop`, not `length`
   - Length in spec (800-3200 chars for our 200-800 token spec)
   - Output is real prose, not transcript echo, not Q&A pairs, not markdown
     headers, not reasoning preamble
   - No safety refusal text ("I'm sorry, but I can't...")
   - Sample-eyeball the content — does it actually summarize?

### Step 5 — Document in this playbook

Add the model to its vendor section with:

- HF id + size
- Required vLLM flags
- Recommended sampling
- Prompt convention (cite the vendor docs)
- Required postprocessor (if any)
- Known failure modes from your boot/smoke tests

Also update the quick-reference table at the top of this doc.

### Step 6 — If onboarding fails

If after **3 distinct attempts** the model still doesn't pass the smoke tests:

- Document each failure mode in the per-model section with empirical evidence
  (which episode, output preview, error)
- Mark status as "needs investigation" — NOT "dropped"
- Flag to the operator with: what you tried, what failed, what's left to try
- **The decision to drop is the operator's, not the agent's**

### Anti-patterns observed during #1016

These specific failure modes informed this checklist; avoid them in future
onboarding:

- **Assuming a model "works" because the container booted to /health 200.**
  Doesn't matter if the first inference call crashes the container (e.g.
  Mistral-Small-4-119B-NVFP4 OOM on first request after a healthy boot).
- **Using a prompt convention from one model on a different model.**
  "detailed thinking off" works for some Qwen variants but NOT for Nemotron
  (which wants `/no_think`). The convention is per-vendor — don't generalize
  from training intuition.
- **Adding negative-constraint stacks ("Do NOT do X, Do NOT do Y") in prompts.**
  Some models (Kimi-Linear specifically) interpret this as adversarial /
  jailbreak attempts and refuse. Use positive instructions where possible.
- **Declaring a model failed after 1-2 attempts.** Real onboarding can take
  3-5 attempts to find the right combination. Don't shrink scope prematurely.
- **Skipping the documentation pass and jumping straight to inference.** Saves
  10 minutes of reading, costs 1-2 hours of debugging.

---

## vLLM-on-GB10 — general guidance

This section is cross-cutting: applies to every model served via our
autoresearch vLLM stack. Per-model overrides are in the sections below.

### Hardware envelope (DGX-Spark / GB10 Blackwell)

- **Single GPU, 128 GB unified memory pool** (CPU + GPU share the same physical
  RAM). Memory budget is the single hardest constraint.
- **sm_121 Blackwell consumer-tier**: ~900 GB/s memory bandwidth (vs H100's
  ~3 TB/s and B200's ~8 TB/s). Bandwidth is the bottleneck for single-stream
  inference; compute is rarely the constraint at single-stream loads.
- **Native quantization preference (per vLLM-DGX-Spark blog)**: NVFP4 > FP8 > BF16.
  NVFP4 reduces memory pressure and fits the bandwidth envelope. Avoid INT4 GGUF
  flavors — vLLM's NVFP4 kernel (`FlashInferCutlassNvFp4LinearKernel`) is
  hardware-aware on Blackwell; INT4 is software-emulated.

### Image, version, compose layout

- Image we run: `nvcr.io/nvidia/vllm:26.05-py3` (vLLM 0.20.1+ tooling)
- Compose: `~/agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml`
- `.env`: `~/agentic-ai-homelab/infra/vllm/autoresearch/.env`
- Volume mounts: `HF_HOME=/opt/llm-models/huggingface` and
  `VLLM_CACHE_PATH=/opt/llm-models/vllm-cache` are bind-mounted host:container
  identically. Models downloaded inside the container land at the same path on
  the host (and vice versa).
- DGX-Spark blog (2026-06-01) recommends `vllm/vllm-openai:cu130-nightly` as the
  newer tested track; we haven't upgraded yet (#1022 follow-up).

### Compose flags we settled on (current as of #1016 Phase 2c)

```yaml
command:
  - vllm
  - serve
  - <MODEL_HF_ID>          # swapped per model via swap_vllm_model.py or sed
  - --host=0.0.0.0
  - --port=${VLLM_PORT:-8003}
  - --api-key=${VLLM_API_KEY:-buddy-is-the-king}
  - --gpu-memory-utilization=${VLLM_GPU_MEM_UTIL:-0.75}   # per-model override in .env
  - --max-model-len=32768                                  # reduce to 8192 for some models
  - --max-num-batched-tokens=4096                          # added in #1016 for Gemma 4
  - --served-model-name=autoresearch
  # If model requires trust_remote_code (e.g. Kimi-Linear):
  # - --trust-remote-code
  # If reasoning parser needed:
  # - --reasoning-parser=<deepseek_r1 | mistral | nemotron_v3 — see per-model section>
```

### How to swap models (the dance)

1. `ssh <DGX> 'sed -i "s|^      - <OLD_MODEL>|      - <NEW_MODEL>|" <compose>'`
2. Adjust `VLLM_GPU_MEM_UTIL=X.XX` in `.env` if the new model needs different
   util budget (per-model table in this doc has the recommended value)
3. `gpu-mode-swap.sh idle` → 3 seconds → `gpu-mode-swap.sh research`
4. Poll `/health` on the tailnet endpoint (`http://<DGX>:8003/health` returns
   200 when ready)
5. Expect 5-40 min boot depending on download + load + compile cost. See "Boot
   stages" below.

### Boot stages (where the time goes)

Cold boot from scratch with no cached weights:

| Stage | Typical time | Notes |
| ------- | -------------- | ------- |
| Image pull (if first time) | 0-5 min | One-shot |
| HF weight download | 5-15 min | Depends on model size (~2-3 GB/min) |
| Safetensors shard load | 3-15 min | ~30s/shard × number of shards |
| torch.compile (first time) | 5-10 min | Cached afterward — subsequent boots ~1-5s |
| torch.compile (cached) | 1-5s | AOT artifacts in `VLLM_CACHE_PATH` |
| FlashInfer autotune | 5-30s | Per-model; one-shot per first-boot |
| CUDA graph capture | 30-90s | PIECEWISE=51 + FULL=35 typical |
| Profile + start server | 5-15s | |

**Re-boot from cached weights + cached compile**: typically 10-15 min total
(load + recompile profile + autotune + CUDA capture). The torch.compile cache
saves the biggest single chunk (5-10 min on first boot → <5s on re-boot).

### Memory budget reasoning

GB10's 128 GB unified pool serves: model weights + KV cache + activations +
CUDA graph capture buffers + OS/container overhead.

For a given model at NVFP4/FP8/BF16:

```text
budget = 128 GB * gpu_memory_utilization
weights_cost = <model size on disk>
cuda_graph_cost = ~5 GB (PIECEWISE+FULL, varies)
activations_cost = ~1-2 GB single-stream
kv_cache_budget = budget - weights_cost - cuda_graph_cost - activations_cost
```

**KV cache budget at default `--max-num-seqs=256` and `--max-model-len=32768`:**

- KV cache size ≈ `2 (k,v) × num_layers × num_heads × head_dim × max_seqs × max_model_len × bytes_per_param`
- For a 24B BF16 model: rough ballpark 25-40 GB KV cache. Not always achievable
  on GB10.

**For single-stream autoresearch (our case)**: `--max-num-seqs=4` per the
DGX-Spark blog drops KV cache cost dramatically. Apply this on any model that
OOMs at default settings (e.g. Mistral-Small-4-119B-NVFP4 — see its section).

### Common boot failures + workarounds

| Error | Cause | Fix |
| ------- | ------- | ----- |
| `Chunked MM input disabled but max_tokens_per_mm_item (N) is larger than max_num_batched_tokens (2048)` | Multimodal model's required mm-token budget exceeds default | `--max-num-batched-tokens=4096` |
| `max_model_len (32768) greater than derived max_model_len (max_position_embeddings=X)` | Model's positional embedding cap below our default | Reduce `--max-model-len` to model's `max_position_embeddings` value |
| `repository contains custom code which must be executed to correctly load the model. Please pass the argument trust_remote_code=True` | Custom tokenizer Python file in HF repo | `--trust-remote-code` |
| `NemotronV3ReasoningParser reasoning parser could not locate think start/end tokens in the tokenizer` | Reasoning parser flag's expected tokens missing from this checkpoint | Remove `--reasoning-parser=<X>` flag; control reasoning via prompt instead |
| OOM during inference (boot succeeded, first request crashes) | KV cache + weights + activations exceed budget | Reduce `--max-num-seqs` to 4 OR reduce `--max-model-len` OR raise `gpu_memory_utilization` (carefully) |
| Container restart loop with no obvious error | Image's required NVIDIA driver newer than host | Check `nvidia-smi` driver version vs container's required release; usually a Spark host driver bump is the right fix |

### Metrics we collect during inference (#1022)

vLLM exposes a Prometheus-format `/metrics` endpoint. Key signals we poll
every 30s while running an eval (stored at
`docs/wip/EVAL_1016_metrics/vllm_kv_metrics.log`):

```text
vllm:gpu_cache_usage_perc       # KV cache utilization — should stay low single-stream
vllm:kv_cache_usage_perc        # alternative cache metric
vllm:num_requests_running       # always 1 in single-stream
vllm:num_requests_waiting       # always 0 in single-stream
vllm:num_requests_swapped       # alert if non-zero
vllm:time_to_first_token_seconds_bucket  # TTFT histogram
vllm:prefix_cache_hits          # # prefix cache hits
vllm:prefix_cache_queries       # # total prefix cache queries
```

Per the DGX-Spark blog: single-user KV cache usage typically stays <5%,
small-batch <30%. If our observed % is much higher, we're likely overcommitted
on KV cache budget and at risk of OOM.

For each model we run we capture: KV % observed, TTFT median (from histogram),
boot wall-clock breakdown, model loading GB, and bookkeeping notes for the
issue 1022 systematic-tuning workstream.

### Things we tried and learned (cross-cutting)

- **`--enforce-eager`**: not used. CUDA graphs are enabled by default and
  provide 1.5-3× decode speedup on Blackwell. Don't disable without reason.
- **`--kv-cache-dtype=fp8`**: not used. Per blog, AVOID unless memory pressure
  demands it — perf cost on Blackwell single-stream is non-trivial.
- **`VLLM_DISABLE_TORCH_COMPILE=1`**: NOT used on the 26.05 image. Was needed on
  25.11 for GB10; the Blackwell torch.compile fix landed upstream and dropping
  this flag is required (it would emit a startup warning).
- **`--max-num-batched-tokens=4096`**: defaults to 2048 which is below some
  models' multimodal-mm token requirements. We bumped to 4096 for the cohort
  uniformly — has no downside for non-multimodal models, fixes Gemma 4.
- **`--max-model-len=131072`**: tested by the blog, we don't use. Our
  transcripts are ~3K tokens; 32K cap is comfortable. Lower = less KV cache
  pressure.
- **`--max-num-seqs=4`**: we have NOT set this in compose because most models
  fit comfortably at default 256. Mistral-Small-4-119B-NVFP4 requires it —
  per-model section flags this. Probably worth setting cohort-wide for
  single-stream autoresearch (#1022 follow-up).

### What we still want to test (#1022 follow-up)

- Image bump: `vllm:26.05-py3` → `vllm/vllm-openai:cu130-nightly` (blog's
  recommended track)
- `--gpu-memory-utilization 0.85` for models that fit (we mostly run 0.55-0.65)
- JIT pre-warm ping (~3-token request before live traffic) to absorb cold-start
  latency penalty
- `--max-num-seqs=4` as a cohort default for single-stream autoresearch
- `--enable-prefix-caching` (currently enabled by default, but verify behavior
  on the autoresearch swap dance)

See `PER_MODEL_OPTIMAL_PARAMS.md` (same directory) for the per-model
param exploration table that informs #1022.

---

## Quick-reference (vLLM serve flags + prompt convention + postprocessor)

| Model | Size | UTIL | max-model-len | max-num-batch | trust-remote | extra flags | Prompt | Postproc | Status |
| ------- | -----: | -----: | --------------: | --------------: | :------------: | ------------- | -------- | ---------- | -------- |
| Qwen3-30B-A3B-Instruct-2507 | ~60 GB BF16 | 0.65 | 32768 | 2048 (default) | no | `extra_body.chat_template_kwargs.enable_thinking: false` | `ollama/qwen3.5_35b/summarization/*` (max_length=500 to tighten) | none | ✓ working |
| Qwen3.5-35B-A3B | 67 GB BF16 | **0.65** | 32768 | 2048 | no | same as Qwen3-30B | same | none | ✓ working |
| Mistral-Small-3.2-24B-Instruct-2506 | ~48 GB BF16 | 0.55 | 32768 | 2048 | no | none | `ollama/qwen3.5_35b/summarization/*` | none | ✓ working |
| Magistral-Small-2509 | ~48 GB BF16 | 0.55 | 32768 | 2048 | no | `--reasoning-parser=mistral`, `--tokenizer-mode=mistral`, `--config-format=mistral`, `--tool-call-parser=mistral` | `ollama/qwen3.5_35b/summarization/*` | none | ✓ working |
| Mistral-Small-4-119B-2603-NVFP4 | 70.8 GB NVFP4 | 0.80 | **8192** | **4096** | no | + Mistral parser flags; **REQUIRED `--max-num-seqs=4`** | `ollama/qwen3.5_35b/summarization/*` | none | ⚠ OOM until num-seqs reduced |
| Ministral-3-14B-Instruct-2512 | 31.5 GB FP8 | 0.65 | 32768 | 2048 | no | none | standard | none | ✓ working |
| Gemma-4-26B-A4B-it | 51.6 GB BF16 | 0.65 | 32768 | **4096 REQUIRED** | no | none | standard | none | ✓ working |
| Llama-3.3-70B-Instruct-NVFP4 (RedHat) | 42.7 GB NVFP4 | 0.70 | 32768 | 4096 | no | none | **`vllm/strict_summarization/*` REQUIRED** (default prompt → length overflow) | none | ✓ working with strict prompt |
| Moonlight-16B-A3B-Instruct | 31.9 GB BF16 | 0.55 | **8192 REQUIRED** | 4096 | no | none | standard | none | ✓ working — **fastest DGX candidate (8.8s/ep)** |
| Kimi-Linear-48B-A3B-Instruct | 98.2 GB BF16 | 0.80 | 32768 | 4096 | **YES REQUIRED** | none | **problematic** — both strict + natural fail (refusals or chain-of-thought-in-prose) | none | ✗ flagged for further investigation |
| ~~DeepSeek-V2-Lite-Chat~~ **DROPPED 2026-06-22** | 31.4 GB BF16 | 0.55 | 32768 | 4096 | no | none | `vllm/strict_summarization/*` | `decode_r1_byte_level` (legacy) | ✗ **DROPPED** — KG 2% / GI 1% across temp 0.0/0.7 + guided_json on/off; emits 1 mega-Topic-as-summary + 0 typed entities per ep. See DSV2-Lite-Chat detailed section below. |
| DeepSeek-R1-Distill-Qwen-32B | 65 GB BF16 | 0.55 | 32768 | 2048 | no | `--reasoning-parser=deepseek_r1` | `vllm/r1_distill_32b/*` (anti-thinking) | **`strip_r1_reasoning_and_decode` REQUIRED** | ✓ working but 204s/ep — speed-disqualified |
| DeepSeek-R1-0528-Qwen3-8B | 16.4 GB BF16 | 0.55 | 32768 | 4096 | no | `--reasoning-parser=deepseek_r1` (still emits `<think>`) | tried `vllm/r1_distill_8b/*` + `enable_thinking: false` — both fail | `strip_r1_reasoning_and_decode` | ✗ flagged for further investigation |
| Nemotron-Super-49B-v1_5-FP8 | 52 GB FP8 | 0.65 | 32768 | 4096 | yes | **do NOT use `--reasoning-parser=nemotron_v3`** (boot loops with "could not locate think tokens") | system prompt MUST start with `/no_think` literal | none (with `/no_think`) | retesting in progress |

**Memory tip**: pre-cached weights cut boot time by ~70%. First boot ~25-40 min;
re-boot from cached weights ~10-15 min.

**Architecturally incompatible with our summary use case** (per #1016 Phase 2b):

- R1-family reasoning models (Distill-32B, 0528-Qwen3-8B): emit `<think>` regardless of prompt OR cannot suppress without breaking output
- Kimi-Linear-48B: Constitutional-AI-style refusals + chain-of-thought-in-plain-text behavior

These models work great for OTHER tasks (math, code, multi-step reasoning) — just not for summarization at a fixed token budget.

---

## Round 3 deep-dive — Kimi-Linear + DeepSeek-R1-0528-Qwen3-8B

These two are the operator-prioritised candidates we couldn't break through in
Round 2. Treating them like Nemotron — research-first, then carefully tested.

### DeepSeek-R1-0528-Qwen3-8B

**Status (2026-06-17): DROPPED.** Round 3 attempts at max_length=800/2048/4096
all hit finish_reason=length. The `--reasoning-parser=deepseek_r1` server
flag IS working (no `<think>` blocks in response content) and the no-system-
prompt configuration is correctly applied (per DeepSeek's README). But the
**answer alone is intrinsically 4096+ tokens** — vendor recommends 32k+
output for benchmarks (AIME averaged 23k tokens/question). Same conclusion
as DeepSeek-R1-Distill-Qwen-32B in Phase 2a: R1 family is reasoning-research-
grade, not summary-stage-grade. **DeepSeek will be represented in the final
landscape ONLY by DeepSeek-V2-Lite-Chat** (Round 2 passer, fastest DGX
candidate at 4.8s/ep — a chat model, not reasoning).

**What we now know** (from DeepSeek-R1 GitHub + vLLM docs):

| Finding | Source | Implication |
| --------- | -------- | ------------- |
| "Avoid adding a system prompt; all instructions should be contained within the user prompt." | DeepSeek-R1 README | **Our Round 2 config wraps a system prompt.** This alone may be triggering thinking storm. |
| `--reasoning-parser=deepseek_r1` is a valid vLLM parser flag | vLLM docs | Confirmed officially supported for DeepSeek R1 distills, including the Qwen-based ones (1.5B example shown). |
| The parser separates `<think>` block from final content server-side | vLLM docs | If we wire the flag, the OpenAI API response's `content` field is post-think. We don't need a post-processor at all. |
| `thinking_token_budget` is a sampling parameter | vLLM docs | We can explicitly cap the reasoning budget without changing `max_tokens`. |
| `reasoning_effort` is a Chat Completions param | vLLM docs | `enable_thinking` chat_template_kwarg works AND takes priority over auto-injection. |
| generation_config sampling: temp=0.6, top_p=0.95 | DeepSeek HF card | Vendor-recommended; we already use these. |
| "It is not required to add `<think>\n` at the beginning of the output" | DeepSeek HF card | Thinking is automatic; we should not force it on. |

**Round 3 experiment plan for DSR1-0528-Qwen3-8B**, one variable at a time:

| # | Change | Hypothesis | Config name |
| --- | -------- | ----------- | ------------- |
| 1 | Add `--reasoning-parser=deepseek_r1` to vLLM compose flags + **no system prompt** + max_tokens=800 | Server strips think block AND no system prompt avoids the trigger → clean summary in 800 budget | `..._round3_v1.yaml` |
| 2 | If (1) still overruns 800: bump max_tokens to 4096 with `thinking_token_budget=2048` | Give model room to think AND answer, but cap reasoning explicitly | `..._round3_v2.yaml` |
| 3 | If (2) still wastes budget: try `reasoning_effort="minimal"` on the OpenAI API call | vLLM-side reasoning-effort lever | `..._round3_v3.yaml` |

### Kimi-Linear-48B-A3B-Instruct

**Status (2026-06-17): DROPPED.** Three boot attempts on GB10 unified mem
all failed at the engine init step with `Available KV cache memory: -15
to -17 GiB`:

| Attempt | Compose flags | KV cache result |
| ---: | --- | --- |
| 1 | `--max-model-len=32768 --max-num-seqs=128 --gpu-memory-utilization=0.75` | −16.98 GiB |
| 2 | `--max-model-len=8192 --max-num-seqs=8 --gpu-memory-utilization=0.88` | −15.06 GiB |
| 3 | `--max-model-len=8192 --max-num-seqs=4 --gpu-memory-utilization=0.92 --enforce-eager` | −16.44 GiB |

Variance is noise; the floor is structural. **Moonlight-16B-A3B-Instruct
(same vendor, smaller, already passed Round 1 gate at 8.8s/episode)
represents Moonshot AI in the final landscape.**

Rationale: Kimi-Linear's 91.5 GiB BF16 checkpoint + KDA activation budget
puts the model at the absolute ceiling of GB10's 128 GiB unified memory.
No FP8/NVFP4/quantized variant exists yet (community has open feature
request for FP6 on the GitHub repo). Dropping max-model-len below 8192
would truncate ~30% of our podcast transcripts, breaking apples-to-apples.

**What we now know** (from Moonshot HF card + GitHub + arxiv:2510.26692):

| Finding | Source | Implication |
| --------- | -------- | ------------- |
| Moonshot's only documented system prompt: "You are a helpful assistant provided by Moonshot-AI." | HF card example | Our strict anti-echo system prompt is OUT-OF-DISTRIBUTION for this model. |
| Tech report is architecture-first (KDA attention, KV cache reduction) | arxiv 2510.26692 | Paper does NOT detail RLHF or instruction-tuning pipeline. The Instruct variant's post-training is light/generic. |
| No documented JSON mode, no refusal-pattern documentation, no constitutional-AI mention | HF + GitHub | Our observed refusals ("I'm sorry, but I can't comply...") with the strict prompt are likely a generic safety-tuning artifact, NOT a documented refusal class. |
| `--trust-remote-code` required for `tokenization_kimi.py` | HF card | Already wired in our compose. |
| `--max-model-len 1048576` in vendor's example | HF card | 1M context window. We don't need that for summary — keep `--max-model-len=32768` for KV cache budget. |
| Only example task: simple Q&A ("Is 123 a prime?") | HF card | The model is documented as a generic assistant, not as a task-specialised summarizer. |

**Why our Round 2 attempts failed in opposite directions:**

- **Strict prompt** ("don't echo, don't copy, don't think"): negative constraints
  trigger the safety circuit (refusal). Likely the model's RLHF saw this
  prompt class as adversarial.
- **Natural prompt** ("write a clear summary"): positive framing without
  in-context calibration; model produces chain-of-thought-in-prose ("First,
  I'll read the transcript carefully. Then I'll identify the main themes...").
  The model is narrating the task because the Instruct training didn't give
  it a strong "execute, don't explain" prior.

**Round 3 experiment plan for Kimi-Linear**, one variable at a time:

| # | Change | Hypothesis | Config name |
| --- | -------- | ----------- | ------------- |
| 1 | NO system prompt + minimal user prompt: `"Summarize this transcript:\n\n{transcript}\n\nSummary:"` | Generic assistant model + completion-style framing → completes the "Summary:" continuation | `..._round3_v1_nosys.yaml` |
| 2 | NO system prompt + 1-shot in-context example (silver from `curated_5feeds_dev_v1`) | In-context calibration is more reliable than instruction-tuning for sparsely-tuned models | `..._round3_v2_oneshot.yaml` |
| 3 | NO system prompt + JSON mode: `"Summarize this. Reply ONLY with JSON: {\"summary\": \"...\"}"` + `response_format={"type": "json_object"}` | Forces structured output; avoids both refusal and narration patterns | `..._round3_v3_json.yaml` |
| 4 | Moonshot's own minimal system prompt ("You are a helpful assistant provided by Moonshot-AI.") + minimal user prompt | If model is sensitive to vendor system prompt, this should be the lowest-resistance path | `..._round3_v4_vendorsys.yaml` |

### Common harness levers to add for both models

1. **Make `--reasoning-parser` a compose-level config knob** (not hard-coded
   for autoresearch). Round 3 needs it for DSR1; future Qwen3.5 retry needs
   `qwen3` parser per HF docs.
2. **Make `enable_thinking` chat_template_kwarg a per-config param** (not
   buried under `extra_body`). DSR1, Qwen3, Gemma 4 all use it.
3. **Make `response_format` configurable per run** for JSON-mode experiments.
4. **Update `onboard_model_smoke.py`** to detect chain-of-thought-in-prose
   patterns ("First, I'll", "Then I'll", "Let me start by"). Currently
   detects refusal + raw `<think>` + BPE artifacts; CoT-in-prose is the
   Kimi-specific failure mode that needs a check.

---

## Round 3 finding — vendor-sampling AMPLIFIES Sonnet-mimicry detection

**Methodology**: every Round 3 candidate is scored against TWO silvers (Opus
4.7 + Sonnet 4.6). The signal is `Δ = (Sonnet score) − (Opus score)` on the
same metric. Positive Δ = model's training distribution leans Sonnet-style;
~0 = style-neutral; negative = Opus-style.

**Round 3 result table** (ROUGE-1, partial cohort, 2026-06-17):

| Candidate | ROUGE-1 vs Opus | ROUGE-1 vs Sonnet | Δ (S−O) | Verdict |
| ----------- | ----------------: | ------------------: | --------: | --------- |
| Qwen3-30B-A3B-Instruct-2507 | 50.1% | 54.5% | **+4.4** | clear Sonnet lean |
| Gemma-4-26B-A4B-it | 51.8% | 56.3% | **+4.5** | clear Sonnet lean |
| Qwen3.5-35B-A3B | 59.4% | 63.0% | **+3.6** | clear Sonnet lean |
| Moonlight-16B-A3B-Instruct | 57.5% | 57.5% | 0.0 | style-neutral |
| Llama-3.3-70B-Instruct-NVFP4 | 48.7% | 49.3% | +0.6 | neutral (noise) |
| DeepSeek-V2-Lite-Chat | 39.2% | 39.9% | +0.7 | neutral (noise) |

**Critical methodology note**:

The original Round 1/2 dual-silver pass (greedy temp=0.0) showed Qwen3.5-35B-A3B
at Δ ≈ +0.002 on ROUGE-L — essentially style-neutral. **Round 3 with vendor-
correct sampling + the `--reasoning-parser=qwen3` server flag revealed the
lean clearly** (Δ = +2.3 on ROUGE-L, +3.6 on ROUGE-1). Implication: **greedy
decoding masks style-mimicry** by collapsing diverse outputs toward neutral
midpoints. The vendor-correct sampling distribution is the one prod would
actually use → it's the one that surfaces the lean.

**Practical implication for [[silver_judge_vendor_bias]]**: when building a
cross-vendor candidate cohort, always run Round 3 (or vendor-correct sampling
equivalent) BEFORE drawing per-stage routing conclusions. Round 1 greedy
numbers can hide a vendor-lean that materially changes which silver/judge
combination is appropriate.

---

## Phase 2b Round 3 plan — vendor-correct sampling re-run

Treating the #1016 Round 1/2 cohort data as a **first-pass baseline** that
used uniform `temperature=0.0`. Round 3 re-runs Phase 2b summary with each
model at its vendor-recommended sampling, plus any missed flags / chat-template
levers discovered in the documentation pass.

### Per-model Round 3 deltas (what changes from Round 2 → Round 3)

| Model | Round 1/2 config | Round 3 changes |
| ------- | ------------------ | ----------------- |
| **Qwen3-30B-A3B-Instruct-2507** | temp=0.0, `enable_thinking: false` | **temp=0.7** (vendor recommends 0.7 for instruct mode) |
| **Qwen3.5-35B-A3B** | temp=0.0, `enable_thinking: false` | **temp=0.7, top_p=0.95** + add `--reasoning-parser=qwen3` server-side flag (we missed it) |
| **Mistral-Small-3.2-24B-Instruct-2506** | temp=0.0 | **temp=0.15** (vendor's recommendation is much lower than typical — calibrated for assistant tasks) |
| **Mistral-Small-4-119B-NVFP4** | temp=0.0, OOM at default seqs | **temp=0.7** + `--max-num-seqs=4` + `--max-model-len=8192` (DGX-Spark blog memory fix) |
| **Magistral-Small-2509** | temp=0.0, reasoning-parser=mistral | **temp=0.7, top_p=0.95** |
| **Ministral-3-14B-Instruct-2512** | temp=0.0 | leave temp=0.0 (vendor doesn't specify); already passed gate |
| **Gemma-4-26B-A4B-it** | temp=0.0 | **temp=1.0, top_p=0.95, top_k=64** (vendor wants high temp — our greedy may be flattening output) + add `enable_thinking` chat_template_kwarg (vendor doc mentions it) |
| **Llama-3.3-70B-Instruct-NVFP4** | temp=0.0, strict prompt | **temp=0.6, top_p=0.9** + keep strict prompt |
| **Moonlight-16B-A3B-Instruct** | temp=0.0 | leave temp=0.0 (vendor doesn't specify); already fastest in cohort |
| **Kimi-Linear-48B-A3B-Instruct** | strict prompt → refusals; natural prompt → chain-of-thought-in-prose | **Try**: (a) no system prompt at all + minimal user instruction, (b) JSON-structured output `{"summary": "..."}` format, (c) few-shot example (1 in-context summary), (d) higher max_tokens=2048 |
| **DeepSeek-V2-Lite-Chat** | strict prompt + decode_r1_byte_level | **temp=0.3, top_p=0.95, do_sample=True** (verbatim from `generation_config.json`!) — currently we run greedy which contradicts vendor defaults |
| **DeepSeek-R1-Distill-Qwen-32B** | temp=0.6, top_p=0.95, anti-think prompt | leave as-is (already speed-disqualified at 204s/ep — no value in re-running) |
| **DeepSeek-R1-0528-Qwen3-8B** | temp=0.6, top_p=0.95, anti-think prompt + `enable_thinking: false` | **Try**: (a) `--reasoning-parser=deepseek_r1` flag (already declared but verify server strips), (b) increase max_tokens to **4096** so model has room to think AND produce summary, (c) keep r1_distill anti-think prompt |
| **Nemotron-Super-49B-v1_5-FP8** | tried "detailed thinking off" → fail; `/no_think` working as of Round 2 | leave temp=0.0 (vendor's reasoning-OFF mode recommends greedy ✓ matches); confirm /no_think run lands ≥7/10 successful summaries |

### Round 3 configuration discipline

To keep the comparison clean across this re-run:

1. **One change per model per round.** If `temp=0.7` and a new prompt are
   both needed, separate the experiments (otherwise we won't know which lever
   moved the score).
2. **Re-use the existing dataset** (`curated_5feeds_dev_v1`) and silvers
   (Opus 4.7 + Sonnet 4.6 paragraph) so results are directly comparable to
   the Round 1/2 scorecard.
3. **Same judge panel** (Sonnet 4.6 primary + GPT-5.4 cross-check) so the
   G-Eval delta isolates the sampling/flag change, not judging.
4. **Apply the Preliminary Result Gate** (`JUDGING.md` §) per cell. Gate
   thresholds unchanged from Round 1/2.
5. **Onboarding smoke test FIRST** (`scripts/eval/onboard_model_smoke.py`)
   before queuing a Round 3 cohort run for any candidate. We'll catch dumb
   mistakes (wrong prompt, missing postprocessor) before burning 10
   episodes' inference + judge cost.

### Round 3 ordering — execute by impact

**Updated 2026-06-17 after deep-dive of 7 cohort-model HF cards**:

Priority 1 — **structural gaps** (not just sampling) that may have invalidated
Round 1/2 results entirely. These need re-run before drawing any conclusions:

- **Qwen3.5-35B-A3B (CURRENT COHORT WINNER, treat with skepticism)**:
  - Vendor: thinking-by-default model. Disable via `chat_template_kwargs.enable_thinking=False`.
  - Vendor: must run with `--reasoning-parser=qwen3` server flag.
  - Vendor instruct-mode sampling (general tasks): `temp=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
  - **Round 1/2 ran**: greedy temp=0.0, no top_k, no parser flag, with `enable_thinking=False` set in `extra_body`.
  - **Risk**: without parser flag, model may have leaked partial think content into the response. Without proper sampling, model may have produced sub-quality outputs that just happened to be the cohort top.
  - **Round 3 must**: wire parser flag at compose swap-in; switch sampling to vendor instruct-general defaults.

- **Magistral-Small-2509 (REASONING MODEL, not summarizer)**:
  - Vendor: explicit reasoning model with `[THINK]/[/THINK]` special tokens.
  - Vendor: system prompt EMBEDS the reasoning template ("First draft your thinking process… [THINK]…[/THINK]Here, provide a self-contained response.").
  - Vendor: requires `--reasoning-parser=mistral` server flag.
  - Vendor sampling: `temp=0.7, top_p=0.95, max_tokens=131072`.
  - **Round 1/2 ran**: as if it were a vanilla summarizer.
  - **Risk**: similar failure mode to R1 (think block fills budget). May explain why Magistral underperformed Mistral-Small-3.2 on KG.
  - **Round 3 must**: either (a) use the model AS a reasoning model with max_tokens=4096+ AND --reasoning-parser=mistral (then strip the THINK block from response), or (b) drop it from the cohort since its design is not summary-stage-fit.

- **Mistral-Small-3.2-24B & Ministral-3-14B (mistral-common vs HF tokenizer)**:
  - Vendor: BOTH require `--tokenizer_mode mistral --config_format mistral --load_format mistral`.
  - Vendor: BOTH have a `SYSTEM_PROMPT.txt` in their repo that should be loaded into the system prompt slot.
  - Mistral-Small-3.2 vendor sampling: `temp=0.15` ("relatively low temperature").
  - Ministral-3-14B vendor sampling: `temperature BELOW 0.1` ("daily-driver and production"). Even more aggressive.
  - **Round 1/2 ran**: HF tokenizer (no mistral flags), no SYSTEM_PROMPT.txt, greedy temp=0.0.
  - **Risk**: subtle tokenization differences between mistral-common and HF can cause text quality drift; lack of SYSTEM_PROMPT.txt strips the assistant priming the vendor expects.
  - **Round 3 must**: wire the 3 mistral flags at compose swap-in; load SYSTEM_PROMPT.txt; switch sampling to vendor temps.

Priority 2 — **sampling-only deltas** (structurally we had it right, sampling was wrong):

- **Gemma-4-26B-A4B-it**: vendor wants `temp=1.0, top_p=0.95, top_k=64`. `enable_thinking=False` is the DEFAULT (we don't need to disable explicitly). We ran greedy.
- **Qwen3-30B-A3B-Instruct-2507**: vendor wants `temp=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`. The 2507 variant supports ONLY non-thinking mode (no parser flag, no enable_thinking lever). We ran greedy with redundant enable_thinking=False.
- **Llama-3.3-70B-Instruct-NVFP4**: vendor wants `temp=0.6, top_p=0.9` (confirmed via RedHatAI quantized variant). We ran greedy + strict prompt; sampling change may free up more headroom.
- **DSV2-Lite**: vendor wants `temp=0.3` (vLLM example) per generation_config.json. We ran greedy + strict prompt + decoder.

Priority 3 — **likely unchanged but worth confirming**:

- **Moonlight-16B-A3B-Instruct**: no vendor sampling documented; we used greedy. Architecture = DeepSeek-V3 (MoE). Context 8K confirmed.
- **Mistral-Small-4-119B-NVFP4**: not retried; OOM was a compose-tuning issue, not a sampling issue.

Priority 4 — **drop from cohort**:

- **DeepSeek-R1-Distill-32B**: speed-disqualified at 204s/ep in Round 1.

### What the 14-model HF doc pass reveals about our harness

| Gap | Affects | Severity |
| ----- | --------- | ---------- |
| `--reasoning-parser=qwen3` never wired for Qwen3.5 series | Qwen3.5-35B-A3B (cohort winner!) | **CRITICAL** |
| `--reasoning-parser=mistral` never wired for Magistral | Magistral-Small-2509 | **CRITICAL** |
| `--tokenizer_mode mistral --config_format mistral --load_format mistral` flags missing | Mistral-Small-3.2, Ministral-3-14B, Mistral-Small-4-119B-NVFP4, Magistral | **HIGH** |
| SYSTEM_PROMPT.txt not loaded for Mistral family | All Mistral variants | **HIGH** |
| `top_p` field missing in SummarizationParams → silently dropped from `params:` block | ALL Round 1/2 configs that set top_p (Magistral, R1-Distill, Qwen3-30B-Instruct, R1-Distill-32B, R1-0528-Qwen3-8B) | **HIGH** — see task #108 |
| `top_k`, `presence_penalty`, `min_p` fields missing in SummarizationParams | Qwen3 / Qwen3.5 family (vendor specifies all 4) | **HIGH** |
| `--reasoning-parser=deepseek_r1` never wired for DSR1-0528-Qwen3-8B | DSR1-0528-Qwen3-8B | **HIGH** |
| `enable_thinking` chat_template_kwarg via `extra_body` partly works but inconsistent across configs | Qwen3.5, Gemma 4 (different defaults) | **MEDIUM** |
| `--max-num-batched-tokens=4096` ad-hoc-added for Gemma 4 (no standardized per-model config) | Gemma 4 (multimodal items) | **MEDIUM** |
| Compose flags hard-coded in homelab repo, not config-driven per-model | Every model that needs a different flag set | **CRITICAL ARCH GAP** |

### Required harness work BEFORE Round 3 cohort run

1. Extend `SummarizationParams` (src/podcast_scraper/providers/params.py) with
   `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`. Plumb
   through openai_provider chat-completions kwargs. (Task #108)
2. Plumb `response_format` extra_body through the OpenAI provider (Kimi JSON
   mode + future structured-output experiments).
3. Document per-model compose flag set as a section in MODEL_PLAYBOOK.md and
   accept that some Round 3 cells will need a compose-swap dance (parser flag
   plus model swap) — not just a model swap.
4. Update `onboard_model_smoke.py` to detect chain-of-thought-in-prose patterns
   (Kimi failure mode) AND `<think>` leakage (Qwen3.5 + Gemma 4 + Magistral
   failure mode if parser is missing).

### Round 3 must-haves vs nice-to-haves

**Must-haves** (block Round 3):

- Parser flag wiring (Qwen3.5, Magistral, DSR1)
- Mistral tokenizer mode + SYSTEM_PROMPT.txt for Mistral family
- top_p/top_k/presence_penalty in SummarizationParams

**Nice-to-haves** (can ship Round 3 without):

- Smoke harness CoT detection
- Per-model compose-flag config layer (Round 3 can hand-edit compose like
  we did for Nemotron)

Priority 2 — vendor-correct sampling that probably won't move much but worth documenting:

- Qwen3-30B-A3B-Instruct-2507 (temp=0.7)
- Magistral-Small-2509 (temp=0.7, top_p=0.95)
- DeepSeek-V2-Lite-Chat (temp=0.3 from generation_config.json)

Priority 3 — broken-on-Round-2, needs harness work first:

- Mistral-Small-4-119B-NVFP4 (compose tuning before any inference attempt)
- Kimi-Linear-48B-A3B-Instruct (still investigating; needs novel prompt approach)
- DeepSeek-R1-0528-Qwen3-8B (server-side reasoning-parser + max_tokens=4096)

Priority 4 — already passed and unlikely to change:

- Moonlight-16B (fastest)
- Ministral-3-14B (Mistral defaults)
- Nemotron-Super-49B (`/no_think` working)

### What the data we just gathered means for harness defaults

From the doc pass:

- **`enable_thinking` chat_template kwarg** is supported by Qwen3, Gemma 4
  (and other Qwen-derived models). Not just Qwen-specific — it's becoming a
  cross-vendor convention. Worth making this a first-class harness param,
  not just a per-model `extra_body` override.
- **`generation_config.json`** carries the vendor's recommended sampling.
  Worth a one-time script that reads this from each HF model card and writes
  it into the playbook + per-model config. Currently we hard-code temp/top_p
  in our YAML configs.
- **DeepSeek-V2-Lite vendor sampling is `temperature=0.3, top_p=0.95,
  do_sample=True`** — we ran it with greedy decoding which contradicts vendor
  defaults. That alone may have caused the transcript-echoing issue in Round 1.
- **Mistral-Small-3.2's temp=0.15 recommendation** is unusual — most models
  recommend 0.6-1.0. The 0.15 reflects a deliberately low-randomness assistant
  tuning. Worth respecting.

---

## Vendor-recommended sampling (per HF model card, 2026-06-17 pass)

We ran the #1016 cohort with `temperature=0.0` (greedy) uniformly. Vendor docs
disagree — most recommend non-greedy sampling. Worth re-running with
vendor-correct sampling for the candidates whose quality looked borderline.

| Model | Vendor temp | Vendor top_p | Notes |
| ------- | ------------: | -------------: | ------- |
| Qwen3-30B-A3B-Instruct-2507 | 0.7 | (default) | + `enable_thinking: false` for non-reasoning mode |
| Qwen3.5-35B-A3B | 1.0 / 0.6 / 0.7 (split) | 0.95 | + `--reasoning-parser=qwen3` flag (we did NOT use this) |
| Mistral-Small-3.2-24B-Instruct-2506 | **0.15** | (default) | Unusually low — likely calibrated for assistant tasks |
| Mistral-Small-4-119B-NVFP4 | 0.7 | (default) | |
| Magistral-Small-2509 | 0.7 | 0.95 | |
| Ministral-3-14B-Instruct-2512 | (default) | (default) | Card doesn't specify |
| **Gemma-4-26B-A4B-it** | **1.0** | 0.95 | top_k=64 also recommended; supports `enable_thinking` chat_template kwarg (we did NOT use) |
| Llama-3.3-70B-Instruct-NVFP4 (RedHat repack) | 0.6 | 0.9 | |
| Moonlight-16B-A3B-Instruct | (default) | (default) | trust_remote_code required |
| Kimi-Linear-48B-A3B-Instruct | (default) | (default) | trust_remote_code required |
| DeepSeek-V2-Lite-Chat | 0.3 (chat) or 0.85 (creative) | (default) | trust_remote_code required |
| DeepSeek-R1-Distill-Qwen-32B | 0.6 | 0.95 | R1 family convention |
| DeepSeek-R1-0528-Qwen3-8B | 0.6 | 0.95 | R1 family convention |
| Nemotron-Super-49B-v1_5-FP8 | 0.6 (reasoning ON) / **greedy** (OFF) | 0.95 (ON) | matches our `/no_think` mode usage ✓ |

**Read**:

- Our `temp=0.0` was correct for Nemotron OFF mode and DeepSeek-V2-Lite (chat).
- It was WRONG for Qwen3.5 (vendor wants 0.6-1.0), Gemma 4 (wants 1.0), Llama 3.3 (wants 0.6), most Mistrals (want 0.7), DeepSeek R1 family (wants 0.6).
- **Re-running with vendor temp could change quality scores meaningfully.** Especially for Gemma 4 — using temp=0.0 on a model calibrated for temp=1.0 may produce flat / repetitive output.

**Per-model playbook entry updated below to reflect vendor-correct sampling**.

---

## Harness implications (cross-cutting lessons from #1016)

Each per-model finding tonight has implications beyond that single model.
Capture them here so the harness/compose defaults can evolve toward "works for
most models out of the box, with named overrides per model".

### Prompt strategy is per-model — make it a config knob

| Finding | Implication |
| --------- | ------------- |
| Strict negative-constraint stack triggers refusals on Kimi-Linear | Don't make "strict" the default. Pick per-model. |
| Standard summary prompts cause Llama-3.3-70B verbosity overflow | "Default" prompt is biased toward Qwen — flag this |
| Natural prompts cause Kimi-Linear to narrate the task | Even "neutral" prompts misfire on certain models |
| DeepSeek-V2-Lite needs anti-echo guard or it transcribes verbatim | Some models need stronger prompts than others |

**Harness change**: prompts directory has 3 named families now:
`ollama/qwen3.5_35b/summarization/*` (default, Qwen-style), `vllm/strict_summarization/*`
(anti-echo, anti-verbosity), `vllm/natural_summarization/*` (positive-only, no
negative constraints). Each model section in this playbook names which family
to use. The smoke harness validates the choice before cohort run.

### Reasoning control mechanism varies — make it explicit per model

| Vendor | Mechanism | What we tried |
| -------- | ----------- | ---------------- |
| Qwen3 family | `extra_body.chat_template_kwargs.enable_thinking: false` | works on Qwen3, doesn't transfer to R1-distilled-from-Qwen3 |
| Nemotron | system prompt LITERAL `/no_think` (with slash) | NOT "detailed thinking off" — that's Qwen3's convention. Wrong-convention bug cost ~1 hour tonight. |
| DeepSeek R1 family | `--reasoning-parser=deepseek_r1` vLLM flag (strips `<think>` server-side) | Strips output tags but doesn't prevent the model from spending the budget on `<think>` content. R1 always thinks; no off switch. |
| Mistral / Magistral | `--reasoning-parser=mistral` + tokenizer/config-format flags | Works for the Magistral 24B variant |
| Nemotron `--reasoning-parser=nemotron_v3` | **BROKEN on Nemotron-Super-49B-v1_5-FP8** — the parser expects think tokens missing from this checkpoint's tokenizer vocab. Boot loop. | Use system prompt convention instead. |

**Harness change**: each model's playbook section now specifies exactly which
mechanism applies. The smoke harness validates by checking for `<think>` leaks
in output BEFORE declaring the model ready.

### Byte-level BPE artifacts — postprocessor as a safety net

vLLM 26.05-py3 has a tokenizer-decode gap on some checkpoints (R1-Distill,
DeepSeek-V2-Lite, possibly others). Raw `Ġ` / `Ċ` / `âĢĻ` tokens leak into
the API response.

**Harness change**:

- Postprocessor registry in `output_postprocess.py` has `decode_r1_byte_level`,
  `strip_r1_reasoning`, `strip_r1_reasoning_and_decode`, `noop`.
- Per-model playbook entry names which to apply.
- Smoke harness CHECKS for these artifacts in the response — refuses to mark
  the model "ready" if they appear without a postprocessor declared.

### `finish_reason=length` is signal, not a hard error

Tonight several candidates hit the guardrail and crashed the run on the first
episode that exceeded their natural output length:

- Llama-3.3-70B with default prompt
- DeepSeek-V2-Lite (untuned)
- Kimi-Linear (preamble + summary exceeded 800 tokens)

**Harness change**: a length-failure is a SIGNAL — it tells us we need a
stricter prompt or a higher token budget. The Preliminary Result Gate (in
JUDGING.md) catches this empirically, but the smoke harness should fail-fast
on a single-episode length overflow so we don't waste 10 episodes' time before
discovering the problem.

### Refusal detection — flag the unique signal

Kimi-Linear was the only model in the 14-candidate cohort to refuse a request
("I'm sorry, but I can't comply...", "Stop."). The signature is unique
enough that we can detect it programmatically.

**Harness change**: the smoke harness has a `REFUSAL_PHRASES` list and flags
refusal patterns distinctly from other failures. Future onboarding sessions
will surface "this model refused N times" as a separate signal rather than
hiding it under "guardrail failure".

### Memory budget needs per-model `gpu-memory-utilization`

GB10's 128 GB unified pool means weight size dictates the util budget for
each model. Tonight we observed:

| Model | Disk size | Required UTIL | Headroom for KV |
| ------- | ----------: | --------------: | ----------------- |
| Moonlight-16B | 31.9 GB | 0.55 (~70 GB budget) | ~38 GB |
| Mistral-Small-3.2-24B | 48 GB | 0.55 | ~22 GB — tight but OK |
| Qwen3.5-35B-A3B | 67 GB | **0.65** (0.55-0.60 OOMs) | ~16 GB |
| Mistral-Small-4-119B-NVFP4 | 70.8 GB | 0.80 + `--max-num-seqs=4` required | KV alone wouldn't fit at default seqs |
| Kimi-Linear-48B | 91.65 GB | 0.80 (very tight) | ~10 GB |

**Harness change**: the playbook quick-reference table has the recommended
UTIL per model. The swap-script should read this and update the `.env` file
automatically rather than relying on operator memory. (#1022 follow-up: write
`apply_model_overrides.py` that reads the playbook table and patches both the
compose model line AND the VLLM_GPU_MEM_UTIL.)

### Compose flag layering — separate boot-time from run-time

Some flags must live in compose (set at container boot):

- `--trust-remote-code` (some custom tokenizers)
- `--reasoning-parser=X` (server-side stripping)
- `--max-model-len=N` (architectural constraint)
- `--max-num-batched-tokens=N` (KV cache calc input)
- `--max-num-seqs=N` (KV cache calc input)

Others should live in the experiment config (set per request):

- `temperature` / `top_p` (sampling)
- `max_tokens` (length cap)
- `extra_body.chat_template_kwargs.*` (model-specific runtime params like Qwen3's `enable_thinking`)
- prompt convention (system / user templates)

**Harness change**: the playbook makes this distinction explicit in each
model entry. The smoke harness validates the FULL stack — boot-time + runtime
plus prompt — before declaring the model ready.

### Onboarding cost — budget it explicitly

A new model costs:

- 5-15 min reading the model card + vendor docs
- 5-15 min compose flag negotiation
- 1-3 boot attempts (each 5-25 min) before flags are right
- 1-3 smoke attempts before prompt is right
- **Total**: 30-90 min of dedicated onboarding before the model can join an autoresearch cohort

**Harness change**: this is a real cost line for any "let's add model X" ask.
The playbook captures the work so it doesn't need to be repeated. Future
onboarding sessions read the playbook first and skip the steps that are
already documented for that vendor family.

---

## DeepSeek

### DeepSeek-V2-Lite-Chat (16B / 2.4B-active MoE, mid-2024) — **DROPPED 2026-06-22**

**Status**: DROPPED from the autoresearch cohort. Eval YAMLs deleted in
the same commit. DeepSeek dropped as a vendor at <35B (no other DeepSeek
architecture viable in that size class — V2 family is the only pure-DeepSeek
sub-35B arch; R1-Distill-* models are Qwen/Llama with DeepSeek-style
post-training, not DeepSeek architecture).

**Why dropped — cohort-comparison evidence (2026-06-22)**:

Two re-runs done on DGX (vllm:26.05-py3) against `curated_5feeds_dev_v1`:

1. greedy + `EVAL_VLLM_GUIDED_JSON=1` (xgrammar shape constraint)
2. vendor sampling per HF card (temp=0.7) + no guided_json

Both produced essentially identical structural failure: 1 mega-topic per
episode (the model writes the episode summary into the Topic.label slot)
and 0 typed Persons / 0 typed Organizations. KG vs Opus-4.7 silver: 2%
weighted coverage in both runs. GI: 1% in both. Cohort-wide tally on the
same 10 dev episodes, same prompt, same harness:

| Candidate | Total Topics | Persons | Orgs |
| --- | ---: | ---: | ---: |
| Ministral-3-14B (dense) | 200 | 20 | 10 |
| Mistral-Small-3.2-24B | 175 | 20 | 10 |
| Qwen3-30B-A3B-Instruct | 149 | 20 | 10 |
| Magistral-Small-2509 | 121 | 20 | 10 |
| Moonlight-16B-A3B (Kimi, **same architecture class**) | 114 | 20 | 5 |
| Qwen3.5-35B-A3B | 105 | 20 | 10 |
| Gemma-4-26B-A4B | 88 | 20 | 10 |
| **DSV2-Lite-Chat** | **10** | **0** | **0** |

Moonlight-16B-A3B (Kimi family, same 16B / 3B-active MoE class) produces
114 typed topics + 20 named persons on the same harness. The failure is
not architecture, not size, not sampling — it's DSV2-Lite's mid-2024
training, which predates the structured-output / tool-use RLHF wave the
rest of the cohort benefits from. The model **names** orgs (`Singletrack
Sessions`, `Practical Systems`, `Long Horizon Notes`) IN the prose blob
but cannot emit them as separate structured nodes.

**Historical context kept** (for the record, not the cohort):

- Round 2 round: with strict prompt + `decode_r1_byte_level` postprocessor,
  summary path scored 4.8s/episode mean, 983 char output — cohort speed
  leader at the time.
- Round 3 v1 (2026-06-17): summary gate PASS (vs Opus-4.7 ROUGE-1=39.2%,
  cos=72.5%). The summary path WAS the speed-quality floor candidate.
- Phase 2c (2026-06-17): GI/KG 0% INVALID — initially blamed on harness
  bug #111 (postprocessor not applied to node.label). Retro audit
  2026-06-22 confirmed #111 was fixed pre-chunk-7 in commit `0295e617`;
  the residual KG/GI floor was the model's structured-extraction ceiling,
  not the postprocessor.

**Replacement?** No <35B DeepSeek architecture fits. R1-Distill family
(Qwen-32B, Qwen3-8B) is speed-disqualified by reasoning-grade token
budgets (see sections below). DeepSeek-Coder-V2-Lite-Instruct is the only
remaining untested same-arch variant; not tested because the operator
decided 2026-06-22 to drop DeepSeek vendor representation at this tier
rather than continue searching. Re-introduce if/when DeepSeek ships a
V3-Lite or chat-tuned <35B model.

- Cite: `https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat`

### DeepSeek-R1-Distill-Qwen-32B (Jan 2025, reasoning)

- HF: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` — 65 GB BF16
- vLLM flags: `--reasoning-parser=deepseek_r1`
- Sampling: `temperature=0.6`, `top_p=0.95` (R1 family convention)
- Prompt convention: anti-thinking prompts in `vllm/r1_distill_32b/summarization/*`
- **Postprocessor REQUIRED**: `strip_r1_reasoning_and_decode` — strips `<think>`
  blocks AND decodes byte-level BPE artifacts
- Known failure modes:
  - 204s mean inference per episode (#1016 Phase 2b speed-disqualified)
  - Always emits `<think>` block regardless of anti-thinking prompts —
    architectural property of R1 distillation, can't be disabled via prompt

### DeepSeek-R1-0528-Qwen3-8B (May 2025, reasoning, newer Qwen3 base)

- HF: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` — 16.4 GB BF16
- vLLM flags: `--reasoning-parser=deepseek_r1` (only partial — see failure modes)
- Sampling: `temperature=0.6`, `top_p=0.95`
- Prompt convention: tried anti-thinking + `enable_thinking: false` extra_body
  (Qwen3 base lever) — neither stops the `<think>` block
- Known failure modes:
  - The R1-distillation overwrites the Qwen3 base's thinking control. Even with
    `extra_body.chat_template_kwargs.enable_thinking: false` the model still
    emits `<think>` and consumes full token budget on reasoning before reaching
    the summary
  - Currently considered architecturally incompatible with our summarization use
    case (needs further investigation per operator direction)

---

## Mistral

### Mistral-Small-3.2-24B-Instruct-2506

- **Round 3 v1 result** (2026-06-17, vendor sampling temp=0.15 + 4 mistral
  flags `--tokenizer_mode=mistral --config_format=mistral --load_format=mistral
  --tool-call-parser=mistral`): gate **PASS** (mean=69.7s, p99=83.9s,
  cv=0.12, chars 1661-2510). vs Opus 4.7: ROUGE-1=55.6%, ROUGE-L=26.1%,
  cos=80.9%, BLEU=15.3%. vs Sonnet 4.6: ROUGE-1=56.8% (Δ=+1.2 noise-level),
  BLEU=15.2%.
- **Cohort role**: solid mid-pack quality, **style-neutral** (no
  Sonnet-mimicry — Mistral's French training data appears resistant to
  Anthropic-style capture, valuable for cross-vendor judging trust).
  **Slowed 2.3× vs Round 1** (69.7s vs ~30s) due to mistral_common
  tokenizer + the 4 flags — quality preserved but speed sacrificed.
  Reasonable candidate for budgeted quality-stable scenarios.

### Mistral-Small-3.2-24B-Instruct-2506 — original notes (pre-Round 3)

- HF: `mistralai/Mistral-Small-3.2-24B-Instruct-2506` — ~48 GB BF16
- vLLM flags: none (`VLLM_GPU_MEM_UTIL=0.55` works)
- Sampling: `temperature=0.0`
- Prompt convention: `ollama/qwen3.5_35b/summarization/*` family works cleanly
- Known failure modes: none — clean cohort participant

### Mistral-Small-4-119B-2603-NVFP4 (Jan 2026, 128 experts / 4 active = ~6.5B active)

- HF: `mistralai/Mistral-Small-4-119B-2603-NVFP4` — 70.8 GB NVFP4
- vLLM flags: `--max-num-batched-tokens=4096` (default 2048 → KV cache too small),
  `--reasoning-parser=mistral`, `--tokenizer-mode=mistral`,
  `--config-format=mistral`, `--tool-call-parser=mistral` (Mistral family
  convention)
- Sampling: `temperature=0.0`
- Memory: weights 67 GB + KV cache at default `max-num-seqs=256` and
  `max-model-len=32768` = ~100-105 GB → OOM on 128 GB GB10 pool at util=0.80
- **Required fix per DGX-Spark blog**: `--max-num-seqs=4` (we're single-stream) +
  `--max-model-len=8192` (our transcripts are ~3K tokens). KV cache budget then
  drops 30× → comfortable fit.
- Known failure modes:
  - At default `max-num-seqs=256`: OOM on first inference (weights fit, KV cache
    won't allocate)
- Cite: DGX-Spark blog <https://vllm.ai/blog/2026-06-01-vllm-dgx-spark>
  recommends `max-num-seqs=4` for single-stream on GB10

### Magistral-Small-2509 (Mistral Small 1.2 derived, partial reasoning)

- **Round 3 v1 result** (2026-06-17, vendor sampling temp=0.7/top_p=0.95 via
  extra_body, `--reasoning-parser=mistral` server flag wired + 4 mistral
  tokenizer flags, max_length=4096 to give reasoning room): gate **PASS**
  (mean=64.1s, p99=70.8s, cv=0.10, chars 1384-2033). vs Opus 4.7:
  ROUGE-1=55.2%, ROUGE-L=28.3%, cos=80.9%, BLEU=14.5%. vs Sonnet 4.6:
  ROUGE-1=56.8% (Δ=+1.6 noise-level), BLEU=14.3%.
- **Cohort role**: reasoning-model that handles summary cleanly when the
  parser flag is wired. Quality identical to Round 1 (55.2% Opus matches
  exactly) — parser + vendor sampling did not shift quality, validates the
  fix didn't break anything. Style-neutral. Bottom-third speed (64.1s),
  hard to justify vs faster Qwen3.5-35B-A3B at higher quality.

### Magistral-Small-2509 — original notes (pre-Round 3)

- HF: `mistralai/Magistral-Small-2509` — ~48 GB BF16
- vLLM flags: `--reasoning-parser=mistral`, `--tokenizer-mode=mistral`,
  `--config-format=mistral`, `--tool-call-parser=mistral`
- Sampling: `temperature=0.0`
- Prompt convention: `ollama/qwen3.5_35b/summarization/*` family works
- Known failure modes: none structural; just slower than Qwen MoE A3B (dense 24B)

### Ministral-3-14B-Instruct-2512 (Oct 2025, FP8 native)

- **Round 3 v1 result** (2026-06-17, vendor sampling temp=0.05 ("below 0.1"
  per vendor) + 4 mistral tokenizer flags): gate **PASS** (mean=30.2s,
  p99=35.5s, cv=0.11, chars 2100-3136 mean=2596). vs Opus 4.7: ROUGE-1=53.3%,
  ROUGE-L=22.7%, cos=81.2%, BLEU=10.7%. vs Sonnet 4.6: ROUGE-1=55.4%
  (Δ=+2.1 borderline noise), BLEU=11.9%.
- **Cohort role**: dense-FP8 candidate at the small-model end (14B). Mid-pack
  quality, mid speed. Style-leaning-Sonnet but Δ is borderline noise (+2.1
  vs Qwen3.5's +3.6 or Gemma 4's +4.5). Reasonable choice when 14B dense
  fits budget constraints (smaller KV cache than MoE A3B family).
- **Phase 2c result** (2026-06-17): GI vs Opus 30%, vs Sonnet 38%
  (Δ = **+8 — LARGEST GI Sonnet-mimicry in cohort**). KG vs Opus 32% (R1/2
  32%; no change). Summary borderline-neutral (+2.1) but GI heavily
  Sonnet-leaning. Reinforces "**style is task-dependent**" finding from
  Qwen3-30B-Instruct.

### Ministral-3-14B-Instruct-2512 — original notes (pre-Round 3)

- HF: `mistralai/Ministral-3-14B-Instruct-2512` — 31.5 GB FP8
- vLLM flags: none non-default
- Sampling: `temperature=0.0`
- Prompt convention: standard summarization prompts work
- Known failure modes: none

---

## Google

### Gemma-4-26B-A4B-it (May 2026, MoE 4B-active)

- **Round 3 v1 result** (2026-06-17, vendor sampling temp=1.0/top_p=0.95/top_k=64):
  gate **PASS** (mean=14.8s, p99=15.8s, **cv=0.06 — cohort consistency leader**,
  chars 1800-2232 mean=2031). vs Opus 4.7: ROUGE-1=51.8%, ROUGE-L=23.2%, cos=80.6%,
  BLEU=8.6%. vs Sonnet 4.6: ROUGE-1=**56.3%** (+4.5pts), BLEU=10.6% (+2.0pts) —
  **first candidate showing visible Sonnet-style mimicry** in its training
  distribution. Worth noting for the silver-vendor-bias rule.
- Cohort role: mid-pack quality, very consistent latency (cv=0.06 unmatched).
  Boot is slow (22 min — multimodal vision shards). Reasonable per-stage summary
  candidate if speed is acceptable.
- **Phase 2c result** (2026-06-17): **GI vs Opus 41% — STAGE WINNER** (R1/2
  was 45%; dropped 4pts but still leads). KG vs Opus 27% (R1/2 29%; small).
  KG Δ S−O = +6 (Sonnet-lean) consistent with summary's +4.5 mimicry.
  **Per-stage Round 3 GI winner**; cross-vendor judging mandatory.
- Other notes below pre-date Round 3:

- HF: `google/gemma-4-26B-A4B-it` — 51.6 GB BF16
- **vLLM flags REQUIRED**: `--max-num-batched-tokens=4096` (default 2048 is below
  Gemma 4's `max_tokens_per_mm_item=2496` for its multimodal-bidirectional attn)
- Sampling: `temperature=0.0`
- Architecture: detected as `Gemma4ForConditionalGeneration` (multimodal),
  uses `TRITON_ATTN` backend due to heterogeneous head dimensions (head_dim=256,
  global_head_dim=512)
- Known failure modes:
  - Without `--max-num-batched-tokens=4096`: ValueError "Chunked MM input
    disabled but max_tokens_per_mm_item (2496) is larger than
    max_num_batched_tokens (2048)" → boot loop
- Cite: HF model card — `Gemma4ForConditionalGeneration` arch entry

---

## Meta

### Llama-3.3-70B-Instruct-NVFP4 (RedHatAI repack of Meta's Llama-3.3-70B-Instruct)

- HF: `RedHatAI/Llama-3.3-70B-Instruct-NVFP4` — 42.7 GB NVFP4
- vLLM flags: standard NVFP4 (`FlashInferCutlassNvFp4LinearKernel` auto-selected
  on Blackwell)
- Sampling: `temperature=0.0`
- Prompt convention: use strict / anti-verbosity prompt
  (`vllm/strict_summarization/*`) — model writes lengthy narrative summaries by
  default that exceed our 800-token cap
- Known failure modes:
  - With default `ollama/qwen3.5_35b/summarization/*` prompt: hits
    `finish_reason=length` at 800 tokens on dense episodes (p02_e02 / similar)
  - With strict prompt: 46s mean, 1347 char mean, well within spec — 2× speedup
    vs default prompt's first 3 episodes (87s mean)
- **Round 3 v1 result** (2026-06-17, vendor sampling temp=0.6/top_p=0.9 via
  extra_body, kept strict prompt): gate **PASS** (mean=42.9s, p99=49.2s,
  cv=0.12, chars 1001-1518 mean=1259). vs Opus 4.7: ROUGE-1=48.7%, ROUGE-L=25.1%,
  cos=77.9%, BLEU=8.9%. vs Sonnet 4.6: ROUGE-1=49.3%, BLEU=7.5%.
- **Cohort role**: REPRESENTS LLAMA / META in the final landscape. Mid-pack
  quality despite being the cohort's largest model — NVFP4 quant tax brings it
  below smaller BF16 MoE candidates (Moonlight 16B, Qwen3.5 35B-A3B). NOT
  top-dog material at this quant; BF16 70B doesn't fit GB10. See #1022 for
  potential quant retuning.
- **Phase 2c result** (2026-06-17): **WORST IN COHORT** on extraction. GI
  vs Opus 16% (cohort floor, tied with Moonlight). KG vs Opus 20% (sole
  cohort floor). Throughput-bound: 5 tok/s (cohort avg 25), TPOT 200ms
  (cohort avg 50ms), TTFT 500ms. NVFP4 quant tax brutal on structured
  extraction. KV cache peak only 2.5% — massive memory headroom unused,
  suggests `--max-num-batched-tokens` >4096 + prefix caching could help.
  **Drop or quant-retune for #1022 follow-up.**
- Cite: `https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-NVFP4`

---

## Moonshot

### Moonlight-16B-A3B-Instruct (Feb 2025, A3B MoE)

- HF: `moonshotai/Moonlight-16B-A3B-Instruct` — 31.9 GB BF16
- **vLLM flags REQUIRED**: `--max-model-len=8192` (model's
  `max_position_embeddings=8192` — cannot exceed without
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` env which risks nan via RoPE drift)
- Sampling: `temperature=0.0` (no vendor recommendation documented)
- Prompt convention: standard summarization prompts work
- Known failure modes:
  - Without `--max-model-len=8192`: ValueError "max_model_len (32768) greater
    than derived max_model_len (max_position_embeddings=8192)" → boot loop
- **Round 3 v1 result** (2026-06-17): gate **PASS** (mean=9.0s, p99=11.3s,
  cv=0.19, chars 1078-2168). vs Opus 4.7 silver: ROUGE-1=57.5%, **ROUGE-L=32.9%
  (cohort leader)**, cos=78.6%, BLEU=18.7%. vs Sonnet 4.6: similar.
- **Cohort role**: REPRESENTS MOONSHOT AI in the final landscape (Kimi-Linear
  dropped — see § Kimi-Linear-48B-A3B-Instruct). Strong summary-stage
  candidate: top-3 quality + cohort speed leader (after DSV2-Lite). Weak GI/KG
  (Round 1 bottom) — top-dog material for summary-only, not autoresearch
  full-stack.
- **Phase 2c result** (2026-06-17): GI vs Opus 16% (R1/2 19%; smallest cohort
  drop). KG vs Opus 29% (R1/2 29% — **PERFECT stability**, only cohort
  candidate with no drop). **30% entity coverage** (rare in cohort — only
  Moonlight + Qwen3-30B emit entities at all). Δ S−O across all 3 stages
  is [−2, 0] — **truly style-neutral**. **Best safe pick for cross-vendor
  judging contexts** where Qwen3.5/Gemma 4's +3.6 to +4.5 Sonnet-mimicry
  would create methodological doubt.

### Kimi-Linear-48B-A3B-Instruct (Oct 2025, novel linear-attention MoE)

- HF: `moonshotai/Kimi-Linear-48B-A3B-Instruct` — 98.2 GB BF16
- **vLLM flags REQUIRED**: `--trust-remote-code` (for custom `tokenization_kimi.py`)
- vLLM backend: TRITON_MLA + FlashAttention prefill + FlashInfer CUTLASS MoE
- Sampling: `temperature=0.0`
- Memory: 91.65 GiB at load. At util=0.80 → very tight, ~10 GB headroom for KV
  cache + activations
- Boot time: ~38 min first time (download 98 GB) + ~12 min from cache
- Known failure modes:
  - With our strict anti-jailbreak-style prompt: produced refusals ("I'm sorry,
    but I can't comply with this request. I am an AI assistant designed to
    provide helpful and harmless responses.") and one "Stop." reply. Unique in
    the cohort — no other model exhibited this. Likely Constitutional-AI-style
    safety training that misfires on adversarial-looking prompt structure.
  - With our natural / positive-framing prompt: produced chain-of-thought-style
    plain-text narration about the task ("The user wants me to write a clean,
    factual prose summary of a podcast episode...") that exhausted the
    800-token budget before reaching the actual summary
  - Both prompt strategies produced different failure modes; the architectural
    issue (trained as a chat assistant that wants to explain its work) is the
    root cause. Further investigation needed before this model can be used.
- Cite: `https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct`

---

## NVIDIA

### Llama-3.3-Nemotron-Super-49B-v1_5-FP8

- HF: `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8` — 52 GB FP8
- vLLM flags: do NOT use `--reasoning-parser=nemotron_v3` — fails at startup with
  "NemotronV3ReasoningParser reasoning parser could not locate think start/end
  tokens in the tokenizer" because this model's tokenizer vocab doesn't include
  the dedicated think delimiter tokens the parser expects. Remove the flag and
  control reasoning via system prompt instead.
- **Prompt convention REQUIRED (per NVIDIA docs)**: system prompt is literally
  the string `/no_think` (with leading slash) to disable reasoning mode. The
  default empty system prompt enables reasoning ON mode.
- Sampling:
  - **Reasoning OFF mode**: greedy decoding (temperature=0.0). NVIDIA's
    explicit recommendation.
  - Reasoning ON mode: temperature=0.6, top_p=0.95
- Known failure modes:
  - Without `/no_think`: emits `<think>` block immediately, fills 800-token
    budget before reaching summary → guardrail
  - With "detailed thinking off" (Qwen3 convention, not Nemotron's): still
    emits `<think>` — that's the wrong convention. NVIDIA uses `/no_think`
    specifically.
  - With `--reasoning-parser=nemotron_v3` flag: container restart loop
    (missing think tokens in tokenizer)
- Cite: NVIDIA model card <https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5>

---

## Qwen

### Qwen3-30B-A3B-Instruct-2507 (Jul 2025, MoE 3B-active)

- HF: `Qwen/Qwen3-30B-A3B-Instruct-2507` — ~60 GB BF16
- vLLM flags: none non-default. `enable_thinking: false` via extra_body controls
  reasoning preamble.
- Sampling: `temperature=0.0`
- Prompt convention: `ollama/qwen3.5_35b/summarization/*` works
- Postprocessor: none
- **Output length lever**: use `max_length=500` instead of 800 (paragraphs_max
  derives as `max_length // 100`, so 500 → 5 paragraphs vs 800 → 8 paragraphs)
- Known failure modes:
  - With `cloud_structured_max_output_tokens` floor active (#Flightcast bug):
    max_tokens silently clamps to 4096 → 4472 char mean output (over-spec).
    Fixed at code level in #1023 (the bug fix patches `openai_provider.py:1095`
    to bypass the floor for plain-text summarize calls).
- **Round 3 v1 result** (2026-06-17, vendor sampling temp=0.7/top_p=0.8/top_k=20
  /presence_penalty=1.5 via extra_body): gate **PASS** (mean=14.0s, p99=15.0s,
  **cv=0.05** matches Gemma's cohort-leading consistency, chars 2293-2636).
  vs Opus 4.7: ROUGE-1=50.1% (down from R1's 53.2% — vendor sampling worse on
  Opus), ROUGE-L=20.5%, cos=78.6%, BLEU=8.2%. vs Sonnet 4.6: ROUGE-1=**54.5%
  (+4.4 vs Opus — clear Sonnet lean)**, BLEU=10.4%.
- **Cohort role**: mid-pack quality; vendor sampling clearly **leans Sonnet
  style** (Δ=+4.4 ROUGE-1). Speed competitive, consistency excellent. NOT
  the per-stage summary winner but a defensible balanced candidate.
- **Phase 2c result** (2026-06-17): GI vs Opus 36% (R1/2 38%; small drop).
  KG vs Opus **35% (IMPROVED +4pts vs R1/2's 31%)** — sole candidate where
  Round 3 helped extraction. Mechanism: 2507 over-produces topic candidates
  (190 vs cohort 110), so vendor sampling's diversity helps coverage. KG
  Δ S−O = **−9** (Opus-leaning on KG) even though summary is +4.4 Sonnet.
  **Sonnet-mimicry is task-dependent, not model-dependent** — important
  methodology finding ([[silver_judge_vendor_bias]]).

### Qwen3.5-35B-A3B (early 2026, MoE 3B-active, multimodal-capable)

- HF: `Qwen/Qwen3.5-35B-A3B` — 67 GB BF16
- vLLM flags: same as Qwen3-30B-A3B. **`VLLM_GPU_MEM_UTIL=0.65` required**
  (Qwen3.5 OOMs at 0.55-0.60 default per #1016 Phase 2a memory)
- **vLLM flag REQUIRED (discovered Round 3)**: `--reasoning-parser=qwen3` —
  Qwen3.5 is THINKING-BY-DEFAULT. Without the parser flag, think content
  leaks into the response. Vendor docs explicitly require this.
- Sampling: vendor instruct-mode general-tasks: `temperature=0.7, top_p=0.8,
  top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
- Prompt convention: same family as Qwen3-30B
- Postprocessor: none
- Same `cloud_structured` floor bug applies — see Qwen3-30B above
- **Round 3 v1 result** (2026-06-17, vendor sampling + parser flag): gate
  **PASS** (mean=13.6s, p99=15.0s, cv=0.07, chars 2080-2701). vs Opus 4.7:
  ROUGE-1=59.4%, ROUGE-L=27.4%, **cos=83.7% (cohort high)**, BLEU=15.0%.
  vs Sonnet 4.6: ROUGE-1=**63.0% (cohort high)**, ROUGE-L=29.7%, cos=85.3%,
  BLEU=17.2%. Δ (S−O) = **+3.6 — clear Sonnet lean**.
- **Cohort role**: **STRONG TOP-DOG SUMMARY CANDIDATE** — cohort-leading
  cosine (vs both silvers), ROUGE-1 leader, competitive BLEU, mid-range
  speed. The `--reasoning-parser=qwen3` flag was the structural fix that
  reveals its full quality. **However, Sonnet-mimicry caveat applies**:
  cross-vendor judging required to validate the top-dog claim
  ([[silver_judge_vendor_bias]]).
- **Phase 2c result** (2026-06-17): GI vs Opus 36% (R1/2 was 44% — DROPPED
  8pts under vendor sampling). KG vs Opus 38% (R1/2 47% — DROPPED 9pts).
  Confirms hypothesis: vendor sampling helps generative tasks, hurts
  structured extraction. GI/KG Δ S−O both ~+1 (style-neutral on extraction,
  despite +3.6 Sonnet-mimicry on summary). **Per-stage Round 3 winner**:
  KG (38% leads cohort), summary (59.4%). **Top dog overall** with 2-of-3
  stage wins.

---

## How to add a new entry

1. Add a section under the appropriate vendor (alphabetical by vendor).
2. Use the heading pattern `### <full HF model id including any quantization tag>`.
3. Include: HF link + size, vLLM flags REQUIRED, sampling, prompt convention,
   required postprocessor, known failure modes (with empirical data — say which
   episode failed and what the preview showed), citation.
4. If you discovered a non-obvious workaround, cite the source (HF card, vendor
   blog, vLLM docs, or your own commit / experiment report).
5. Cross-reference the `feedback_*` memory entry if there's one that codifies a
   related rule (e.g. `feedback_silver_judge_vendor_bias` for the methodology
   side).
