# Per-model optimal vLLM-on-GB10 params — #1016 observations

Live notes collected while Round 2 runs the harness fixes. Used as input to #1022
(systematic vLLM-on-GB10 tuning exploration). Each model row captures:

- **VLLM_GPU_MEM_UTIL** used (the env var that scaled `--gpu-memory-utilization`)
- **max-model-len** required (from compose)
- **max-num-batched-tokens** required (added per Gemma 4 multimodal constraint)
- **trust-remote-code** required (yes/no)
- **GPU memory at load** (from `Model loading took XX GiB memory`)
- **Boot wall-clock** (download + load + compile + autotune)
- **Boot stage cache hits** (torch.compile cache hits on second boot)
- **KV cache % observed during inference** (from `/metrics` poll)
- **TTFT / TPOT** (from histogram buckets)
- **Best chars/episode / s/episode** observed (gate run)
- **Required prompt adaptations** (strict / anti-think / etc.)
- **Failure modes if any**

| Model | UTIL | max-model-len | max-num-batched | trust-remote | Load mem | Boot | KV % obs | s/ep | chars/ep | Prompt adapt | Failure mode |
|-------|-----:|--------------:|----------------:|:------------:|---------:|-----:|---------:|-----:|---------:|:-------------|-------------|
| Qwen3-30B-A3B-Instruct-2507 | 0.65 | 32768 | default 2048 | no | ~60 GB | ~12 min | tbd | 25.6 | 4472 (over-spec) | tight prompts | floor bug |
| Qwen3.5-35B-A3B | 0.65 | 32768 | default 2048 | no | 65.5 GB | ~15 min | tbd | 21.7 | 3779 (over-spec) | tight prompts | floor bug |
| Mistral-Small-3.2-24B-Instruct-2506 | 0.55 | 32768 | default 2048 | no | tbd | ~10 min | tbd | 72.6 | 2057 | none | none |
| Magistral-Small-2509 | 0.55 | 32768 | default 2048 | no | tbd | ~10 min | tbd | 64.8 | 1762 | none (reasoning_parser=mistral) | none |
| Ministral-3-14B-Instruct-2512 | 0.65 | 32768 | default 2048 | no | tbd | ~6 min | tbd | 30.9 | 2678 | none | none |
| Gemma-4-26B-A4B-it | 0.65 | 32768 | **4096 (required for multimodal)** | no | 48.5 GB | ~32 min | tbd | 14.6 | 2009 | none | needs max-num-batched bump |
| Moonlight-16B-A3B-Instruct | 0.55 | **8192 (max_position_embeddings limit)** | 4096 | no | tbd | ~12 min | tbd | 8.8 | 1674 | none | needs max-model-len reduce |
| Mistral-Small-4-119B-NVFP4 | 0.80 | 32768 | 4096 | no | ~67 GB | ~32 min | OOM | n/a | n/a | n/a | OOM on inference — needs max-num-seqs=4 + max-model-len=8192 |
| Llama-3.3-70B-Instruct-NVFP4 | 0.70 | 32768 | 4096 | no | tbd | ~18 min | tbd | 85 (3 eps) | 2283-2595 | strict prompt | verbosity at 800 cap |
| DeepSeek-V2-Lite-Chat | 0.55 | 32768 | 4096 | no | tbd | ~9 min | tbd | **4.8 (decoded)** | **983 (decoded)** | strict prompt + decode_r1_byte_level postprocessor | round 1: BPE + verbosity |
| DeepSeek-R1-0528-Qwen3-8B | 0.55 | 32768 | 4096 | no | tbd | ~5 min | tbd | n/a | n/a | strict + reasoning-off prompt (TBD) | round 1: reasoning consumed budget |
| Kimi-Linear-48B-A3B-Instruct | 0.80 | 32768 | 4096 | **YES** (tokenization_kimi.py) | tbd | ~38 min | tbd | 10 (2 eps) | 1318 | strict prompt | round 1: verbosity at 800 cap |
| Nemotron-Super-49B-v1_5-FP8 | 0.65 | 32768 | 4096 | no | tbd | ~22 min | tbd | n/a | n/a | "/no_think" or "detailed thinking off" system msg | round 1: reasoning consumed budget |

Updated as runs land. Cross-reference: vllm_kv_metrics.log in same dir.
