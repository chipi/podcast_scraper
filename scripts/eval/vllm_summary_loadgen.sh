#!/usr/bin/env bash
# Continuous vLLM summary load generator for #963 scenario 3.
#
# Loops indefinitely over the v2 fixture transcripts, firing one
# chat.completions request after another against the vLLM autoresearch
# endpoint. Keeps the GPU under sustained summary-inference load so the
# concurrent whisper sweep can measure contention impact.
#
# Usage:
#   VLLM_URL=http://your-dgx.tailnet.ts.net:8003/v1 \
#     scripts/eval/vllm_summary_loadgen.sh
#
# Stop with Ctrl-C or `kill` from the parent shell.
set -euo pipefail

: "${VLLM_URL:=http://your-dgx.tailnet.ts.net:8003/v1}"
: "${MODEL:=Qwen/Qwen3.6-35B-A3B}"
: "${EPISODES:=p01_e01 p02_e01 p03_e01 p04_e01 p05_e01}"
: "${MATERIALIZED_DIR:=data/eval/materialized/curated_5feeds_smoke_v1}"
: "${PROMPT_SYSTEM:=src/podcast_scraper/prompts/ollama/qwen3.5_35b/summarization/system_v1.j2}"
: "${PROMPT_USER:=src/podcast_scraper/prompts/ollama/qwen3.5_35b/summarization/long_v1.j2}"

iteration=0
while true; do
  iteration=$((iteration + 1))
  for ep in ${EPISODES}; do
    out_dir=$(mktemp -d)
    echo ">>> [$(date -u +%H:%M:%S)] loadgen iter=${iteration} ep=${ep}"
    .venv/bin/python -u scripts/eval/score/summary_vllm_predict_v1.py \
      --vllm-url "${VLLM_URL}" \
      --model "${MODEL}" \
      --prompt-system "${PROMPT_SYSTEM}" \
      --prompt-user "${PROMPT_USER}" \
      --materialized-dir "${MATERIALIZED_DIR}" \
      --episodes "${ep}" \
      --output-dir "${out_dir}" \
      --disable-thinking 2>&1 | tail -3 || echo "  loadgen ${ep} errored (continuing)"
    rm -rf "${out_dir}"
  done
done
