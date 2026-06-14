#!/usr/bin/env bash
# Continuous vLLM load generator for #963 scenario 3 heavy contention.
#
# Mirrors _dgx_ollama_load_generator.sh but targets the coder-next vLLM
# on :9000 (homelab repo, see infra/vllm/coder-next/docker-compose.yml).
#
# Usage:
#   bash scripts/eval/score/_dgx_vllm_load_generator.sh &
#   <run whisper eval>
#   kill %1
#
# Env vars:
#   DGX_HOST (default: dgx-llm-1.tail6d0ed4.ts.net)
#   VLLM_PORT (default: 9000)
#   VLLM_MODEL (default: Qwen/Qwen3-Coder-Next-FP8)
#   VLLM_API_KEY (default: buddy-is-the-king)
#   LOAD_LOG (default: /tmp/dgx_vllm_load.jsonl) — per-request timing
set -euo pipefail
DGX_HOST="${DGX_HOST:-dgx-llm-1.tail6d0ed4.ts.net}"
VLLM_PORT="${VLLM_PORT:-9000}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
VLLM_API_KEY="${VLLM_API_KEY:-buddy-is-the-king}"
LOAD_LOG="${LOAD_LOG:-/tmp/dgx_vllm_load.jsonl}"

PROMPT='Summarise this podcast excerpt in 3 bullet points. Excerpt: A long podcast about markets and macro economics, discussing inflation, employment, interest rates, and the implications for portfolios. Two hosts trade observations about recent Fed meetings and the resilience of consumer spending into the holiday season. Mention what each speaker would do next quarter.'

: > "$LOAD_LOG"
echo "[vllm-load-gen] started against $DGX_HOST:$VLLM_PORT/$VLLM_MODEL, logging to $LOAD_LOG" >&2

while true; do
    payload=$(/usr/bin/python3 -c "
import json
import sys
print(json.dumps({
    'model': '$VLLM_MODEL',
    'messages': [{'role':'user','content':'''$PROMPT'''}],
    'max_tokens': 120,
    'temperature': 0.3,
}))")
    t0=$(/usr/bin/python3 -c 'import time; print(time.time())')
    /usr/bin/curl -s -m 90 -X POST "http://$DGX_HOST:$VLLM_PORT/v1/chat/completions" \
        -H "Authorization: Bearer $VLLM_API_KEY" \
        -H 'Content-Type: application/json' -d "$payload" >/dev/null 2>/dev/null || true
    t1=$(/usr/bin/python3 -c 'import time; print(time.time())')
    elapsed=$(/usr/bin/python3 -c "print(round($t1 - $t0, 2))")
    echo "{\"ts\":$t1,\"elapsed_s\":$elapsed}" >> "$LOAD_LOG"
done
