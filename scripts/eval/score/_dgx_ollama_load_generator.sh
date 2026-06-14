#!/usr/bin/env bash
# Continuous Ollama load generator for #963 scenario 3 heavy contention.
#
# Loops POSTs to qwen3.5:35b until killed. Each POST is a realistic-size
# summarisation request (~500 tokens input, ~120 tokens output) so the
# GPU stays warm with actual compute, not just VRAM-resident weights.
#
# Usage:
#   bash scripts/eval/score/_dgx_ollama_load_generator.sh start &
#   <run whisper eval>
#   kill %1
#
# Env vars:
#   DGX_HOST (default: dgx-llm-1.tail6d0ed4.ts.net)
#   OLLAMA_MODEL (default: qwen3.5:35b)
#   LOAD_LOG (default: /tmp/dgx_ollama_load.jsonl) — per-request timing
set -euo pipefail
DGX_HOST="${DGX_HOST:-dgx-llm-1.tail6d0ed4.ts.net}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:35b}"
LOAD_LOG="${LOAD_LOG:-/tmp/dgx_ollama_load.jsonl}"

PROMPT='Summarise this podcast excerpt in 3 bullet points. Excerpt: A long podcast \
about markets and macro economics, discussing inflation, employment, interest \
rates, and the implications for portfolios. Two hosts trade observations about \
recent Fed meetings and the resilience of consumer spending into the holiday \
season. Mention what each speaker would do next quarter.'

: > "$LOAD_LOG"
echo "[load-gen] started against $DGX_HOST/$OLLAMA_MODEL, logging to $LOAD_LOG" >&2

while true; do
    payload=$(printf '{"model":"%s","prompt":"%s","stream":false,"options":{"num_predict":120,"temperature":0.3}}' \
        "$OLLAMA_MODEL" "$PROMPT")
    t0=$(/usr/bin/python3 -c 'import time; print(time.time())')
    response=$(/usr/bin/curl -s -m 90 -X POST "http://$DGX_HOST:11434/api/generate" \
        -H 'Content-Type: application/json' -d "$payload" 2>/dev/null || echo '{"error":"curl_failed"}')
    t1=$(/usr/bin/python3 -c 'import time; print(time.time())')
    elapsed=$(/usr/bin/python3 -c "print(round($t1 - $t0, 2))")
    echo "{\"ts\":$t1,\"elapsed_s\":$elapsed}" >> "$LOAD_LOG"
done
