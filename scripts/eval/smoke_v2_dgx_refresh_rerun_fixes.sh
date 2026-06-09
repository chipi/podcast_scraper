#!/usr/bin/env bash
#
# Smoke v2 DGX refresh — rerun the 2 cells that failed in the original retry
# sweep after upstream fixes landed:
#
#  - qwen3.6:latest produced 0-token output because qwen3.6 is a reasoning
#    model that emits tokens to the `thinking` channel by default. Fix in
#    src/podcast_scraper/providers/ollama/ollama_provider.py extends the
#    `reasoning_effort: none` shim from qwen3.5 to all qwen3.x variants.
#
#  - deepseek-r1:70b hit Ollama "Request timed out" on the first episode.
#    R1 chain-of-thought generates thousands of tokens before the final
#    answer; the default 120s read timeout is too short. We pass
#    EXPERIMENT_OLLAMA_READ_TIMEOUT=1200 (20min) so the run completes.
#
# Overwrites the run dirs at the same path as the original retry sweep.
# Output goes to data/eval/runs/llm_ollama_<slug>_dgx_smoke_v2_2026_06/.

set -euo pipefail

: "${DGX_TAILNET_FQDN:?DGX_TAILNET_FQDN must be set — source infra/.env.dgx.local}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://${DGX_TAILNET_FQDN}:11434/v1}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 20-min read timeout per Ollama request — covers r1:70b chain-of-thought on
# a single full-length podcast episode. Doesn't affect other cells (their
# requests complete in seconds and unused timeout is free).
export EXPERIMENT_OLLAMA_READ_TIMEOUT=1200

SWEEP_TAG="dgx_smoke_v2_2026_06"
STABLE_BASELINE="baseline_llm_ollama_qwen35_35b_smoke_paragraph_v1"

# Just the cells that failed before. The other 5 retry-sweep cells
# (deepseek-r1 7b/14b/32b, gpt-oss_20b) produced valid scores and don't
# need to be re-run.
RERUN_MODELS=(
  "qwen3.6_latest"
  "deepseek-r1_70b"
)

START_TS=$(date +%s)
echo "=== rerun for fixed cells ==="
echo "OLLAMA_API_BASE=$OLLAMA_API_BASE"
echo "EXPERIMENT_OLLAMA_READ_TIMEOUT=${EXPERIMENT_OLLAMA_READ_TIMEOUT}s"
echo "stable reference: $STABLE_BASELINE"
echo "models: ${#RERUN_MODELS[@]}"
echo "start: $(date)"
echo ""

PASS=()
FAIL=()
for slug in "${RERUN_MODELS[@]}"; do
  CFG="data/eval/configs/summarization/autoresearch_prompt_ollama_${slug}_smoke_paragraph_v1.yaml"
  OUT="data/eval/runs/llm_ollama_${slug}_${SWEEP_TAG}"

  if [ ! -f "$CFG" ]; then
    echo "SKIP: $slug — config not found at $CFG"
    FAIL+=("$slug")
    continue
  fi

  echo ""
  echo "--- $slug (vs $STABLE_BASELINE) ---"
  date +%H:%M:%S
  if make benchmark CONFIG="$CFG" BASELINE="$STABLE_BASELINE" OUTPUT_DIR="$OUT" SMOKE=1; then
    PASS+=("$slug")
  else
    echo "FAIL on $slug — continuing"
    FAIL+=("$slug")
  fi
done

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
echo ""
echo "=== rerun done in $((DURATION / 60))m $((DURATION % 60))s ==="
echo "pass: ${#PASS[@]}"
printf "  %s\n" "${PASS[@]}"
echo "fail: ${#FAIL[@]}"
[ ${#FAIL[@]} -gt 0 ] && printf "  %s\n" "${FAIL[@]}"
