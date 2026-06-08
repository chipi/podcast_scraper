#!/usr/bin/env bash
#
# Smoke v2 DGX refresh — retry the 7 new models that the main sweep
# failed because their per-model baseline didn't exist.
#
# Uses BASELINE=baseline_llm_ollama_qwen35_35b_smoke_paragraph_v1 as the
# stand-in reference. qwen3.5:35b leads Ollama paragraph quality per the
# April 2026 AI comparison guide, no known stability issues (the qwen3.5:9b
# bundled flakiness in #912 is for the bundled summarize+clean pipeline
# path, not the paragraph autoresearch track this sweep evaluates). Using
# the best stable existing model as the reference frames every new model's
# delta as "how close to the local-Ollama ceiling does it get."
#
# Output goes to the SAME data/eval/runs/llm_ollama_<slug>_dgx_smoke_v2_2026_06/
# paths as the main sweep — failed runs get replaced cleanly.

set -euo pipefail

: "${DGX_TAILNET_FQDN:?DGX_TAILNET_FQDN must be set — source infra/.env.dgx.local}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://${DGX_TAILNET_FQDN}:11434/v1}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

SWEEP_TAG="dgx_smoke_v2_2026_06"
STABLE_BASELINE="baseline_llm_ollama_qwen35_35b_smoke_paragraph_v1"

NEW_MODELS=(
  "deepseek-r1_7b"
  "deepseek-r1_14b"
  "deepseek-r1_32b"
  "deepseek-r1_70b"
  "gpt-oss_20b"
  "qwen3.6_latest"
  "qwen3-coder_30b"
)

START_TS=$(date +%s)
echo "=== retry sweep for new models ==="
echo "OLLAMA_API_BASE=$OLLAMA_API_BASE"
echo "stable reference: $STABLE_BASELINE"
echo "models: ${#NEW_MODELS[@]}"
echo "start: $(date)"
echo ""

PASS=()
FAIL=()
for slug in "${NEW_MODELS[@]}"; do
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
  if make benchmark CONFIG="$CFG" BASELINE="$STABLE_BASELINE" OUTPUT_DIR="$OUT" SMOKE=1 2>&1 | tail -10; then
    PASS+=("$slug")
  else
    echo "FAIL on $slug — continuing"
    FAIL+=("$slug")
  fi
done

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
echo ""
echo "=== retry sweep done in $((DURATION / 60))m $((DURATION % 60))s ==="
echo "pass: ${#PASS[@]}"
printf "  %s\n" "${PASS[@]}"
echo "fail: ${#FAIL[@]}"
[ ${#FAIL[@]} -gt 0 ] && printf "  %s\n" "${FAIL[@]}"
