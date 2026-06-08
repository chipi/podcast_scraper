#!/usr/bin/env bash
#
# Smoke v2 DGX refresh sweep (#924).
#
# Runs the autoresearch smoke v1 dataset against 19 Ollama models on DGX
# (12 existing + 7 new: deepseek-r1:{7b,14b,32b,70b}, gpt-oss:20b,
# qwen3.6:latest, qwen3-coder:30b).
#
# Output: data/eval/runs/llm_ollama_<slug>_dgx_smoke_v2_2026_06/
#
# Pre-flight:
#   - ``ollama pull`` all 19 models on DGX (script assumes they're there)
#   - ``OLLAMA_API_BASE`` points at DGX
#   - ``PYTHONPATH`` includes repo root (run_benchmark expects it)
#
# Total wall-clock estimate on DGX: ~2-3 hours.
#
# Idempotent-ish: if a run's OUTPUT_DIR already exists, ``make benchmark``
# overwrites. Safe to re-run.

set -euo pipefail

# --- Configuration ---------------------------------------------------------

: "${DGX_TAILNET_FQDN:?DGX_TAILNET_FQDN must be set — source infra/.env.dgx.local}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://${DGX_TAILNET_FQDN}:11434/v1}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

SWEEP_TAG="dgx_smoke_v2_2026_06"

# Existing 12 (config slugs use underscore separator everywhere — match
# existing config file naming).
EXISTING=(
  "llama32_3b"
  "phi3_mini"
  "mistral_7b"
  "qwen25_7b"
  "gemma2_9b"
  "llama31_8b"
  "qwen35_9b"
  "mistral_nemo_12b"
  "mistral_small3_2"
  "qwen35_27b"
  "qwen25_32b"
  "qwen35_35b"
)

# 7 new — config slugs preserve hyphens + dots from the actual model name
# (deviates slightly from existing convention; framework doesn't care).
NEW=(
  "deepseek-r1_7b"
  "deepseek-r1_14b"
  "deepseek-r1_32b"
  "deepseek-r1_70b"
  "gpt-oss_20b"
  "qwen3.6_latest"
  "qwen3-coder_30b"
)

# Run order: small → large, mixed providers, defers slowest (70b) to the end
# so an early failure doesn't waste hours.
ORDER=(
  llama32_3b phi3_mini mistral_7b qwen25_7b gemma2_9b llama31_8b qwen35_9b
  deepseek-r1_7b deepseek-r1_14b
  mistral_nemo_12b mistral_small3_2
  gpt-oss_20b
  qwen3-coder_30b
  qwen35_27b qwen25_32b deepseek-r1_32b qwen35_35b
  qwen3.6_latest
  deepseek-r1_70b
)

# --- Sweep -----------------------------------------------------------------

mkdir -p data/eval/runs
START_TS=$(date +%s)
echo "=== smoke v2 DGX refresh sweep ==="
echo "OLLAMA_API_BASE=$OLLAMA_API_BASE"
echo "models to run: ${#ORDER[@]}"
echo "start: $(date)"
echo ""

PASS=()
FAIL=()
for slug in "${ORDER[@]}"; do
  CFG="data/eval/configs/summarization/autoresearch_prompt_ollama_${slug}_smoke_paragraph_v1.yaml"
  BASE="baseline_llm_ollama_${slug}_smoke_paragraph_v1"
  OUT="data/eval/runs/llm_ollama_${slug}_${SWEEP_TAG}"

  if [ ! -f "$CFG" ]; then
    echo "SKIP: $slug — config not found at $CFG"
    FAIL+=("$slug")
    continue
  fi

  echo ""
  echo "--- $slug ---"
  date +%H:%M:%S
  if make benchmark CONFIG="$CFG" BASELINE="$BASE" OUTPUT_DIR="$OUT" SMOKE=1 2>&1 | tail -10; then
    PASS+=("$slug")
  else
    echo "FAIL on $slug — continuing sweep"
    FAIL+=("$slug")
  fi
done

# --- Summary ---------------------------------------------------------------

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
echo ""
echo "=== sweep complete in $((DURATION / 60))m $((DURATION % 60))s ==="
echo "pass: ${#PASS[@]} models"
printf "  %s\n" "${PASS[@]}"
echo "fail: ${#FAIL[@]} models"
[ ${#FAIL[@]} -gt 0 ] && printf "  %s\n" "${FAIL[@]}"
echo ""
echo "outputs under data/eval/runs/llm_ollama_*_${SWEEP_TAG}/"
