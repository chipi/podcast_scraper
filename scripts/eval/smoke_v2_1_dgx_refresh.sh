#!/usr/bin/env bash
#
# Smoke v2.1 DGX refresh — second-pass sweep with newer Ollama models that
# emerged in the months after the v2 matrix was finalized (#45).
#
# Candidates picked from a small deep-research pass on 2026-06-08:
#
#   gemma3:27b           — Google's Gemma 3 (March 2025) — direct successor
#                          to gemma2:9b/27b already in v2 matrix
#   phi4:14b             — Microsoft's Phi-4 (December 2024) — reasoning-tuned
#                          mid-size, fills the 14B gap between qwen3.5:9b and
#                          qwen3.5:27b
#   hermes3:8b           — Nous Research Llama 3.1 fine-tune — often outperforms
#                          base Llama on prose tasks; cheap to evaluate
#   mistral-small:24b    — Mistral Small successor to the v2 matrix's
#                          mistral-small3.2 entry
#
# Excluded (out of memory budget or out of scope):
#   - hermes3:70b (~40 GB) — memory tight w/ Speaches + pyannote + Ollama
#   - deepseek-v3:671b — way too big
#   - llama4:scout — size unclear; defer
#   - qwen3-coder:30b — coder-specialized, wrong domain (also in v2 exclusions)
#
# Uses STABLE_BASELINE=baseline_llm_ollama_qwen35_35b_smoke_paragraph_v1 as the
# reference (same as smoke v2 retry sweep). qwen3.6:latest from v2 will be
# included in cross-comparison via the eval reports once this lands.
#
# Output goes to data/eval/runs/llm_ollama_<slug>_dgx_smoke_v2_1_2026_06/.

set -euo pipefail

: "${DGX_TAILNET_FQDN:?DGX_TAILNET_FQDN must be set — source infra/.env.dgx.local}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://${DGX_TAILNET_FQDN}:11434/v1}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Read timeout for the larger reasoning-capable models (Phi-4 has implicit
# reasoning behavior, Gemma 3 is multimodal). 600s covers any one-episode
# call; the qwen3 reasoning_effort=none shim from #924 also applies to
# Qwen3-family models in this sweep (none here, but the safety net stands).
export EXPERIMENT_OLLAMA_READ_TIMEOUT=600

SWEEP_TAG="dgx_smoke_v2_1_2026_06"
STABLE_BASELINE="baseline_llm_ollama_qwen35_35b_smoke_paragraph_v1"

V2_1_MODELS=(
  "gemma3_27b"
  "phi4_14b"
  "hermes3_8b"
  "mistral-small_24b"
)

START_TS=$(date +%s)
echo "=== smoke v2.1 sweep ==="
echo "OLLAMA_API_BASE=$OLLAMA_API_BASE"
echo "EXPERIMENT_OLLAMA_READ_TIMEOUT=${EXPERIMENT_OLLAMA_READ_TIMEOUT}s"
echo "stable reference: $STABLE_BASELINE"
echo "models: ${#V2_1_MODELS[@]}"
echo "start: $(date)"
echo ""

PASS=()
FAIL=()
for slug in "${V2_1_MODELS[@]}"; do
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
echo "=== smoke v2.1 done in $((DURATION / 60))m $((DURATION % 60))s ==="
echo "pass: ${#PASS[@]}"
printf "  %s\n" "${PASS[@]}"
echo "fail: ${#FAIL[@]}"
[ ${#FAIL[@]} -gt 0 ] && printf "  %s\n" "${FAIL[@]}"
