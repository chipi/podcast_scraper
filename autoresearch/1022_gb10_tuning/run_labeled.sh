#!/usr/bin/env bash
# Helper for #1022 vLLM-on-GB10 tuning eval cells.
#
# Runs summary + GI + KG via run_experiment.py against the autoresearch vLLM
# on DGX, captures wall-clock per stage, and relocates the run dirs to a
# label-tagged subtree under data/eval/runs/1022/<label>/ so cells don't
# overwrite each other.
#
# Usage:
#   bash autoresearch/1022_gb10_tuning/run_labeled.sh <label> [vllm_base_url]
#
# Example:
#   bash autoresearch/1022_gb10_tuning/run_labeled.sh baseline_1
#   bash autoresearch/1022_gb10_tuning/run_labeled.sh cell_a_mem_util_085
#
# Default vllm_base_url: http://dgx-llm-1:8003/v1
#
# Output:
#   data/eval/runs/1022/<label>/{summary,gi,kg}/{predictions.jsonl,run.log,...}
#   autoresearch/1022_gb10_tuning/runs.tsv  (appended row)
#   autoresearch/1022_gb10_tuning/compose_snapshots/<label>.yaml  (compose at run time)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <label> [vllm_base_url]" >&2
  exit 2
fi

LABEL="$1"
VLLM_BASE_URL="${2:-http://dgx-llm-1:8003/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-buddy-is-the-king}"
export VLLM_API_KEY  # run_experiment.py reads this from env
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS_TSV="${REPO}/autoresearch/1022_gb10_tuning/runs.tsv"
LABEL_DIR="${REPO}/data/eval/runs/1022/${LABEL}"
SNAP_DIR="${REPO}/autoresearch/1022_gb10_tuning/compose_snapshots"

mkdir -p "${LABEL_DIR}" "${SNAP_DIR}"

# TSV header (idempotent — appends only if missing)
if [[ ! -s "${RUNS_TSV}" ]]; then
  printf 'label\tstage\tconfig_id\twall_clock_s\tstart_iso\tend_iso\tvllm_base_url\texit_code\n' > "${RUNS_TSV}"
fi

# Stage configs — round3_v1 (vendor-correct sampling, post-#1016).
# Resolved via case in the loop (avoids bash 4+ associative arrays — mac ships 3.2).
config_for() {
  case "$1" in
    summary) echo "data/eval/configs/summarization/autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_paragraph_round3_v1.yaml" ;;
    gi)      echo "data/eval/configs/gi_autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_round3_v1.yaml" ;;
    kg)      echo "data/eval/configs/kg_autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_round3_v1.yaml" ;;
    *) echo "unknown stage: $1" >&2; return 1 ;;
  esac
}

# Pre-flight: snapshot compose state on DGX for audit trail.
#
# `docker compose config` resolves and inlines every ${...} env var — including
# secrets like HF_TOKEN and VLLM_API_KEY. The post-snapshot scrub below restores
# the placeholder form before the file ever touches git so the audit-trail file
# can be committed safely. This was added after a 2026-06-20 push-protection
# block on the prior #1022 snapshots (a literal HF token was captured in
# compose_snapshots/*.yaml; cleaned via filter-branch + this scrubber).
SNAP_FILE="${SNAP_DIR}/${LABEL}.yaml"
echo "[snap] capturing compose snapshot → ${SNAP_FILE}"
ssh dgx-llm-1 'cd ~/agentic-ai-homelab/infra/vllm/autoresearch && docker compose config 2>/dev/null' > "${SNAP_FILE}" || {
  echo "  WARN: compose snapshot failed (continuing)" >&2
}
# Scrub: replace known secret env values with their placeholder form so the
# snapshot file never carries a live secret. Add new patterns here when new
# secret-bearing env vars enter the compose file.
if [[ -s "${SNAP_FILE}" ]]; then
  sed -i.bak \
    -e 's|HF_TOKEN: hf_[A-Za-z0-9]*|HF_TOKEN: ${HF_TOKEN}|g' \
    -e 's|VLLM_API_KEY: [^[:space:]]*|VLLM_API_KEY: ${VLLM_API_KEY}|g' \
    "${SNAP_FILE}" && rm -f "${SNAP_FILE}.bak"
fi

# Pre-flight: confirm vLLM is healthy + served-model-name = autoresearch.
# /health is unauthenticated; /v1/models requires the bearer token.
echo "[probe] vLLM /health + /v1/models"
if ! curl -sf --connect-timeout 5 "${VLLM_BASE_URL%/v1}/health" >/dev/null; then
  echo "ERROR: vLLM /health not 200 at ${VLLM_BASE_URL}" >&2
  exit 3
fi
MODEL_ID="$(curl -sf -H "Authorization: Bearer ${VLLM_API_KEY}" "${VLLM_BASE_URL}/models" | jq -r '.data[0].id')"
if [[ "${MODEL_ID}" != "autoresearch" ]]; then
  echo "ERROR: served-model-name = '${MODEL_ID}', expected 'autoresearch'" >&2
  exit 4
fi
echo "  vLLM ready, served-model-name = ${MODEL_ID}"

# Warmup: discard first N calls — cold-load + JIT settling + CUDA graph capture.
# WARMUP_CALLS=3 (default) is the minimal version. WARMUP_CALLS=10 + longer
# max_tokens approximates Cell D ("JIT pre-warm ping in entrypoint") — bigger
# warmup amortizes more CUDA graph captures before the eval starts measuring.
WARMUP_CALLS="${WARMUP_CALLS:-3}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-4}"
echo "[warmup] ${WARMUP_CALLS} throwaway calls (max_tokens=${WARMUP_MAX_TOKENS})"
for i in $(seq 1 "${WARMUP_CALLS}"); do
  curl -sf -X POST "${VLLM_BASE_URL}/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${VLLM_API_KEY}" \
    -d "{\"model\":\"autoresearch\",\"messages\":[{\"role\":\"user\",\"content\":\"Summarize one sentence: the quick brown fox jumps over the lazy dog. Now write four sentences about that pattern.\"}],\"max_tokens\":${WARMUP_MAX_TOKENS}}" \
    >/dev/null || true
done

# Run each stage; time it; move outputs to label dir
for stage in summary gi kg; do
  CONFIG="${REPO}/$(config_for "${stage}")"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 5
  fi
  # Python YAML parse handles both quoted and unquoted id: values (the
  # summary config quotes them, GI/KG configs don't — first run hit this).
  CONFIG_ID="$("${REPO}/.venv/bin/python" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['id'])" "${CONFIG}")"
  if [[ -z "${CONFIG_ID}" ]]; then
    echo "ERROR: could not extract id from ${CONFIG}" >&2
    exit 6
  fi
  RUN_DIR="${REPO}/data/eval/runs/${CONFIG_ID}"

  echo ""
  echo "[stage:${stage}] config=${CONFIG_ID}"
  START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  START_S=$(date +%s)

  set +e
  PYTHONPATH="${REPO}" "${REPO}/.venv/bin/python" \
    "${REPO}/scripts/eval/experiment/run_experiment.py" \
    "${CONFIG}" \
    --vllm-base-url "${VLLM_BASE_URL}" \
    --dry-run \
    --force 2>&1 | tee "${LABEL_DIR}/${stage}.runlog"
  RC=${PIPESTATUS[0]}
  set -e

  END_S=$(date +%s)
  END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  WALL=$((END_S - START_S))

  # Relocate run dir under label
  if [[ -d "${RUN_DIR}" ]]; then
    rm -rf "${LABEL_DIR}/${stage}"
    mv "${RUN_DIR}" "${LABEL_DIR}/${stage}"
    echo "  moved ${RUN_DIR} → ${LABEL_DIR}/${stage}"
  fi

  printf '%s\t%s\t%s\t%d\t%s\t%s\t%s\t%d\n' \
    "${LABEL}" "${stage}" "${CONFIG_ID}" "${WALL}" \
    "${START_ISO}" "${END_ISO}" "${VLLM_BASE_URL}" "${RC}" \
    >> "${RUNS_TSV}"

  if [[ ${RC} -ne 0 ]]; then
    echo "FAIL [${stage}] exit=${RC}" >&2
    exit ${RC}
  fi
done

echo ""
echo "=== ${LABEL} complete ==="
column -t -s $'\t' "${RUNS_TSV}" | tail -n 4
