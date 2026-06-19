#!/usr/bin/env bash
# #1033 step 2 — cohort rerun on the corrected pipeline (kg_extraction_src=provider).
#
# Sweeps the 7-candidate #1016 Round 3 cohort, running both GI + KG eval
# per candidate against dev_v1. Each candidate gets one compose-swap cycle:
# load model → run GI → run KG → score → move to next.
#
# Restores Cell F (NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4) at the end.
#
# Output:
#   data/eval/runs/1033_rerun/<candidate>/{gi,kg}/{predictions.jsonl,...}
#   autoresearch/1033_cohort_rerun/runs.tsv  (per-stage timing + score)
#
# Refs: #1033, #112.

set -uo pipefail  # NOT -e — one candidate's failure shouldn't stop the sweep

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DGX_HOST="${DGX_HOST:-dgx-llm-1}"
VLLM_API_KEY="${VLLM_API_KEY:-buddy-is-the-king}"
export VLLM_API_KEY

OUT_DIR="${REPO_ROOT}/data/eval/runs/1033_rerun"
SCRATCH_DIR="${REPO_ROOT}/autoresearch/1033_cohort_rerun"
RUNS_TSV="${SCRATCH_DIR}/runs.tsv"
LOGS_DIR="${SCRATCH_DIR}/logs"
mkdir -p "${OUT_DIR}" "${SCRATCH_DIR}" "${LOGS_DIR}"

# Candidate registry. Each line:
#   <key> <hf_model_id> <max_model_len> <extras_pipe_delimited_or_underscore>
CANDIDATES=(
  "cell_f                  NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4                    32768  _"
  "deepseek_v2_lite        deepseek-ai/DeepSeek-V2-Lite-Chat                        32768  --trust-remote-code"
  "gemma_4_26b_a4b         google/gemma-4-26B-A4B-it                                32768  _"
  "llama_3_3_70b_nvfp4     RedHatAI/Llama-3.3-70B-Instruct-NVFP4                    32768  _"
  "qwen3_5_35b_a3b         Qwen/Qwen3.5-35B-A3B                                     32768  --reasoning-parser=qwen3"
  "moonlight_16b_a3b       moonshotai/Moonlight-16B-A3B-Instruct                    8192   --trust-remote-code"
  "ministral_3_14b         mistralai/Ministral-3-14B-Instruct-2512                  32768  --tokenizer_mode=mistral|--config_format=mistral|--load_format=mistral"
)

if [[ ! -s "${RUNS_TSV}" ]]; then
  printf 'candidate\tstage\twall_clock_s\tscore_metric\tscore_value\trun_dir\tstatus\n' > "${RUNS_TSV}"
fi

SKIP_LIST=""
if [[ "${1:-}" == "--skip" ]] && [[ -n "${2:-}" ]]; then
  SKIP_LIST=",${2},"
fi

_log() { echo "[$(date -u +%H:%M:%S)] $1"; }
_should_skip() { [[ "${SKIP_LIST}" == *",$1,"* ]]; }

_swap_compose() {
  local model="$1" max_len="$2" extras_pipe="$3"
  ssh "${DGX_HOST}" "cd ~/agentic-ai-homelab/infra/vllm/autoresearch && python3 - <<PYEOF
import re

with open('docker-compose.yml') as f:
    content = f.read()

content = re.sub(
    r'^(      - )(?:NVFP4/[^\s]+|moonshotai/[^\s]+|Qwen/[^\s]+|google/[^\s]+|RedHatAI/[^\s]+|deepseek-ai/[^\s]+|mistralai/[^\s]+)\$',
    r'\g<1>${model}',
    content, count=1, flags=re.MULTILINE)

content = re.sub(
    r'^(      - --max-model-len=)\d+',
    r'\g<1>${max_len}',
    content, count=1, flags=re.MULTILINE)

for legacy in ('--trust-remote-code', '--reasoning-parser=qwen3',
               '--tokenizer_mode=mistral', '--config_format=mistral',
               '--load_format=mistral'):
    content = re.sub(rf'^      - {re.escape(legacy)}\n', '',
                     content, flags=re.MULTILINE)

extras = '''${extras_pipe}'''
if extras and extras != '_':
    new_lines = ''.join(f'      - {flag.strip()}\n' for flag in extras.split('|') if flag.strip())
    content = re.sub(
        r'^(      - --served-model-name=autoresearch\n)',
        r'\g<1>' + new_lines,
        content, count=1, flags=re.MULTILINE)

with open('docker-compose.yml', 'w') as f:
    f.write(content)
PYEOF
"
}

_restart_vllm() {
  ssh "${DGX_HOST}" '~/agentic-ai-homelab/infra/dgx/bin/gpu-mode-swap.sh idle >/dev/null 2>&1 || true; ~/agentic-ai-homelab/infra/dgx/bin/gpu-mode-swap.sh research >/dev/null 2>&1 || true'
}

_wait_for_vllm() {
  local deadline=$(( $(date +%s) + 900 ))
  while (( $(date +%s) < deadline )); do
    if curl -sf --connect-timeout 3 "http://${DGX_HOST}:8003/health" >/dev/null 2>&1; then
      local model_id
      model_id="$(curl -sf -H "Authorization: Bearer ${VLLM_API_KEY}" "http://${DGX_HOST}:8003/v1/models" 2>/dev/null | jq -r '.data[0].id' 2>/dev/null || true)"
      [[ "${model_id}" == "autoresearch" ]] && return 0
    fi
    if ssh "${DGX_HOST}" 'docker logs vllm-autoresearch 2>&1 | tail -5' 2>/dev/null | \
        grep -qE 'CUDA out of memory|RuntimeError|Traceback|FAILED|exited'; then
      _log "  vLLM boot error detected"
      ssh "${DGX_HOST}" 'docker logs vllm-autoresearch 2>&1 | tail -15' 2>/dev/null
      return 1
    fi
    sleep 15
  done
  _log "  vLLM didn't become ready within 15 min"
  return 1
}

_run_stage() {
  local candidate_key="$1" stage="$2"

  case "${candidate_key}" in
    cell_f)               local cname="qwen3_30b_a3b_instruct_2507" ;;
    deepseek_v2_lite)     local cname="deepseek_v2_lite_chat" ;;
    gemma_4_26b_a4b)      local cname="gemma_4_26b_a4b" ;;
    llama_3_3_70b_nvfp4)  local cname="llama_3_3_70b_nvfp4" ;;
    qwen3_5_35b_a3b)      local cname="qwen3_5_35b_a3b" ;;
    moonlight_16b_a3b)    local cname="moonlight_16b_a3b" ;;
    ministral_3_14b)      local cname="ministral_3_14b" ;;
    *) _log "  unknown candidate_key: ${candidate_key}"; return 1 ;;
  esac

  local config_path="${REPO_ROOT}/data/eval/configs/${stage}_autoresearch_prompt_vllm_${cname}_dev_round3_v1.yaml"
  if [[ ! -f "${config_path}" ]]; then
    _log "  config missing: ${config_path}"
    printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\n' "${candidate_key}" "${stage}" 0 "n/a" "n/a" "n/a" "no_config" >> "${RUNS_TSV}"
    return 1
  fi

  local config_id
  config_id="$("${REPO_ROOT}/.venv/bin/python" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['id'])" "${config_path}")"
  local src_run_dir="${REPO_ROOT}/data/eval/runs/${config_id}"
  local dst_run_dir="${OUT_DIR}/${candidate_key}/${stage}"
  mkdir -p "$(dirname "${dst_run_dir}")"

  local stage_log="${LOGS_DIR}/${candidate_key}_${stage}.log"
  local start_s=$(date +%s)
  PYTHONPATH="${REPO_ROOT}" "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/scripts/eval/experiment/run_experiment.py" \
    "${config_path}" \
    --vllm-base-url "http://${DGX_HOST}:8003/v1" \
    --dry-run --force >"${stage_log}" 2>&1
  local rc=$?
  local wall=$(( $(date +%s) - start_s ))

  if [[ ${rc} -ne 0 ]]; then
    _log "  ${stage} run failed (exit ${rc}); tail of log:"
    tail -5 "${stage_log}"
    printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\n' "${candidate_key}" "${stage}" "${wall}" "n/a" "n/a" "${dst_run_dir}" "fail" >> "${RUNS_TSV}"
    return 1
  fi

  if [[ -d "${src_run_dir}" ]]; then
    rm -rf "${dst_run_dir}"
    mv "${src_run_dir}" "${dst_run_dir}"
  fi

  local silver_id score_script score_value score_metric
  if [[ "${stage}" == "gi" ]]; then
    silver_id="silver_opus47_gi_dev_v1"
    score_script="${REPO_ROOT}/scripts/eval/score/score_gi_insight_to_insight.py"
  else
    silver_id="silver_opus47_kg_dev_v1"
    score_script="${REPO_ROOT}/scripts/eval/score/score_kg_topic_coverage.py"
  fi

  rm -rf "${REPO_ROOT}/data/eval/runs/${config_id}"
  ln -s "${dst_run_dir}" "${REPO_ROOT}/data/eval/runs/${config_id}"
  local score_out
  score_out=$("${REPO_ROOT}/.venv/bin/python" "${score_script}" \
    --run-id "${config_id}" \
    --silver "${silver_id}" \
    --dataset "curated_5feeds_dev_v1" 2>&1 || true)
  rm -f "${REPO_ROOT}/data/eval/runs/${config_id}"

  if [[ "${stage}" == "gi" ]]; then
    score_value=$(echo "${score_out}" | grep -oE 'AVG MAX SIMILARITY: [0-9.]+' | awk '{print $NF}')
    [[ -z "${score_value}" ]] && score_value="parse_error"
    score_metric="gi_avg_max_sim"
  else
    local topic_cov entity_cov
    topic_cov=$(echo "${score_out}" | grep -oE 'TOPICS:.*[0-9]+%' | grep -oE '[0-9]+%' | head -1 | tr -d '%')
    entity_cov=$(echo "${score_out}" | grep -oE 'ENTITIES:.*[0-9]+%' | grep -oE '[0-9]+%' | head -1 | tr -d '%')
    score_value="topic=${topic_cov:-na}% entity=${entity_cov:-na}%"
    score_metric="kg_topic_entity_cov"
  fi

  printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\n' "${candidate_key}" "${stage}" "${wall}" "${score_metric}" "${score_value}" "${dst_run_dir}" "ok" >> "${RUNS_TSV}"
  _log "  ${stage} ok (${wall}s)  ${score_metric}=${score_value}"
}

# ---------- main ----------

if ! curl -sf --connect-timeout 5 "http://${DGX_HOST}:8003/health" >/dev/null 2>&1; then
  _log "vLLM not reachable at start. Aborting."
  exit 2
fi
_log "Sweep start — vLLM reachable. Cohort: ${#CANDIDATES[@]} candidates."

for line in "${CANDIDATES[@]}"; do
  read -r key model max_len extras <<< "${line}"
  if _should_skip "${key}"; then
    _log "[skip ${key}] requested via --skip"
    continue
  fi
  _log "[candidate ${key}] model=${model} max_len=${max_len} extras=${extras}"
  if [[ "${key}" != "cell_f" ]]; then
    _log "  swapping compose + restarting vLLM..."
    if ! _swap_compose "${model}" "${max_len}" "${extras}" >/dev/null 2>&1; then
      _log "  compose swap failed"
      printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\n' "${key}" "all" 0 "n/a" "n/a" "n/a" "swap_fail" >> "${RUNS_TSV}"
      continue
    fi
    _restart_vllm
    if ! _wait_for_vllm; then
      printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\n' "${key}" "all" 0 "n/a" "n/a" "n/a" "boot_fail" >> "${RUNS_TSV}"
      continue
    fi
    _log "  vLLM ready."
  fi
  _run_stage "${key}" "gi"
  _run_stage "${key}" "kg"
done

_log "[restore] swapping back to Cell F..."
_swap_compose "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4" "32768" "_" >/dev/null 2>&1 || true
_restart_vllm
_wait_for_vllm && _log "  Cell F restored + healthy." || _log "  WARN: Cell F restore not healthy."

_log "Sweep complete."
echo ""
echo "=== runs.tsv ==="
column -t -s $'\t' "${RUNS_TSV}" | tail -20
