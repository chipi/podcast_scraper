#!/usr/bin/env bash
# Per-episode whisper sweep with Tailscale-hang timeout workaround.
# Used for #963 contention re-test: runs each episode as an independent
# harness invocation so a hang only kills one episode, not the sweep.
#
# Usage:
#   WHISPER_DGX_URL=http://your-dgx.tailnet.ts.net:8002/v1/audio/transcriptions \
#     scripts/eval/whisper_contention_perep.sh <scenario_subdir>
#
# Output:
#   data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-openai-contention/<scenario_subdir>/perep/<ep>/metrics.json
#   data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-openai-contention/<scenario_subdir>/metrics.json (aggregated)
set -euo pipefail

scenario="${1:?usage: $0 <scenario_subdir>}"
root="data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-openai-contention/${scenario}"
mkdir -p "${root}/perep"

EPISODES=(p01_e01 p02_e01 p03_e01 p04_e01 p05_e01)

for ep in "${EPISODES[@]}"; do
  out="${root}/perep/${ep}"
  mkdir -p "${out}"
  echo ">>> scenario=${scenario} ep=${ep} $(date -u +%FT%TZ)"
  perl -e 'alarm 1200; exec @ARGV' -- \
    .venv/bin/python scripts/eval/score/whisper_dgx_vs_cloud_v1.py \
      --backend dgx \
      --models large-v3 \
      --audio-dir tests/fixtures/audio/v2 \
      --transcripts-dir tests/fixtures/transcripts/v2 \
      --episodes "${ep}" \
      --output "${out}" \
    || echo "!!! ${ep} FAILED (timeout or error)"
done

echo ">>> aggregating ${root}/metrics.json"
.venv/bin/python scripts/eval/aggregate_whisper_perep.py "${root}/perep" "${root}/metrics.json"
echo ">>> done: ${root}/metrics.json"
