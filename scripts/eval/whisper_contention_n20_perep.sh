#!/usr/bin/env bash
# Per-episode whisper sweep with a reliable Python-side timeout — used for
# #996 catastrophic-tail characterization at N=20.
#
# Why this exists instead of `whisper_contention_perep.sh`: the older script
# uses `perl -e 'alarm N; exec @ARGV'` which does NOT fire on macOS after the
# exec replaces perl with python (the SIGALRM scheduled before exec is lost
# on this platform). During the original #996 sweep two episodes hung — one
# for 2 hours — because the alarm never triggered. This script wraps every
# per-episode invocation in `subprocess.run(..., timeout=...)` from Python,
# which is portable and actually kills the child on deadline.
#
# Usage:
#   WHISPER_DGX_URL=http://your-dgx.tailnet.ts.net:8002/v1/audio/transcriptions \
#     scripts/eval/whisper_contention_n20_perep.sh <output_subdir>
#
# Env vars:
#   WHISPER_DGX_URL    (required)  whisper-openai transcriptions endpoint
#   EPISODES           (optional)  space-separated list. Default = the 20-episode
#                                  v2 sample used in #996.
#   PER_EPISODE_TIMEOUT_SEC (optional, default 1500)
#   AUDIO_DIR          (default tests/fixtures/audio/v2)
#   TRANSCRIPTS_DIR    (default tests/fixtures/transcripts/v2)
#
# Output:
#   data/eval/runs/whisper_contention_n20/<output_subdir>/perep/<ep>/metrics.json
#   data/eval/runs/whisper_contention_n20/<output_subdir>/metrics.json (aggregated)

set -euo pipefail

output_subdir="${1:?usage: $0 <output_subdir>}"
root="data/eval/runs/whisper_contention_n20/${output_subdir}"
mkdir -p "${root}/perep"

PER_EPISODE_TIMEOUT_SEC="${PER_EPISODE_TIMEOUT_SEC:-1500}"
AUDIO_DIR="${AUDIO_DIR:-tests/fixtures/audio/v2}"
TRANSCRIPTS_DIR="${TRANSCRIPTS_DIR:-tests/fixtures/transcripts/v2}"

# Default 20-episode sample — one episode per feed plus all e01-e03 from the
# top-5 feeds. Matches the #996 sweep so the data is comparable.
DEFAULT_EPISODES="p01_e01 p01_e02 p01_e03 \
p02_e01 p02_e02 p02_e03 \
p03_e01 p03_e02 p03_e03 \
p04_e01 p04_e02 p04_e03 \
p05_e01 p05_e02 p05_e03 \
p06_e01 p07_e01 p08_e01 p09_e01 p09_e02"

EPISODES="${EPISODES:-$DEFAULT_EPISODES}"

# Sanity check
if [[ -z "${WHISPER_DGX_URL:-}" ]]; then
    echo "ERROR: WHISPER_DGX_URL must be set." >&2
    echo "  e.g. WHISPER_DGX_URL=http://dgx-llm-1.tail6d0ed4.ts.net:8002/v1/audio/transcriptions" >&2
    exit 2
fi

for ep in $EPISODES; do
    out="${root}/perep/${ep}"
    mkdir -p "${out}"
    echo ">>> ep=${ep} $(date -u +%FT%TZ)" >&2

    # Run via Python wrapper so the timeout actually fires. The harness ignores
    # SIGALRM but `subprocess.run(..., timeout=...)` kills the child on deadline
    # cleanly. Captures the timeout case as a JSON metrics file the aggregator
    # can pick up — otherwise a hung episode leaves an empty dir and shows up
    # as "NO METRICS" in the aggregate.
    PYTHONPATH=. .venv/bin/python -c "
import json
import os
import subprocess
import sys
import time

ep = '${ep}'
out = '${out}'
audio_dir = '${AUDIO_DIR}'
transcripts_dir = '${TRANSCRIPTS_DIR}'
timeout_s = ${PER_EPISODE_TIMEOUT_SEC}

cmd = [
    '.venv/bin/python', 'scripts/eval/score/whisper_dgx_vs_cloud_v1.py',
    '--backend', 'dgx',
    '--models', 'large-v3',
    '--audio-dir', audio_dir,
    '--transcripts-dir', transcripts_dir,
    '--episodes', ep,
    '--output', out,
]
t0 = time.time()
try:
    subprocess.run(cmd, timeout=timeout_s, check=False)
    sys.exit(0)
except subprocess.TimeoutExpired:
    elapsed = time.time() - t0
    sys.stderr.write(f'!!! {ep} TIMEOUT after {elapsed:.0f}s (limit {timeout_s}s)\n')
    # Write a metrics.json the aggregator can read so the sweep summary
    # records this episode as a catastrophic-tail case rather than NO METRICS.
    record = {
        'schema': 'metrics_whisper_dgx_vs_cloud_v1',
        'summary': [{
            'backend': 'dgx', 'model': 'large-v3', 'episodes': 1,
            'mean_wer': None, 'max_wer': None, 'min_wer': None,
            'mean_elapsed_s': None, 'stdev_elapsed_s': None,
            'mean_realtime_multiple': None, 'total_audio_seconds': 0,
            'total_cost_usd': 0.0,
        }],
        'rows': [{
            'episode_id': ep, 'backend': 'dgx', 'model': 'large-v3',
            'wer': None, 'elapsed_s': elapsed,
            'realtime_multiple': None, 'cost_usd': 0.0,
            'error': f'TIMEOUT after {elapsed:.0f}s (limit {timeout_s}s)',
        }],
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(record, f, indent=2)
    sys.exit(0)
" || echo "!!! ${ep} FAILED (non-timeout error)" >&2
done

echo ">>> aggregating ${root}/metrics.json" >&2
.venv/bin/python scripts/eval/aggregate_whisper_perep.py "${root}/perep" "${root}/metrics.json"
echo ">>> done: ${root}/metrics.json" >&2
