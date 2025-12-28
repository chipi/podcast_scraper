#!/usr/bin/env bash
set -euo pipefail

# Generate MP3s from transcripts using transcripts_to_mp3.py
# Input (optional): transcripts folder (default: ../transcripts relative to script)
# Output: ../audio/*.mp3 (handled by the Python script)
#
# Usage (run from tests/fixtures/scripts/ directory):
#   ./generate_audio.sh
#   ./generate_audio.sh ../transcripts
#   ./generate_audio.sh /absolute/path/to/transcripts

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/transcripts_to_mp3.py"

# Default transcripts directory: ../transcripts relative to script location
DEFAULT_TRANSCRIPTS_DIR="${SCRIPT_DIR}/../transcripts"
TRANSCRIPTS_DIR="${1:-${DEFAULT_TRANSCRIPTS_DIR}}"

# Resolve to absolute path if relative
if [[ ! "$TRANSCRIPTS_DIR" =~ ^/ ]]; then
    TRANSCRIPTS_DIR="$(cd "$(dirname "$TRANSCRIPTS_DIR")" && pwd)/$(basename "$TRANSCRIPTS_DIR")"
fi

python3 "${PYTHON_SCRIPT}" "${TRANSCRIPTS_DIR}" --overwrite