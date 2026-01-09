#!/usr/bin/env bash
set -euo pipefail

# Generate MP3s from transcripts using transcripts_to_mp3.py
# Input (optional): transcript files or folder (default: ../transcripts relative to script)
# Output: ../audio/*.mp3 (handled by the Python script)
#
# Usage (run from tests/fixtures/scripts/ directory):
#   ./generate_audio.sh                                    # Generate all transcripts
#   ./generate_audio.sh ../transcripts                     # Generate all in folder
#   ./generate_audio.sh ../transcripts/p07_e01.txt         # Generate single file
#   ./generate_audio.sh ../transcripts/p07_e01.txt ../transcripts/p08_e01.txt  # Multiple files

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/transcripts_to_mp3.py"

# Default transcripts directory: ../transcripts relative to script location
DEFAULT_TRANSCRIPTS_DIR="${SCRIPT_DIR}/../transcripts"

# If no arguments, use default directory
if [ $# -eq 0 ]; then
    python3 "${PYTHON_SCRIPT}" "${DEFAULT_TRANSCRIPTS_DIR}" --overwrite
else
    # Pass all arguments to the Python script
    python3 "${PYTHON_SCRIPT}" "$@" --overwrite
fi
