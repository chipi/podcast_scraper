#!/usr/bin/env bash
set -euo pipefail

# Generate MP3s from transcripts using scripts/transcripts_to_mp3.py
# Input (optional): transcripts folder (default: tests/fixtures/transcripts)
# Output: ./fixtures/audio/*.mp3 (handled by the Python script)
#
# Usage:
#   ./scripts/generate_audio.sh
#   ./scripts/generate_audio.sh tests/fixtures/transcripts

TRANSCRIPTS_DIR="${1:-tests/fixtures/transcripts}"

python3 "$(dirname "$0")/transcripts_to_mp3.py" "${TRANSCRIPTS_DIR}" --overwrite