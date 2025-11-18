#!/usr/bin/env bash
set -euo pipefail

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -e "$PROJECT_DIR"

cat <<'INSTRUCTIONS'
Venv created.
To activate: source .venv/bin/activate
To install package (from this directory): pip install -e .
To run CLI: python -m podcast_scraper.cli <rss_url> [options]
Ensure ffmpeg is installed (e.g. brew install ffmpeg).
INSTRUCTIONS

