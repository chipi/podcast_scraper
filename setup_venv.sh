#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$PROJECT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -e "$REPO_ROOT"

cat <<'INSTRUCTIONS'
Venv created.
To activate: source podcast_scraper/.venv/bin/activate
To run CLI: python -m podcast_scraper.cli <rss_url> [options]
(Optional) With Whisper extras: pip install podcast-scraper[whisper]
INSTRUCTIONS

