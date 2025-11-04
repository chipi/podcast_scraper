#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "Venv created at: $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"
echo "To run: $VENV_DIR/bin/python $PROJECT_DIR/podcast_scraper.py <rss_url> [options]"

