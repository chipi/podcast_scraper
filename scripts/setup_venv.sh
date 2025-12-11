#!/usr/bin/env bash
set -euo pipefail

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "Installing package in editable mode..."
"$VENV_DIR/bin/pip" install -e "$PROJECT_DIR"

echo ""
echo "✅ Virtual environment created successfully!"
echo ""
cat <<'INSTRUCTIONS'
Next steps:

1. Activate the virtual environment:
   source .venv/bin/activate

2. Install development dependencies (recommended):
   make init
   # Or manually: pip install -e .[dev,ml]

3. Set up environment variables (if using OpenAI providers):
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY

4. Run the CLI:
   python -m podcast_scraper.cli <rss_url> [options]

5. Ensure ffmpeg is installed (required for Whisper):
   # macOS: brew install ffmpeg
   # Ubuntu/Debian: sudo apt-get install ffmpeg
   # Windows: Download from https://ffmpeg.org/download.html

For more information, see:
- README.md - Usage and installation
- CONTRIBUTING.md - Development setup
- docs/DEVELOPMENT_NOTES.md - Development tips
INSTRUCTIONS

