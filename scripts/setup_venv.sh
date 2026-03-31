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
SPACY_WHEELS_DIR="$PROJECT_DIR/wheels/spacy"
if compgen -G "$SPACY_WHEELS_DIR"/*.whl > /dev/null 2>&1; then
  export PIP_FIND_LINKS="$SPACY_WHEELS_DIR"
  echo "PIP_FIND_LINKS=$PIP_FIND_LINKS (using local wheels/spacy for pip)"
fi
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
   # If wheels/spacy/*.whl exists (from: make download-spacy-wheels), make init uses it
   # automatically — no need to export PIP_FIND_LINKS manually.
   # Or manually: pip install -e .[dev,ml,gemini]  (set PIP_FIND_LINKS to wheels/spacy if needed)

3. Set up environment variables (if using OpenAI providers):
   cp config/examples/.env.example .env
   # Edit .env and add your OPENAI_API_KEY

4. Run the CLI:
   python -m podcast_scraper.cli <rss_url> [options]

5. Ensure ffmpeg is installed (required for Whisper):
   # macOS: brew install ffmpeg
   # Ubuntu/Debian: sudo apt-get install ffmpeg
   # Windows: Download from https://ffmpeg.org/download.html

6. (Optional) To regenerate architecture diagrams locally: install Graphviz
   # macOS: brew install graphviz
   # Ubuntu/Debian: sudo apt-get install graphviz
   # Then: make visualize
   # CI does this automatically; only needed if you want to update docs/architecture/*.svg yourself.

For more information, see:
- README.md - Usage and installation
- CONTRIBUTING.md - Development setup
- docs/DEVELOPMENT_GUIDE.md - Development tips
INSTRUCTIONS

