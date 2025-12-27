# Scripts

Utility scripts for the podcast scraper project.

## eval_cleaning.py

Evaluates transcript cleaning quality by comparing raw vs cleaned transcripts.

### Usage

Evaluate all episodes with auto-generated filename:

```bash
python scripts/eval_cleaning.py
# Outputs to: results/cleaning_eval_YYYYMMDD_HHMMSS.json
```

Evaluate single episode:

```bash
python scripts/eval_cleaning.py --episode ep01
```

Specify custom output file:

```bash
python scripts/eval_cleaning.py --output results/my_cleaning_eval.json
```

### Options

- `--eval-dir`: Directory containing evaluation episodes (default: `data/eval`)
- `--episode`: Evaluate single episode only (e.g., `ep01`)
- `--output`: Output JSON file path (default: `results/cleaning_eval_<timestamp>.json`)
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

### What It Evaluates

For each episode, the script provides:

1. **Removal Statistics**:
   - Character removal (raw/cleaned/removal %)
   - Word removal (raw/cleaned/removal %)

2. **Sponsor/Ad Pattern Detection**:
   - Counts sponsor phrases before and after cleaning
   - Patterns checked: "this episode is brought to you by", "sponsored by", etc.
   - Shows removal effectiveness

3. **Brand Mention Detection**:
   - Counts mentions of common podcast sponsor brands (Figma, Stripe, Justworks, etc.)
   - Shows if brand mentions were removed

4. **Outro Pattern Detection**:
   - Counts outro patterns (subscribe, rate/review, newsletter, etc.)
   - Shows if outro content was removed

5. **Quality Flags**:
   - ⚠️ Too much removed (>60%): Flags if cleaning removed too much content
   - ❌ Cleaning ineffective: Flags if sponsor patterns weren't actually removed

6. **Diff Snippets**:
   - Shows what was removed (unified diff format)

### Output

The script generates a JSON file with:

- Aggregate statistics (average removal rates, pattern removal rates)
- Per-episode detailed results
- Flags for potential issues
- Diff snippets showing what was removed

---

## eval_summaries.py

Evaluates summarization quality using ROUGE metrics and reference-free checks.

### Installation (eval_summaries.py)

Requires the `rouge-score` library. Install with:

```bash
# Option 1: Install as part of dev dependencies (recommended)
pip install -e .[dev]

# Option 2: Install rouge-score directly
pip install rouge-score
```

### Usage (eval_summaries.py)

Use defaults (BART-large for MAP, LED/long-fast for REDUCE - same as app) with auto-generated filename:

```bash
python scripts/eval_summaries.py
# Outputs to: results/eval_YYYYMMDD_HHMMSS.json
```

Specify custom output file:

```bash
python scripts/eval_summaries.py --output results/my_evaluation.json
```

Specify MAP model only (REDUCE defaults to LED):

```bash
python scripts/eval_summaries.py --map-model bart-large
```

Specify both MAP and REDUCE models:

```bash
python scripts/eval_summaries.py \
    --map-model bart-large \
    --reduce-model long-fast
```

Using a config file (overrides CLI arguments):

```bash
python scripts/eval_summaries.py --config config.yaml
```

Use short reference summaries:

```bash
python scripts/eval_summaries.py --use-short-reference
```

### Options (eval_summaries.py)

- `--eval-dir`: Directory containing evaluation episodes (default: `data/eval`)
- `--map-model`: MAP model name/key (e.g., `bart-large`, `bart-small`,
  `pegasus`) or HuggingFace model ID. Defaults to `bart-large` (same as app
  default)
- `--reduce-model`: REDUCE model name/key (e.g., `long-fast`, `long`,
  `bart-large`) or HuggingFace model ID. Defaults to `long-fast` (LED-base,
  same as app default)
- `--model`: (Deprecated: use `--map-model`) Backward compatibility alias for `--map-model`
- `--config`: Path to config file (JSON or YAML) - overrides model arguments
- `--output`: Output JSON file path for results (default: `results/eval_<timestamp>.json`)
- `--device`: Device to use (`cuda`, `mps`, `cpu`, or `None` for auto)
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--use-short-reference`: Use `summary.gold.short.txt` instead of
  `summary.gold.long.txt` for ROUGE scoring

### Model Defaults

When models are not specified, the script uses the same defaults as the main application:

- **MAP model**: `bart-large` (BART-large CNN) - fast, efficient chunk
  summarization
- **REDUCE model**: `long-fast` (LED-base-16384) - accurate, long-context final summarization

This hybrid approach (BART for map, LED for reduce) is widely used in
production summarization systems.

### Output (eval_summaries.py)

The script generates a JSON file with:

- Model configuration (MAP and REDUCE models)
- Summary statistics (average generation time, compression ratio, keyword coverage)
- ROUGE scores (if reference summaries exist)
- Per-episode results with detailed metrics
- Check pass rates (compression, repetition, keyword coverage)

### Evaluation Dataset Structure

See `data/eval/README.md` for details on the evaluation dataset structure.

Each episode directory should contain:

- `transcript.raw.txt` - Raw transcript from Whisper (optional, for validation)
- `transcript.cleaned.txt` - Cleaned transcript (input for summarization)
- `summary.gold.long.txt` - Detailed human-written reference summary
  (default for ROUGE)
- `summary.gold.short.txt` - Optional concise reference summary
- `metadata.json` - Optional episode metadata

---

## setup_venv.sh Script

Creates a Python virtual environment and installs the package in editable mode.

### setup_venv.sh Usage

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

### What It Does

1. Creates `.venv/` virtual environment
2. Upgrades pip
3. Installs package in editable mode (`pip install -e .`)

### Next Steps

After running `setup_venv.sh`:

1. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

2. **Install development dependencies:**

   ```bash
   make init
   # Or manually: pip install -e .[dev,ml]
   ```

3. **Set up environment variables (optional, for OpenAI providers):**

   ```bash
   cp examples/.env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Run the CLI:**

   ```bash
   python -m podcast_scraper.cli <rss_url> [options]
   ```

See `README.md` and `CONTRIBUTING.md` for more details.

---

## fix_markdown.py

Automatically fixes common markdown linting issues that can be corrected programmatically.

### Usage

Fix all markdown files in the project:

```bash
python scripts/fix_markdown.py
```

Fix specific files:

```bash
python scripts/fix_markdown.py docs/TESTING_STRATEGY.md docs/rfc/RFC-020.md
```

Dry run (show what would be fixed without making changes):

```bash
python scripts/fix_markdown.py --dry-run
```

### What It Fixes

1. **Table Separator Formatting** (MD060):
   - Adds spaces around pipes in table separator rows
   - Converts `|-------------|--------|` to `| ----------- | ------ |`

2. **Trailing Spaces** (MD009):
   - Removes trailing whitespace from all lines

3. **Blank Lines Around Lists** (MD032):
   - Adds blank lines before and after lists when missing

4. **Code Block Language Specifiers** (MD040):
   - Adds language specifiers to code blocks when detectable (Python, JavaScript, etc.)

### When to Use

Run this script regularly, especially:

- Before committing markdown files
- After bulk edits to documentation
- When CI fails on markdown linting errors
- As part of your pre-commit workflow

### Integration with Pre-commit

The pre-commit hook runs `markdownlint` which catches issues, but doesn't auto-fix them. Use this script to fix issues before committing:

```bash
# Fix all markdown issues
python scripts/fix_markdown.py

# Then commit
git add -A
git commit -m "your message"
```

Or use markdownlint's auto-fix feature:

```bash
# Enable auto-fix in pre-commit hook
export MARKDOWNLINT_FIX=1
git commit -m "your message"
```

### Note

This script handles common, safe-to-fix issues. Some markdownlint errors (like content issues, heading levels, etc.) still need manual review.
