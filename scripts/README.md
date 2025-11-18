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

Requires the `rouge-score` library:

```bash
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
- `--map-model`: MAP model name/key (e.g., `bart-large`, `bart-small`, `pegasus`) or HuggingFace model ID. Defaults to `bart-large` (same as app default)
- `--reduce-model`: REDUCE model name/key (e.g., `long-fast`, `long`, `bart-large`) or HuggingFace model ID. Defaults to `long-fast` (LED-base, same as app default)
- `--model`: (Deprecated: use `--map-model`) Backward compatibility alias for `--map-model`
- `--config`: Path to config file (JSON or YAML) - overrides model arguments
- `--output`: Output JSON file path for results (default: `results/eval_<timestamp>.json`)
- `--device`: Device to use (`cuda`, `mps`, `cpu`, or `None` for auto)
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--use-short-reference`: Use `summary.gold.short.txt` instead of `summary.gold.long.txt` for ROUGE scoring

### Model Defaults

When models are not specified, the script uses the same defaults as the main application:

- **MAP model**: `bart-large` (BART-large CNN) - fast, efficient chunk summarization
- **REDUCE model**: `long-fast` (LED-base-16384) - accurate, long-context final summarization

This hybrid approach (BART for map, LED for reduce) is widely used in production summarization systems.

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
- `summary.gold.long.txt` - Detailed human-written reference summary (default for ROUGE)
- `summary.gold.short.txt` - Optional concise reference summary
- `metadata.json` - Optional episode metadata
