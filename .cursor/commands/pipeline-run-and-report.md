# Pipeline run + self-serve post-mortem

Use this for a **single-turn** workflow: you run the pipeline, read artifacts yourself, and finish with a concise report—do not ask me to paste logs unless something is truly blocked (secrets, missing tool, or ambiguous paths).

## What to do

1. **Run** (pick one and substitute paths; use Makefile targets from this repo, not raw pytest):
   - Acceptance: `make test-acceptance` with the config this session uses, **or**
   - Manual config: `python -m podcast_scraper.cli --config <path-to-yaml>`  
   Run in the **foreground**, wait for exit, show terminal output in your summary if useful.

2. **Resolve output directory** from the config / env actually used (e.g. `output_dir` in YAML, or `SERVE_OUTPUT_DIR` / acceptance temp dir). If unclear, infer from the run output or ask **one** clarifying question.

3. **Read** (as files under that corpus/output root, when present):
   - `metrics.json`, `run.json`, `fingerprint.json`
   - `.pipeline_status.json`, `.monitor.log` (monitor / non-TTY runs)
   - `pipeline.log` if configured as a relative `log_file`
   - Any failing test logs or CI-style output from the command you ran

4. **Finish** with:
   - Pass/fail (or partial) in one line
   - **3–7 bullets**: timing hotspots, RSS/CPU if relevant, errors, config surprises
   - If something failed: **root cause** → **minimal fix** → re-run the same command until green **or** stop with a clear blocker

5. Follow **`.cursorrules`**: no background `make`/`pytest`; use **`make format` / `make lint`** if you touch Python.

## Placeholders (fill before or at start of run)

- Config: `________________`
- Expected `output_dir` or acceptance artifact root: `________________`
