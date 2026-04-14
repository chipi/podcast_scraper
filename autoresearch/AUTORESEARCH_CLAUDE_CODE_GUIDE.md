# Running AutoResearch Overnight with Claude Code

A practical guide for running autonomous autoresearch optimization loops on your M4 Pro
laptop, specific to the podcast_scraper prompt tuning setup.

> **Framework: v2** ([RFC-073](../docs/rfc/RFC-073-autoresearch-v2-framework.md)). Iteration
> happens on `curated_5feeds_dev_v1` (10 ep); champions are validated on
> `curated_5feeds_benchmark_v2` (5 held-out ep). **Never iterate against held-out.** All v2
> configs set `params.seed: 42`. Original v1 smoke/benchmark configs are preserved but frozen —
> don't use them for new work unless you're specifically reproducing a prior result.

---

## How It Works

Claude Code drives the autoresearch loop. It reads `program.md`, mutates the target
template, runs `make autoresearch-score`, checks the scalar, commits improvements or
reverts failures, logs to `results.tsv`, and repeats — with no human in the loop.

The three files that define every experiment:

| File | Role | Editable by agent? |
| --- | --- | --- |
| `autoresearch/prompt_tuning/program_summary_bullets.md` | Agent instructions and loop rules (bullets) | Human only |
| `autoresearch/prompt_tuning/program_summary.md` | Agent instructions and loop rules (paragraphs) | Human only |
| `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2` | What the agent optimizes (bullets) | Agent only |
| `autoresearch/prompt_tuning/eval/score.py` | Immutable scoring harness | Never |

---

## Prerequisites

### 1. Install Claude Code CLI

```bash
npm install -g @anthropic/claude-code
claude --version  # verify
```

### 2. Environment file

All API keys must be available without interactive prompts. The score script loads
`.env` from repo root, then `.env.autoresearch` (overrides). Minimum required:

```bash
# In .env or .env.autoresearch
AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1        # lets score.py use OPENAI_API_KEY / ANTHROPIC_API_KEY
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Tuning knobs
AUTORESEARCH_EVAL_N=10                       # episodes per run (v2 dev has 10; set to 10 for full dev)
AUTORESEARCH_SCORE_ROUGE_WEIGHT=0.70         # default; don't change between experiments
```

> **N=10 is the v2 default for dev iteration.** Use N=5 only if you're specifically reproducing
> a v1 smoke-scale run (v1 smoke dataset had 5 episodes). For held-out validation
> (`curated_5feeds_benchmark_v2`), use N=5 (its size). N=1 scores are too noisy for a reliable
> ratchet — we saw 0.660 vs actual 0.571 baseline because of this in the first session.

### 3. Python environment activated

```bash
source .venv/bin/activate
python -c "import podcast_scraper; print('ok')"
```

### 4. Preflight check

No `make preflight` target exists yet. **Add it before the first overnight run:**

```makefile
autoresearch-preflight:
 @echo "=== AutoResearch Preflight ==="
 @test -f .env || (echo "ERROR: .env missing" && exit 1)
 @python -c "import openai, anthropic" || (echo "ERROR: providers not installed" && exit 1)
 @test -f data/eval/configs/autoresearch_prompt_openai_smoke_bullets_v1.yaml \
  || (echo "ERROR: experiment config missing" && exit 1)
 @test -d data/eval/references/silver/silver_gpt4o_smoke_bullets_v1 \
  || (echo "ERROR: silver reference missing" && exit 1)
 @make autoresearch-score DRY_RUN=1 > /dev/null \
  && echo "DRY_RUN: ok" || (echo "ERROR: dry-run failed" && exit 1)
 @echo "=== All checks passed ==="
```

---

## program.md Contract

This file is the only thing Claude reads before starting. Write it carefully — it is
your only lever once the loop is running.

### podcast_scraper-specific template

````markdown
# AutoResearch Track A — Bullet Summarization Prompt Tuning

## Objective
Improve `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2`.
Metric: scalar from `make autoresearch-score` (higher = better, range ~0.5–1.0).
Current best is the last committed score. Baseline is logged in `results.tsv`.

## Setup (run once before loop)
1. Run `make autoresearch-preflight` — abort immediately if anything fails.
2. Run `make autoresearch-score DRY_RUN=1` — confirm a scalar prints to stdout.
3. Confirm `autoresearch/prompt_tuning/results_summary_bullets.tsv` has a baseline row.

## Experiment Loop
For each experiment 1 to $AUTORESEARCH_MAX_EXPERIMENTS:
1. Form a one-sentence hypothesis about which dimension to improve
   (coverage, fidelity, conciseness, or format — see `eval/rubric.md`).
2. Edit ONLY `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2`.
3. Run `make autoresearch-score`.
4. Read the single float on stdout. Compare to current best.
5. Delta > +1%: run `git add <template> autoresearch/prompt_tuning/results_summary_bullets.tsv`
   then `git commit -m "[autoresearch] exp-N: <hypothesis>, score X.XXX (+Y.Y%)"`.
6. Delta ≤ +1%: run `git checkout HEAD -- <template>` to restore.
7. Append one row to `autoresearch/prompt_tuning/results_summary_bullets.tsv` (see column format below).
8. If 3 consecutive experiments all ≤ +1%: stop early and write summary.
9. Start next experiment.

## Boundaries — never cross these
- Edit ONLY `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2`.
- NEVER edit `autoresearch/prompt_tuning/eval/score.py` or anything under `data/eval/`.
- NEVER edit `autoresearch/prompt_tuning/program_summary_bullets.md`.
- NEVER install new packages.
- NEVER commit without running the full eval first.
- NEVER pause to ask the human a question. If uncertain, make a conservative choice,
  log it in the results notes column, and continue.
- Stop after $AUTORESEARCH_MAX_EXPERIMENTS experiments or 3 consecutive fails.

## Allowed Commands
```bash
make autoresearch-score
make autoresearch-score DRY_RUN=1
git diff src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2
git add src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2 \
        autoresearch/prompt_tuning/results_summary_bullets.tsv
git commit -m "..."
git checkout HEAD -- src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2
git log --oneline -20
```

## results.tsv Column Format
Tab-separated. Append one row per experiment:
```
experiment_id  score     delta     status    notes                        judge_a_model  judge_b_model  rubric_hash  eval_dataset_ref
exp-7          0.623456  +0.01234  accepted  hypothesis text here         gpt-4o-mini    claude-haiku-4-5  9f43a4b9  curated_5feeds_smoke_v1
```
Status values: `baseline`, `accepted`, `rejected`, `error`.

## On Completion
Write a summary to `autoresearch/prompt_tuning/summary_<YYYY-MM-DD>.md` with:
- Total experiments run
- Best score vs baseline (absolute and % improvement)
- Top changes that improved the score with their deltas
- Patterns: what consistently helped vs hurt
- Suggested next directions
- Any crashes or anomalies

Print "AUTORESEARCH COMPLETE" as the final stdout line.
````

> **Important:** Use `git checkout HEAD -- <file>` to restore on failure, NOT
> `git reset --hard HEAD`. `git reset` resets ALL tracked files including `results.tsv`,
> losing the experiment log. `git checkout HEAD -- <file>` restores only the template.

---

## Running the Loop

### Step 1 — Watch it manually first (always)

Never start an overnight run blind. Watch 3–5 experiments with full output visible:

```bash
# For bullets line:
claude --dangerously-skip-permissions \
  "Read autoresearch/prompt_tuning/program_summary_bullets.md and run exactly 3 experiments, then stop and summarize."

# For paragraph summary line:
claude --dangerously-skip-permissions \
  "Read autoresearch/prompt_tuning/program_summary.md and run exactly 3 experiments, then stop and summarize."
```

Verify it:

- Reads `program.md` without asking questions
- Mutates only `bullets_json_v1.j2`
- Runs `make autoresearch-score` and reads the scalar correctly
- Commits on >1% improvement, restores on fail
- Logs to `results.tsv` correctly

Only proceed to overnight once you have seen this cycle work cleanly.

### Step 2 — Overnight run

```bash
caffeinate -i claude --dangerously-skip-permissions \
  "Read autoresearch/prompt_tuning/program_summary_bullets.md and begin the autoresearch loop."
```

`caffeinate -i` prevents macOS idle sleep for the duration without keeping the display
on. The run ends when Claude hits the experiment cap, triggers early stop, or runs out
of ideas.

### Step 3 — Overnight run with logging (recommended)

```bash
mkdir -p autoresearch/logs

nohup caffeinate -i claude --dangerously-skip-permissions \
  "Read autoresearch/prompt_tuning/program_summary_bullets.md and begin the autoresearch loop." \
  > autoresearch/logs/run_$(date +%Y%m%d_%H%M).log 2>&1 &

echo "AutoResearch started. PID: $!"
echo "Tail logs: tail -f autoresearch/logs/run_$(date +%Y%m%d_%H%M).log"
```

`nohup` ensures the process survives terminal close. You can shut the lid — the process
continues. Add `autoresearch/logs/` to `.gitignore`.

### Step 4 — Morning review

```bash
# What was committed overnight
git log --oneline --since="10 hours ago"

# Top scoring experiments
sort -t$'\t' -k2 -rn autoresearch/prompt_tuning/results.tsv | head -10

# Session summary (if Claude wrote one)
cat autoresearch/prompt_tuning/summary_$(date +%Y-%m-%d).md

# Last 100 lines of log
tail -100 autoresearch/logs/run_*.log | grep -E "exp-|SCORE|EARLY STOP|COMPLETE"
```

If results look good, open a PR from the `autoresearch` branch.

---

## Improving Score Reliability

### Use N=5 minimum

Our smoke dataset has 5 episodes. Always run with N=5 for overnight:

```bash
AUTORESEARCH_EVAL_N=5  # in .env — this is the default
```

N=1 scores vary ±10% run-to-run. We logged 0.660 as baseline (N=1) and the real
N=5 baseline was 0.571 — 15% off. Never compare N=1 and N=5 scores.

### DRY_RUN for fast iteration

When debugging a loop or testing if `program.md` reads correctly, use DRY_RUN to
skip API judge calls (ROUGE-only, ~5 seconds instead of ~30):

```bash
make autoresearch-score DRY_RUN=1
```

Note: DRY_RUN score is not comparable to full score (no judge component). Use only
for debugging, not for experiment decisions.

### Ratchet mechanics

| Step | Command | What it touches |
| --- | --- | --- |
| Accept (win) | `git add <template> results.tsv && git commit` | Both files committed |
| Reject (loss) | `git checkout HEAD -- <template>` | Template only restored; results.tsv safe |
| Log rejected run | Write row to results.tsv manually after restore | results.tsv updated |

---

## Phase 2: Multi-Tier Judge Pipeline (not yet implemented)

Once the basic loop is validated, replace the current dual-judge setup with a
three-tier pipeline to reduce overnight cost:

```text
Experiment output
      │
      ▼
Tier 1: Local Ollama (Qwen 2.5 72B — 48GB RAM, ~30s/call, free)
      │  if delta ≤ 0.02 → reject immediately, skip Tier 2/3
      ▼
Tier 2: gpt-4o-mini (current judge_a — cheap, consistent)
      │  if Tier 1 and Tier 2 disagree (gap > 0.15) → treat as no-improvement
      ▼
Tier 3: gpt-4o + claude-opus-4-6 averaged
         definitive score for genuine candidates only
```

**Cost comparison for 50-experiment overnight run:**

| Pipeline | Cost |
| --- | --- |
| Current (gpt-4o-mini + claude-haiku, N=5) | ~$2–4 |
| Tier 1 only (local Ollama) | ~$0 |
| Full 3-tier | ~$5–10 |
| Tier 3 only (flagship judges) | ~$40–80 |

**Before enabling Tier 1**, verify local model calibration against gpt-4o-mini on
20 known outputs. Target Pearson r > 0.75. Below that, skip Tier 1 and use Tier 2
as the first filter.

---

## Git Conventions

```bash
# Branch naming (already set)
autoresearch

# Commit messages (agent writes these)
[autoresearch] exp-7: <hypothesis>, score 0.623 (+2.1%)

# Results file
autoresearch/prompt_tuning/results.tsv       # one row per experiment
autoresearch/prompt_tuning/summary_YYYY-MM-DD.md  # written by agent at end of run

# Logs (gitignored)
autoresearch/logs/run_YYYYMMDD_HHMM.log
```

---

## Troubleshooting

**Agent stops and asks questions mid-run**

Add to `program.md` boundaries:

```markdown
NEVER pause to ask the human a question. If uncertain, make a conservative choice,
log it in the notes column, and continue.
```

**Score is noisy between runs**

Check `AUTORESEARCH_EVAL_N`. Must be 5 for reproducible scores. N=1 is only for
interactive debugging.

**`git reset --hard HEAD` wiping results.tsv**

Use `git checkout HEAD -- <template_file>` instead. Only restores the template,
leaves results.tsv untouched.

**Run completes but no summary file**

Add explicitly to `program.md`: "Before printing AUTORESEARCH COMPLETE, write a
summary to `autoresearch/prompt_tuning/summary_<YYYY-MM-DD>.md`."

**Agent edits files it shouldn't**

Name the exact file path in program.md boundaries. Claude Code respects explicit
file path constraints reliably.

**`caffeinate` exits before run completes**

Use `nohup` in addition to `caffeinate` (see Step 3 above).

---

## Remaining TODO Before First Overnight Run

- [ ] Add `make autoresearch-preflight` to Makefile (template above)
- [ ] Add `autoresearch/logs/` to `.gitignore`
- [ ] Set `AUTORESEARCH_MAX_EXPERIMENTS` handling in `program.md` (currently Claude
      reads the value from program.md directly, not from env)
- [ ] Watch 3 experiments manually via CLI first (Step 1)
- [ ] Verify `make autoresearch-score DRY_RUN=1` completes cleanly from a fresh shell

---

## Later: Production Scheduling on Mac Mini

Once the loop is validated on the M4 Pro, the Mac Mini becomes the always-on
orchestrator. This is Phase 3 — don't think about it until the loop works reliably
overnight on the laptop.

A separate `PRODUCTION_SCHEDULING.md` will cover nightly cron, ground-truth
collection, and auto-reoptimize triggers.
