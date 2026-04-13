# AutoResearch — OpenAI Bundled Prompt Tuning

## Objective

Improve the OpenAI-specific bundled summarization prompts:

- `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_system_v1.j2`
- `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_user_v1.j2`

Metric: scalar from `make autoresearch-score-bundled` (higher = better, range ~0.5–1.0).
Current best is the last committed score. Baseline is the first row in `results/results_openai_r1.tsv`.

The bundled path performs cleaning + summarization in a single LLM call (no separate cleaning
step). Both templates must stay internally consistent — the system prompt defines the job,
the user prompt delivers the transcript and output rules.

## Round 2 — Priority Hypotheses (try in this order)

r1 best: **0.474687** (ROUGE-L 27.3%, embed 76.7%, judge_mean 0.946).

1. **Paragraph count: fix at exactly 2** — current cap is 2–3; try forcing exactly 2 paragraphs.
   Hypothesis: tighter output closes the gap with the silver reference style further.
2. **Paragraph count: fix at exactly 3** — if exp above rejects, try the other direction.
3. **Bullet count: reduce to 5–6** — current target is 6–8; silver reference may use fewer.
   Hypothesis: fewer, denser bullets improve ROUGE-L and judge score.
4. **Bullet lead instruction** — require each bullet to open with a named concept, tool, or
   finding from the transcript (not a generic observation).
5. **Embed push: transcript-phrasing anchor** — add instruction to prefer phrasing close to the
   transcript's own wording in the summary (not just bullets).

## Round 3 — Priority Hypotheses (try in this order)

r2 best: **0.474687** (all r2 experiments rejected; champion unchanged from r1 exp-4).

Silver reference style (observed from `silver_sonnet46_smoke_bullets_v1`): dense noun-anchored
sentences, em-dash or semicolon elaboration, "X rather than Y" contrasts, specific concept/finding
as grammatical subject, no generic openers. r3 focuses on closing the style gap.

1. **Few-shot style examples** — add 3 silver-quality example bullets to the system prompt as a
   target style anchor. Hypothesis: showing the model the exact style closes the vocabulary and
   structural gap with the silver reference more directly than instructions alone.
2. **Anchor lead + ban generic openers** — require each bullet to open with the specific
   concept/technique/tool/finding as its grammatical subject; explicitly ban openers like
   "The episode", "Speakers", "The host", "One key". Hypothesis: enforces the noun-anchor pattern
   observed in the silver.
3. **Explicit style narration** — add prose description: noun-heavy, prefer "X rather than Y"
   contrasts, use em-dash for precision, no filler phrases. Hypothesis: meta-description of target
   style helps when examples alone are insufficient.
4. **Model upgrade: gpt-4o** — change model from gpt-4o-mini to gpt-4o in the experiment YAML.
   Hypothesis: gpt-4o produces output closer to Sonnet-4.6 silver style, potentially closing 5–8pp
   ROUGE-L gap that is a model-quality ceiling not a prompt ceiling.
   Exception: this experiment requires editing the YAML at
   `data/eval/configs/summarization_bullets/autoresearch_prompt_openai_bundled_smoke_bullets_v1.yaml`
   (model field only). Restore YAML if rejected; commit alongside prompt files if accepted.
5. **Separate task framing** — restructure user prompt into numbered Task 1 (title), Task 2
   (summary), Task 3 (bullets) sections. Hypothesis: explicit task decomposition helps gpt-4o-mini
   budget attention and produce tighter per-section output.

After these, if none land, try combinations of the winners from r1 + r2.

## Setup (run once before loop)

1. Run `make autoresearch-score-bundled DRY_RUN=1` — confirm a scalar prints to stdout.
2. Confirm `autoresearch/bundled_prompt_tuning/results/results_openai_r1.tsv` gets a baseline row.

## Experiment Loop

For each experiment 1 to $AUTORESEARCH_MAX_EXPERIMENTS:

1. Form a one-sentence hypothesis about which dimension to improve
   (cleaning quality, coverage, fidelity, bullet conciseness, or JSON format compliance).
2. Edit ONLY the two bundled templates listed above. Both files may be edited in one
   experiment if the hypothesis spans system + user, but treat it as one atomic change.
3. Run `make autoresearch-score-bundled`.
4. Read the single float on stdout. Compare to current best.
5. Delta > +1%: run

   ```bash
   git add src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_system_v1.j2 \
           src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_user_v1.j2 \
           autoresearch/bundled_prompt_tuning/results/results_openai_r1.tsv
   git commit -m "[autoresearch-bundled] exp-N: <hypothesis>, score X.XXX (+Y.Y%)"
   ```

6. Delta ≤ +1%: restore with

   ```bash
   git checkout HEAD -- src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_system_v1.j2
   git checkout HEAD -- src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_user_v1.j2
   ```

7. Append one row to `autoresearch/bundled_prompt_tuning/results/results_openai_r1.tsv`.
8. If 3 consecutive experiments all ≤ +1%: stop early and write summary.
9. Start next experiment.

## Boundaries — never cross these

- Edit ONLY the two bundled templates listed above.
- NEVER edit `autoresearch/bundled_prompt_tuning/eval/score.py` or anything under `data/eval/`.
- NEVER edit `autoresearch/bundled_prompt_tuning/program_openai_bundled.md`.
- NEVER edit the shared prompts at `src/podcast_scraper/prompts/shared/summarization/`.
- NEVER install new packages.
- NEVER commit without running the full eval first.
- NEVER pause to ask the human a question. If uncertain, make a conservative choice,
  log it in the results notes column, and continue.
- Stop after $AUTORESEARCH_MAX_EXPERIMENTS experiments or 3 consecutive fails.

## Allowed Commands

```bash
make autoresearch-score-bundled
make autoresearch-score-bundled DRY_RUN=1
git diff src/podcast_scraper/prompts/openai/summarization/
git add src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_system_v1.j2 \
        src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_user_v1.j2 \
        autoresearch/bundled_prompt_tuning/results/results_openai_r1.tsv
git commit -m "..."
git checkout HEAD -- src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_system_v1.j2
git checkout HEAD -- src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_user_v1.j2
git log --oneline -20
```

## results.tsv Column Format

Tab-separated. Append one row per experiment:

```text
experiment_id  score     delta     status    notes                        judge_a_model  judge_b_model  rubric_hash  eval_dataset_ref
exp-1          0.623456  +0.01234  accepted  hypothesis text here         gpt-4o-mini    claude-haiku-4-5-20251001  9f43a4b9  curated_5feeds_smoke_v1
```

Status values: `baseline`, `accepted`, `rejected`, `error`.

## On Completion

Write a summary to `autoresearch/bundled_prompt_tuning/summary_openai_<YYYY-MM-DD>.md` with:

- Total experiments run
- Best score vs baseline (absolute and % improvement)
- Top changes that improved the score with their deltas
- Patterns: what consistently helped vs hurt
- Suggested next directions
- Any crashes or anomalies

Print "AUTORESEARCH COMPLETE" as the final stdout line.
