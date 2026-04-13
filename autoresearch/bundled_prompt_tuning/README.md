# Bundled Prompt Tuning

Autoresearch loop for the bundled LLM path — one-pass clean + summarize per provider.

The bundled path performs semantic cleaning and summarization in a single LLM call,
using provider-specific prompt templates (not the shared fallback). The shared templates
under `src/podcast_scraper/prompts/shared/summarization/` remain untouched and serve
as a fallback for new providers that do not yet have their own bundled templates.

## Structure

```text
bundled_prompt_tuning/
├── eval/
│   ├── score.py           ← immutable scoring harness (do not edit during a run)
│   ├── judge_config.yaml  ← pinned judge models (human-edited between runs only)
│   └── rubric.md          ← judge rubric (immutable during a run)
├── results/
│   └── results_openai_r1.tsv   ← experiment log, OpenAI round 1
└── program_openai_bundled.md   ← agent loop instructions for OpenAI
```

## Providers

| Provider | Templates | Program doc | Results log | Status |
| :--- | :--- | :--- | :--- | :--- |
| OpenAI | `openai/summarization/bundled_clean_summary_{system,user}_v1.j2` | `program_openai_bundled.md` | `results/results_openai_r1.tsv` | Active |

## Running

```bash
# Score OpenAI bundled (full run with judges)
make autoresearch-score-bundled

# Dry run (ROUGE only, no judge API calls)
make autoresearch-score-bundled DRY_RUN=1

# Override config or reference
make autoresearch-score-bundled CONFIG=data/eval/configs/... REFERENCE=silver_sonnet46_smoke_bullets_v1
```

## Ratchet mechanics

| Outcome | Command | What it touches |
| :--- | :--- | :--- |
| Accept (win > +1%) | `git add <templates> results/*.tsv && git commit` | Both committed |
| Reject | `git checkout HEAD -- <template>` | Template only restored; TSV safe |

## Silver reference

`silver_sonnet46_smoke_bullets_v1` — same as initial prompt tuning Track A.

## Historical work

See `autoresearch/initial_prompt_tuning/` for the RFC-057 Track A/B sweep results.
