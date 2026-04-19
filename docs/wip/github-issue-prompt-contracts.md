# GitHub issue body (paste into UI or `gh issue create --body-file`)

**Created:** https://github.com/chipi/podcast_scraper/issues/604


## Problem

Multiple LLM providers used `*/cleaning/v1.j2` templates that **did not include** `{{ transcript }}` in the Jinja source (OpenAI's copy did). Hybrid / LLM transcript cleaning then called the model with **no episode text**, producing fabricated "cleaned" transcripts. With `save_cleaned_transcript: true`, bad `*.cleaned.txt` files were written and **summaries/metadata** followed that wrong content—many episodes looked unrelated to their real audio.

## Root cause

Template drift / incomplete copy when adding provider-specific cleaning prompts: same instruction block as OpenAI but missing:

```text
Transcript to clean:
{{ transcript }}
```

## Fix

- Add the transcript block to every `*/cleaning/v1.j2` that was missing it (aligned with `openai/cleaning/v1.j2`).

## Tests (no live LLM)

- `tests/unit/podcast_scraper/prompts/test_cleaning_prompt_templates.py`
  - Assert template source contains `{{ transcript }}`
  - Assert rendered prompt includes a unique marker passed as `transcript=`
  - Assert on-disk `*/cleaning/v1.j2` providers match `_CONTRACT_CLEANING_V1_PROVIDERS`
- Follow-up: shared summarization / KG prompt contracts, per-provider summarization glob checks, `module_prompts` marker + `find_impacted_tests` mapping for `src/podcast_scraper/prompts/**`.

## Operator / data note

Existing corpora with bad `*.cleaned.txt` or metadata built from them need **re-run** of cleaning/summary (or delete bad cleaned files) after deploying this fix.

## Follow-ups (separate issues OK)

- Extend the same **offline** pattern to **shared** summarization / KG prompts under `src/podcast_scraper/prompts/shared/` (see `shared/README.md` and RFC-017).
- Optional: central **registry** (logical name to required substrings) as coverage grows.
- Optional: `find_impacted_tests` / `validate-files-*` mapping for prompt paths (implemented alongside this issue).

## References

- Prompt loading / shared fallback: `src/podcast_scraper/prompts/store.py` (`_resolve_template_path`, `render_prompt`)
- Design: `docs/rfc/RFC-017-prompt-management.md`
