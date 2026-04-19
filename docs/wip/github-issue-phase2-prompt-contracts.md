## Scope

**Issue:** https://github.com/chipi/podcast_scraper/issues/605

Offline Jinja prompt contract tests (no LLM, no API calls). Builds on #604 (cleaning transcript bug + phase-1 `module_prompts` tests).

1. **Insight extraction:** every `**/insight_extraction/*.j2` must include `{{ transcript }}` (same failure class as missing transcript in cleaning).
2. **Jinja parse:** every `prompts/**/*.j2` must compile with `jinja2.Template` (syntax-only).
3. **Shared JSON user prompts:** stable instruction substrings on `shared/summarization/bullets_json_v1` and `bundled_clean_summary_user_v1` (parser-facing copy).
4. **NER:** `**/ner/guest_host_v1.j2` must reference `{{ episode_title }}`; `**/ner/system_ner_v1.j2` must not contain `{{ transcript }}`.

## Out of scope (follow-up)

- Evidence templates (`openai/evidence/*.j2` and future multi-provider evidence).

## Acceptance

- New tests under `tests/unit/podcast_scraper/prompts/`.
- `make format` + `pytest tests/unit/podcast_scraper/prompts/` green.
