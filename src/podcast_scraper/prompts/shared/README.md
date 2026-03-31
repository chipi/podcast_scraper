# Shared prompts (`prompts/shared/`)

Templates here are **reused across providers** when no provider-specific file exists for the same logical name.

## Summarization (`shared/summarization/`)

**Why shared defaults:** API summarization often targets one **stable output shape** (e.g. JSON bullet arrays) so parsers, GI, and KG (`summary_bullets`) do not fork per vendor. Maintaining identical instructions in `openai/`, `gemini/`, `anthropic/`, … tends to drift.

**How loading works:** For a logical name like `gemini/summarization/bullets_json_v1`, the prompt store loads:

1. `prompts/gemini/summarization/bullets_json_v1.j2` **if present**
2. Otherwise `prompts/shared/summarization/bullets_json_v1.j2`

Implementation: `podcast_scraper.prompts.store._resolve_template_path`.

**Per-provider optimization:** Add a file under `prompts/<provider>/summarization/` with the **same filename** as the shared template. That file wins; use it for model-specific wording while keeping the same config path if the contract is unchanged.

**If you change the JSON/schema shape:** Treat that as a pipeline + schema change, not only a template edit.

## Other shared subtrees

- **`shared/kg_graph_extraction/`** — KG extraction prompts shared where applicable (same override pattern if extended to provider paths).

## Further reading

- **RFC-017** (design context): `docs/rfc/RFC-017-prompt-management.md` — section *Shared summarization templates vs per-provider overrides*.
- **Config:** `docs/api/CONFIGURATION.md` — summary bullets and `summary_prompt_params`.
