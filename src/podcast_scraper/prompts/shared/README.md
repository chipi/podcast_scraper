# Shared prompts (`prompts/shared/`)

Templates here are **reused across providers** when no provider-specific file exists for the same logical name.

## Summarization (`shared/summarization/`)

**Why shared defaults:** API summarization often targets one **stable output shape** (e.g. JSON bullet arrays) so parsers and downstream consumers do not fork per vendor. Maintaining identical instructions in `openai/`, `gemini/`, `anthropic/`, … tends to drift.

**How loading works:** For a logical name like `gemini/summarization/bullets_json_v1`, the prompt store loads:

1. `prompts/gemini/summarization/bullets_json_v1.j2` **if present**
2. Otherwise `prompts/shared/summarization/bullets_json_v1.j2`

Implementation: `podcast_scraper.prompts.store._resolve_template_path`.

**Per-provider optimization:** Add a file under `prompts/<provider>/summarization/` with the **same filename** as the shared template. That file wins; use it for model-specific wording while keeping the same config path if the contract is unchanged.

**If you change the JSON/schema shape:** Treat that as a pipeline + schema change, not only a template edit.

**Board summaries (RFC-127):** `board_system_v1.j2` + `board_user_v1.j2` are the chat prompts for a
board summary; `board_document_v1.j2` writes the same material as plain prose for the seq2seq
summarizer (BART/LED), which follows no instructions. They render board material, never a
transcript, and are passed as a use case of each provider's `summarize`
(`providers/common/summary_use_case.py`); config: `board_summary_system_prompt`,
`board_summary_user_prompt`, `board_summary_document_prompt`, `board_summary_max_tokens`.

## Other shared subtrees

- **`shared/speaker_naming/`** — transcript naming (`providers/ml/diarization/transcript_naming.py`): `instructions_v1` is the per-voice instruction block, `standalone_v1` the whole staged prompt around it. The rendered prompt is pinned byte for byte to the one measured on the labelled naming sets (`tests/unit/.../diarization/test_transcript_naming.py`); change it only with a new measurement.
- **`shared/kg_graph_extraction/`** — KG extraction prompts shared where applicable (same override pattern if extended to provider paths).

## Further reading

- **Prompt management** (design context): see repository docs under `docs/guides/` and `docs/api/CONFIGURATION.md` — section *Shared summarization templates vs per-provider overrides*.
- **Config:** `docs/api/CONFIGURATION.md` — summary bullets and `summary_prompt_params`.
