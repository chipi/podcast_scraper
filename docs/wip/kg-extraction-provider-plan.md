# Plan: `kg_extraction_provider` from day one

**Status:** WIP — design + implementation checklist  
**Aligns with:** Provider-first architecture (per-capability backend choice); same factory pattern as `quote_extraction_provider` / `entailment_provider` / `summary_provider`.

---

## Goal

Introduce **`kg_extraction_provider`** so **`kg_extraction_source: provider`** can call **`extract_kg_graph`** on:

- **`summary_provider`** when the operator wants one stack (default), **or**
- **Any other** registered summarization backend (e.g. OpenAI summaries + Gemini KG) without overloading `summary_provider`.

**Default:** `None` (or explicit sentinel documented as “use `summary_provider`”) — **zero behavior change** for existing YAML until someone sets a different provider.

---

## Config

- **Field:** `kg_extraction_provider: Optional[Literal["transformers", "hybrid_ml", "openai", …]]` — **same enum** as `summary_provider` / GIL evidence providers (reuse existing literal set from `config.py` or a shared type alias).
- **Default:** `None` → resolve to **`summary_provider` instance** at runtime (current behavior).
- **When ignored:** If `kg_extraction_source` is **`stub`** or **`summary_bullets`**, `kg_extraction_provider` is **not used** (no `extract_kg_graph`). Document clearly to avoid confusion.

- **Optional:** `kg_extraction_model` already overrides model **id** for the KG call when using the **same** provider instance; keep it working when `kg_extraction_provider` differs from `summary_provider` (params passed into `extract_kg_graph` unchanged).

---

## Runtime wiring

**File:** `src/podcast_scraper/workflow/metadata_generation.py` (KG block next to today’s `kg_provider_arg = summary_provider`).

1. If `kg_extraction_source != "provider"`: `kg_provider_arg = None` (or omit provider path as today).
2. If `kg_extraction_source == "provider"`:
   - If `kg_extraction_provider` is **None** or **equal** to `cfg.summary_provider` (string): `kg_provider_arg = summary_provider` (reuse instance — no double init).
   - Else: `create_summarization_provider(cfg, provider_type_override=cfg.kg_extraction_provider)`, **`initialize()`** if present, pass as `kg_extraction_provider` to `kg.build_artifact`.
3. **Cleanup:** If a **dedicated** KG provider instance was created and is not `summary_provider`, call **`cleanup()`** after `write_artifact` (mirror GIL quote/entailment cleanup loop).

---

## CLI

- Add **`--kg-extraction-provider`** with the same `choices=` set as other provider flags.
- Thread into `_build_config` / args → `Config`.

---

## Logging

- **`cli.py` `_log_configuration`:** When `generate_kg` and `kg_extraction_source == provider`, log effective KG backend, e.g.  
  `KG extraction provider: openai (or: same as summary)`  
  or reuse a single line with `ownership`-style clarity.

- **`orchestration`:** Optional second line `ownership_kg_extraction: …` when `generate_kg` + provider source (parallel to `ownership_gil_evidence`).

---

## Docs

- **`docs/api/CONFIGURATION.md`:** New row for `kg_extraction_provider`; update `kg_extraction_source` description to say LLM path uses **`kg_extraction_provider` or falls back to `summary_provider`**.
- **`docs/guides/KNOWLEDGE_GRAPH_GUIDE.md`:** Short subsection “KG LLM provider vs summary provider.”
- **`docs/wip/gil-evidence-defaults-openai-hybrid-problem.md`:** Replace “optional future” with link to this plan + “in scope from start.”

---

## Tests

- **Unit:** Config load with `kg_extraction_provider` set / unset; string equality with `summary_provider` reuses same mock instance (patch `create_summarization_provider` assert call count).
- **Unit:** `kg/pipeline.build_artifact` unchanged contract — still receives optional provider instance.
- **Integration / metadata:** One test path where `kg_extraction_provider != summary_provider` and `extract_kg_graph` is invoked on the second provider (mock).

---

## Acceptance criteria

- Existing configs without `kg_extraction_provider` behave as today.
- `kg_extraction_source: provider` + `kg_extraction_provider: <X>` uses provider **X** for `extract_kg_graph` when **X ≠ summary_provider**.
- Cleanup runs for extra provider instances; no duplicate long-lived clients when X equals summary.

---

## Related code

- `src/podcast_scraper/workflow/metadata_generation.py` — KG provider wiring  
- `src/podcast_scraper/kg/pipeline.py` — `_try_provider_extraction`, `build_artifact`  
- `src/podcast_scraper/summarization/factory.py` — `create_summarization_provider`  
- `src/podcast_scraper/cli.py` — argparse + config dict  
- `src/podcast_scraper/config.py` — new field + validator if needed (e.g. reject unknown provider string)
