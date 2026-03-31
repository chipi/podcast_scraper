# Plan: Remove `gi_insight_model` — derive GIL insight provenance from implementation

**Status:** WIP — implementation plan  
**Related:** Issue #467 (provider parity); confusion between config “labels” and actual API/ML calls

---

## Problem

- **`gi_insight_model`** in `Config` reads like “the model that produces insight text,” but:
  - For **`gi_insight_source: summary_bullets`**, insight strings come from **summary bullets**; the **live** summarization model is `openai_summary_model` / `summary_model` (per provider), **not** `gi_insight_model`.
  - For **`gi_insight_source: provider`** (e.g. OpenAI `generate_insights`), the code calls **`self.summary_model`**, not `gi_insight_model` (see `openai_provider.generate_insights`).
- In practice `gi_insight_model` is mostly a **duplicate string** stamped into **`gi.json` → `model_version`**, easy to **drift** from what actually ran. Operators should not maintain a second “documentation” model id.

**Intent (restated):** Provenance should reflect **implementation** — whatever backend actually supplied or would supply insight text for that run — without a separate user-facing field that only copies intent.

---

## Principles

1. **No standalone config key** whose sole job is to label `gi.json` when the pipeline can infer the value.
2. **`gi.json` `model_version`** remains **required** by schema; its value becomes **derived** from `gi_insight_source`, `summary_provider`, and `cfg` (provider-specific summary model fields), using one small resolver.
3. If we later need **separate** models for “summarize” vs “generate_insights” on the same provider, that is a **provider implementation** change (e.g. optional override inside OpenAI provider config), not a parallel `gi_insight_model` string.

---

## Proposed behavior (`model_version` in artifact)

Central helper, e.g. `resolve_gil_artifact_model_version(cfg, summary_provider, *, gi_insight_source: str) -> str`:

| `gi_insight_source` | Rule |
| --- | --- |
| **`summary_bullets`** | Model id that **produced bullets** for `summary_provider` (e.g. `getattr(summary_provider, "summary_model", None)` or registry; fallback to provider name + `"unknown"`). |
| **`provider`** | Model id used by **`generate_insights`** on that provider (today OpenAI uses `summary_model` — same as bullets path until we add a dedicated insight model in provider config). |
| **`stub`** | `"stub"`. |

**Edge cases:**

- `summary_provider` is `None` or missing `summary_model`: fallback `"stub"` or `"unknown"` (document choice; prefer explicit `"unknown"` for non-stub sources).
- **ML `summary_provider`:** expose or read the configured map/reduce or display model string used for summaries so bullets path is still accurate.

---

## Implementation checklist

### 1. Resolver module

- Add `resolve_gil_artifact_model_version` in a small module (e.g. `src/podcast_scraper/gi/provenance.py` or under `workflow/helpers.py` if you want to avoid new file — prefer **`gi/provenance.py`** to keep GIL concerns together).
- Unit tests: matrix of `gi_insight_source` × mock providers × cfg stubs.

### 2. Call sites

- **`metadata_generation.py`:** Replace `model_version=getattr(cfg, "gi_insight_model", "stub")` with `model_version=resolve_gil_artifact_model_version(cfg, summary_provider, gi_insight_source=...)`.
- **`gi/pipeline.py`:** Remove `getattr(cfg, "gi_insight_model", model_version)` overrides; use the `model_version` argument passed into `build_artifact` (caller is source of truth).

### 3. Config

- **Remove** `gi_insight_model` from `Config` (or deprecate one release: Pydantic `Field(deprecated=...)` / warning if present in YAML, then remove).
- **`cli.py`:** Replace log line `GI insight model: …` with something accurate, e.g. `GI insight source: …` and optional `GI artifact model_version (derived): …` at debug, or single line `GI insight provenance: <source> / <resolved model>`.

### 4. Docs

- **`docs/api/CONFIGURATION.md`:** Drop `gi_insight_model` row; document derived `model_version` behavior under GIL.
- **`docs/guides/GROUNDED_INSIGHTS_GUIDE.md`:** Short subsection “Insight text provenance in gi.json.”
- **`docs/rfc/RFC-049`:** Update list of config keys; note `gi_insight_model` removed in favor of resolver.
- **`docs/gi/gi.schema.json`:** Refresh `model_version` description: “Model identifier for insight text lineage, derived from pipeline (see …).”

### 5. YAML and tests

- Remove `gi_insight_model` from:
  - `config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml`
  - `config/manual/manual_planet_money_openai_gi_kg_provider.yaml`
  - `config/manual/manual_planet_money_ml_gi_kg_summary_bullets.yaml`
- **Acceptance configs:** grep `gi_insight_model` and remove or migrate.
- **Unit tests:** `test_pipeline.py`, `test_metadata_generation.py` — stop setting `cfg.gi_insight_model`; assert `model_version` in artifact matches resolver given mock provider.

### 6. Optional follow-up (separate PR)

- If product wants **different** API model for `generate_insights` vs map/reduce summary: add **`openai_insight_model`** (or generic per-provider field) and use it **inside** `OpenAIProvider.generate_insights`; resolver then reads **that** for `gi_insight_source=provider`. Still no orphan `gi_insight_model` on global `Config`.

---

## Acceptance criteria

- No `gi_insight_model` in `Config` (or deprecated with clear migration message).
- `gi.json` `model_version` matches **derived** rule for bullets / provider / stub paths in tests.
- Manual configs and docs do not ask users to duplicate summary model id for GIL.

---

## Related files (grep anchors)

- `src/podcast_scraper/config.py` — field removal  
- `src/podcast_scraper/workflow/metadata_generation.py` — `build_artifact` call  
- `src/podcast_scraper/gi/pipeline.py` — drop cfg override  
- `src/podcast_scraper/cli.py` — logging  
- `tests/unit/podcast_scraper/gi/test_pipeline.py`  
- `tests/unit/podcast_scraper/workflow/test_metadata_generation.py`  
- `config/manual/*.yaml`, `config/acceptance/**` if present
