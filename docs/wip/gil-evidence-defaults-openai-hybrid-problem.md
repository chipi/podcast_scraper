# GIL evidence defaults vs OpenAI-first configs (problem + proposed fixes)

**Status:** WIP — design / product note  
**Audience:** Maintainers defining provider coverage and config semantics for GIL

Also covers **how KG layering differs** from GIL (no separate default local stack for the usual `summary_bullets` path).

---

## Architecture (target)

- **Provider-first:** Capabilities (summarize, GIL quote extraction, GIL entailment, KG extraction, etc.) each choose a **backend** from the **same set of providers** the project supports (local ML stack vs various LLM APIs). Users pick what they have: local hardware, free/paid APIs, etc.
- **Mixing is intentional:** Config may assign **different providers to different capabilities** (e.g. one API for summaries, local ML for QA, another API for NLI). That is **valid product behavior**, not a hybrid accident to forbid.
- **Primary fix (engineering):** Implement **GIL evidence surface** (`extract_quotes`, `score_entailment`, and related hooks) **on every provider we claim to support**, document the **capability × provider matrix**, and add **strong tests**. Defaults and manual YAML are **secondary** (documentation, ergonomics, optional policy such as “default evidence = summary provider”), not the architectural story.

---

## Summary

A concrete example: YAML with **`summary_provider`** set to an **LLM API** (e.g. OpenAI — only one instance of the pattern) **does not** by itself set **GIL evidence** providers. Today **quote extraction** and **entailment** default to **`transformers`** (local Hugging Face + **sentence-transformers** for NLI). If the operator assumed “everything follows my summary provider,” they hit **hidden `.[ml]`** deps without having chosen local evidence explicitly.

**KG is different:** default **`kg_extraction_source: summary_bullets`** builds Topic nodes from **already-produced summary bullets** (no extra KG-specific ML stack). LLM KG JSON uses **`extract_kg_graph` on the same `summary_provider`** only when **`kg_extraction_source: provider`**.

This note names the GIL gap (implementation parity + defaults/docs), why it bites in practice, and follow-ups including KG provider override **if** we need a different backend than `summary_provider` for `extract_kg_graph` only.

---

## The problem

### What people assume

If a config sets:

- `summary_provider: openai`
- `gi_insight_source: summary_bullets`

…it is natural to read the run as: **“insights and summaries come from OpenAI; the rest is lightweight.”**

### What actually runs

Insight **text** comes from OpenAI summary bullets. **Grounding** (find a transcript span + score entailment) uses **separate** fields:

- `quote_extraction_provider` — default **`transformers`**
- `entailment_provider` — default **`transformers`**

So the pipeline is **hybrid**: OpenAI for generation, **local ML** for evidence unless the YAML overrides those two fields.

### What was “missing” in the broken run

Not OpenAI. The local NLI path imports **`sentence_transformers`** (CrossEncoder). A venv can have `transformers`/`torch` from other work and still lack **`sentence-transformers`**. Failures show up mid-run (or at startup after validation) as **missing `sentence_transformers`**, which feels unrelated to “we use OpenAI.”

### Why this is harmful

1. **Wrong mental model** — “OpenAI config” silently implies **GPU/CPU ML stack + large downloads** for grounding.
2. **Operational surprise** — CI or laptops without `.[ml]` fail on GIL despite “API-only” appearance.
3. **Cost / latency opacity** — API bill is visible; **local** QA/NLI cost (RAM, time, cold load) is easy to miss in the same run.
4. **Preset drift** — Recommended manual configs (e.g. Planet Money OpenAI + bullets) advertise a smooth first pass but **require** either `.[ml]` or explicit evidence overrides; that coupling is easy to under-document.

---

## Root cause (design)

GIL was modeled with **orthogonal knobs**:

- **Where insight strings come from** (`gi_insight_source`, bullets vs provider vs stub)
- **Which backend implements QA/NLI** (`quote_extraction_provider`, `entailment_provider`)

Defaults favor **local** evidence (no API key for grounding, reproducible offline tests). That made sense for **ML-first** and **CI** scenarios. It does **not** match the **“all OpenAI”** story unless configs explicitly align the evidence providers with the summary provider.

So the gap is not “mixing providers is wrong”; it is **incomplete parity** (not every provider fully implements GIL evidence), plus **defaults/docs** that do not spell out which capability uses which backend until the matrix is obvious.

---

## Proposed solutions

**Primary track — provider parity + tests:** audit and implement GIL evidence methods across all providers; document the matrix; unit/integration tests per provider and for mixed-capability configs.

**Secondary track — ergonomics** (optional; does not replace parity): presets, startup hints, default policy. Options below can be combined; rough order within the secondary track: quick wins first, then behavior changes that need care for backward compatibility.

### A — Presets and docs (low risk, immediate)

- **Manual / acceptance YAML** that are marketed as “OpenAI-first” should either:
  - set `quote_extraction_provider: openai` and `entailment_provider: openai` when the goal is **no local GIL deps**, **or**
  - keep local evidence but **title and comment** the file as **“OpenAI summaries + local grounding (requires .[ml])”** so the hybrid is explicit in the filename and header.
- **Manual test plan** (`docs/wip/manual-test-plan-gi-kg.md`) already nudges toward `.[ml]` or LLM evidence; link this WIP from that doc for “why.”

**Pros:** No code churn, truthful UX.  
**Cons:** Every new preset must remember the two extra keys.

### B — CLI / config printout (low risk)

- When `generate_gi` and `gi_require_grounding` are true, if `summary_provider` is an API provider but **either** evidence provider is `transformers` / `hybrid_ml`, emit a single **INFO** (or WARNING) line, e.g.  
  `GIL evidence is local (transformers); install .[ml] or set entailment_provider to match summary_provider for API-only grounding.`

**Pros:** Surfaces hybrid at run start.  
**Cons:** Noise for users who want local grounding intentionally.

### C — “Align evidence with summary” helper in YAML (medium)

- Document a **copy-paste block** in CONFIGURATION / GROUNDED_INSIGHTS_GUIDE:

  ```yaml
  quote_extraction_provider: openai
  entailment_provider: openai
  ```

  when `summary_provider: openai` and the operator wants **API-only** grounding.

- Optional future field (needs RFC): `gil_evidence_stack: auto` where `auto` means “use `summary_provider` for both quote and entailment when it supports `extract_quotes` / `score_entailment`.”

**Pros:** One conceptual switch.  
**Cons:** `auto` must be specified precisely (what if summary is `transformers` but user wanted API for NLI only?).

### D — Change defaults when summary is API (higher risk)

- If `summary_provider` is in `{openai, gemini, anthropic, ...}` **and** `generate_gi` + `gi_require_grounding`, default `quote_extraction_provider` and `entailment_provider` to **match** `summary_provider` unless explicitly set.

**Pros:** Matches “OpenAI-first” intuition.  
**Cons:** **Breaking change** for anyone relying on implicit local grounding with API summaries; tests and acceptance configs must be audited; offline runs need explicit `transformers`.

**Mitigation:** Ship behind a **major version** or a new opt-in flag first, e.g. `gi_evidence_align_with_summary: true`, then flip default later.

### E — Split “recommended” presets in `config/manual/README.md`

- **Preset B1:** OpenAI bullets + **API** grounding (keys only, `.[ml]` not required for GIL).  
- **Preset B2:** OpenAI bullets + **local** grounding (current default behavior, `.[ml]` required).

**Pros:** Clear product surface; user picks dep profile.  
**Cons:** More files or clearer naming of existing ones.

---

## Recommendation (for closeout)

1. **Ship provider parity + tests** for GIL evidence on every supported backend; publish the **capability × provider** matrix in CONFIGURATION / GROUNDED_INSIGHTS_GUIDE.  
2. **Then** (optional): **A + E** for manual presets, **B** for startup clarity, **C / D** only if product wants a simpler default (e.g. evidence defaults follow `summary_provider`) **without** removing the ability to mix providers per capability.

---

## Knowledge Graph (KG): how logic and providers work

KG does **not** mirror GIL’s split defaults. There is a **single** switch: `kg_extraction_source` ∈ `stub` | `summary_bullets` | `provider` (see `src/podcast_scraper/kg/pipeline.py`).

| Source | What runs | Extra deps beyond summaries? |
| --- | --- | --- |
| **`summary_bullets`** (default) | Topic nodes from **strings** passed in as `topic_labels` (from metadata: summary bullets). No `extract_kg_graph` call. | **No** KG-specific ML. You already paid for bullets via `summary_provider`. |
| **`provider`** | Calls **`extract_kg_graph(transcript, …)`** on the **summarization provider instance** passed from metadata generation — today that is **`summary_provider` only** (`metadata_generation.py` sets `kg_provider_arg = summary_provider` when source is `provider`). | Whatever that provider needs (API key for OpenAI, etc.). |
| **`stub`** | Episode-centric graph; minimal / placeholder-style extraction label. | No LLM for topics from bullets. |

**ML summarization + `kg_extraction_source: provider`:** `MLProvider.extract_kg_graph` returns **`None`** (not implemented). The pipeline then falls back to bullets when available; the CLI warns that provider KG is a no-op for ML summaries.

### Do we need to “expand providers” for KG?

**For the same problem as GIL (hidden local stack):** **No.** The recommended OpenAI + **`summary_bullets`** manual path does **not** pull in `sentence-transformers` for KG. The surprise you hit was **GIL evidence**, not KG.

**For product parity (“KG LLM from a different backend than summary”):** **Planned from the start** as first-class config — see **[kg-extraction-provider-plan.md](kg-extraction-provider-plan.md)**. Summary: add **`kg_extraction_provider`** (default = use **`summary_provider`**), same factory/cleanup pattern as GIL evidence providers; only applies when **`kg_extraction_source: provider`**.

Until that field is implemented, LLM KG extraction remains **`summary_provider` only**.

---

## Related

- `config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml` — GIL hybrid unless evidence keys added; KG bullets path needs no extra KG ML  
- `docs/wip/manual-test-plan-gi-kg.md` — Step B operator notes  
- `docs/wip/kg-extraction-provider-plan.md` — **`kg_extraction_provider`** implementation plan  
- `docs/api/CONFIGURATION.md` — GIL evidence provider table; KG extraction source  
- `src/podcast_scraper/gi/deps.py` — startup check for local GIL NLI (`sentence-transformers`)  
- `src/podcast_scraper/kg/pipeline.py` — `build_artifact`, `_try_provider_extraction`  
- `src/podcast_scraper/workflow/metadata_generation.py` — wires KG provider argument (today: `summary_provider` only)
