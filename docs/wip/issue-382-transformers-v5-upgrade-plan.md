# Issue #382 — Transformers v5 upgrade plan

Plan for [GitHub issue #382](https://github.com/chipi/podcast_scraper/issues/382): allow **transformers v5.x** and keep local ML summarization, hybrid reduce, and extractive QA working.

**Branch:** `feat/upgrade-transformers-v5` (or successor).

**Primary references:**

- [Transformers v5 migration guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Transformers v5 release notes](https://github.com/huggingface/transformers/releases/tag/v5.0.0)

## 1. Why this is larger than a one-line pin change

Transformers v5 **removes** several high-level `pipeline()` task types that this repo still uses:

- `pipeline("summarization", …)` — **SummarizationPipeline**
- `pipeline("question-answering", …)`
- `pipeline("text2text-generation", …)` — **Text2TextGenerationPipeline** (same family as summarization/translation)

The migration guide recommends **`text-generation` with chat models** for many tasks. For **podcast_scraper**, the practical approach is to **keep existing model IDs** (BART, Pegasus, LED, seq2seq reduce models) and replace pipelines with **`AutoTokenizer` + `AutoModelForSeq2SeqLM` (or QA head) + `generate()` / forward pass**, so behavior and config stay aligned with today’s local ML story.

## 2. Code touchpoints (pipeline removal)

| Module | Current API | v5 action |
| ------ | ----------- | --------- |
| `src/podcast_scraper/providers/ml/summarizer.py` | `pipeline("summarization", …)` | Replace with `generate()`-based path |
| `src/podcast_scraper/providers/ml/hybrid_ml_provider.py` | `pipeline("text2text-generation", …)` | Replace with `generate()` on loaded seq2seq |
| `src/podcast_scraper/providers/ml/extractive_qa.py` | `pipeline("question-answering", …)` | Replace with `AutoModelForQuestionAnswering` + logits |
| `src/podcast_scraper/providers/ml/model_loader.py` | `pipeline("question-answering", …)` (warmup) | Align with new QA loading path |

Other areas to verify after install: `set_seed`, logging imports, version sniffing (`metadata_generation`, `run_manifest`), and `cache/directories.py` (`transformers.file_utils` fallback — primary path is already `huggingface_hub`).

## 3. Dependency pass (single resolver shot)

1. **`pyproject.toml`**
   - Bump **`transformers`** from `>=4.30.0,<5.0.0` to **`>=4.30.0,<6.0.0`** (or set a **minimum v5** once the lowest working version is known).
   - Revisit **`sentence-transformers>=2.2.0,<3.0.0`** in **`ml`** and **`dev`** — expect possible **cap raise** or major bump if pip cannot resolve with transformers 5.
   - Revisit **`accelerate`** floor if Trainer or upstream docs require it (guide mentions higher minimums in some setups); keep within project policy caps.

2. **Install**

   ```bash
   pip install --upgrade -e .[dev,ml]
   ```

3. **Repository grep (follow-up fixes)**
   - `use_auth_token` → `token`
   - `safe_serialization=False` on save/push (parameter removed in v5)
   - Any imports from removed internal modules (`tokenization_utils` paths called out in the migration guide)

## 4. Implementation phases

### Phase A — Summarization (highest risk / largest file)

- **`summarizer.py`:** Remove summarization `pipeline`; use loaded **`AutoModelForSeq2SeqLM`** + **`AutoTokenizer`** + **`model.generate()`** with equivalent generation arguments (`max_length`, `min_length`, `num_beams`, etc.).
- Preserve **device selection**, **OOM retry**, and **threading/tokenizer** error handling where still relevant.
- Prefer a **small internal helper** for “run seq2seq summarization on this text” to keep map/reduce logic readable.

### Phase B — Hybrid reduce

- **`hybrid_ml_provider.py`:** Model and tokenizer are already loaded; replace **`pipeline("text2text-generation")`** with **`generate()`** on the same prompt string built today; map return shape to existing `HybridReduceResult`.

### Phase C — Extractive QA

- **`extractive_qa.py`** and **`model_loader.py`:** Replace QA `pipeline` with **`AutoModelForQuestionAnswering`** + tokenizer + forward pass to obtain answer span; keep public behavior of helpers stable for callers.

### Phase D — Misc cleanup

- **`cache/directories.py`:** If `transformers.file_utils` disappears, rely on **`huggingface_hub.constants`** and documented env vars (`HF_HOME`, `HF_HUB_CACHE`).
- Fix any **import** or **typing** breakages (`Pipeline` types, etc.).

## 5. Testing and validation

1. **`make format`** on edited Python files; **`make lint`** and **`make type`** as needed.
2. **`make ci-fast`** before commit (project default; use Makefile timeout guidance for long runs).
3. Pay attention to tests that **load real transformers** or depend on optional **`[ml]`** installs.
4. Optional manual smoke: **short transcript**, map/reduce summarization path once.

## 6. Documentation and issue checklist

- Add a short **developer/ML note** (where maintainers look for HF behavior): local paths **no longer use** removed **`summarization` / `text2text-generation` / `question-answering` pipelines**; they use **direct generation / QA forward**.
- Extend **issue #382** checklist with explicit items: **pipeline removal**, **sentence-transformers/accelerate alignment**, **QA refactor**.

## 7. Suggested order of execution

1. Pin updates + `pip install -e .[dev,ml]` → capture resolver and import errors.
2. **Phase C** (QA + loader) — smaller surface, validates pattern.
3. **Phase B** (hybrid reduce).
4. **Phase A** (summarizer).
5. **Phase D** + full **`make ci-fast`**.

---

*Last updated: 2026-04-02 — aligns with maintainer comment on issue #382 and codebase survey.*
