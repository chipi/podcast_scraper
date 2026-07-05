# PR body draft — `feat: Upgrade transformers to v5 + unify ML architecture (#382)`

Pre-drafted so the push flow is one shot. Paste into `gh pr create --body ...`
or the GitHub UI when push is authorized. Update the "Files changed" summary
and any specific SHAs if the branch rebases pick up new work between now and
merge.

---

## Summary

Closes #382. Raises `transformers` to `>=5.0.0,<6.0.0` and unifies the local
ML architecture behind two shared backends. Public APIs preserved
(backwards-compatible). No operator action required at deploy time; no
config or schema changes.

- **`transformers` 4.57.6 → 5.13.0**, `sentence-transformers` 5.5.1 → 5.6.0.
- **`pipeline()` retired** for `summarization` / `text2text-generation` /
  `question-answering` (all three removed in v5). Replaced with direct
  `AutoModelFor*.from_pretrained` + `generate()` / forward.
- **New shared backends:**
  - `HFEvidenceBackend` (`providers/ml/hf_evidence_backend.py`) collapses
    the three parallel loaders in QA / NLI / embedding into one shape.
  - `HFSeq2SeqBackend` (`providers/ml/hf_seq2seq_backend.py`) collapses
    the summarizer + hybrid-reduce load/generate plumbing into one shape.
  - `_download_hf_evidence_model(kind, model_id)` in `model_loader.py`
    replaces three parallel `_download_*_for_cache` helpers.
- **Nine `pip-audit --ignore-vuln` entries removed** — CVE-2026-1839 +
  PYSEC-2025-211..218 are now fixed upstream in the 5.x line.

Delivered as one epic (Path C). 18 commits on `feat/upgrade-transformers-v5`;
see `docs/wip/ISSUE-382-TRANSFORMERS-V5-EXECUTION-PLAN.md` for the phase
ledger with per-commit SHAs.

## Behavior parity — measured

Frozen artifacts under `data/eval/`:

| Metric | Value | Threshold |
|---|---|---|
| Summarizer ROUGE-L (v4-pipeline → v5-forward, 5 smoke episodes) | **min=1.0000, mean=1.0000** | ≥ 0.95 |
| QA answer-text match (8 fixture pairs) | **7/8 = 87.5%** | ≥ 85% |
| QA offset match | 6/8 = 75% | (secondary datapoint) |
| Overall parity gate | **PASS** | — |

The one QA divergence (`p05_e01_q2_topic`) picks `"people care about"` (v5)
vs `"Index investing"` (v4) — both valid extractive answers; drift is from
the algorithmic switch (pipeline post-processing → our softmax over
aggregated top-k). Documented in the parity report and in Phase 3's commit
message.

Full report: `data/eval/runs/v5_parity_2026-07-05.json`.

## Architecture — before / after

Before (three parallel shapes for "load HF model, cache it, forward"):

- `SummaryModel` (summarizer.py) — 800+ LOC map-model class
- `TransformersReduceBackend` (hybrid_ml_provider.py) — 160 LOC reduce-model class
- `extractive_qa` / `nli_loader` / `embedding_loader` — bare-function triplet

After:

- `HFSeq2SeqBackend` — one loader/generator for BART/LED/Pegasus/LongT5/FLAN-T5.
  Both `SummaryModel` and `TransformersReduceBackend` are thin profiles over it.
- `HFEvidenceBackend` — one loader for QA/NLI/embedding. Three modules become
  thin heads over it; public API surface unchanged.
- `SummarizationProvider` Protocol (the "just different profiles and providers"
  seam) unchanged. Confirmed by mid-phase AI-provider audit
  (`docs/wip/ISSUE-382-AI-PROVIDER-AUDIT-2026-07-05.md`).

## Compatibility

**Public API preserved:**
- All `SummarizationProvider` methods (`initialize` / `summarize` / `cleanup`).
- `extractive_qa.answer` / `answer_candidates` / `answer_multi` / `QASpan` — unchanged.
- `nli_loader.entailment_score` / `entailment_scores_batch` — unchanged.
- `embedding_loader.encode` / `cosine_similarity` — unchanged.
- `hybrid_ml_provider.TransformersReduceBackend.reduce()` shape — unchanged.
- `SummaryModel.summarize()` signature — unchanged.

**Deprecations (aliases retained, warn on call):**
- `get_qa_pipeline` → `get_qa_model` (canonical).
- `load_qa_pipeline` → `load_qa_model` (canonical).
- Both aliases emit `DeprecationWarning`; kept until v3.0.0.

**Type-level (external code shouldn't have depended on it):**
- `SummaryModel.pipeline` field went from `Optional[Pipeline]` to `bool`
  (loaded sentinel). Callable use is impossible now.

## Test infrastructure

- **`HFEvidenceBackend`**: 15 new unit tests (cache, threading, subclass
  isolation, ABC contract) — 87% coverage.
- **`HFSeq2SeqBackend`**: 27 unit tests (defaults, snapshot family_class,
  snapshot fallback, device fallback matrix, adopt/unload) — 85% coverage.
- **31 pre-existing integration tests rewritten** to mock at the new
  backend seams instead of removed private symbols.
- **7 dead `@patch("transformers.pipeline", create=True)` decorators**
  removed from `test_summarizer.py` (Issue #677 anti-pattern).
- **New nightly regression test** (`tests/e2e/test_v5_parity_regression.py`)
  re-runs the compare script; catches future drift in `SummaryModel` /
  `HFEvidenceBackend` / `HFSeq2SeqBackend`.
- **New docs guide** (`docs/guides/testing-strategy-ml.md`) codifies
  where to mock (facade / class-method seams) and where not to
  (library-internal names, with the `sys.modules` stub exception).

## CVE / security cleanup

- CVE-2026-1839 (transformers Trainer `torch.load` weights_only) — fixed
  upstream in `>=5.0.0rc3`.
- PYSEC-2025-211..218 (8 advisories against transformers-4.x
  conversion-script / X-CLIP / Trainer deserialization) — fixes only in
  the 5.x line.
- All nine `pip-audit --ignore-vuln` entries removed from Makefile.
- Also fixed a pre-existing shell-syntax bug in `run-promote`
  (`# comment inside a line-continuation` ate the trailing backslash).

## Docs

- `docs/releases/RELEASE_v2.7.0.md` — full release note.
- `docs/adr/ADR-068-bart-led-as-ml-production-baseline.md` § Post-Implementation Notes — updated.
- `docs/guides/testing-strategy-ml.md` — NEW.
- `docs/wip/ISSUE-382-*` — deep analysis + execution plan + AI-provider audit.
- `mkdocs strict` build: green.

## Follow-up tracked outside this PR

- **#1142** — `feat(providers): extract shared BundledSummarizationMixin
  across cloud LLM providers`. Surfaced during audit; ~1460 LOC of
  duplication across OpenAI/Gemini/Anthropic/Mistral. Orthogonal to
  transformers v5; separate PR when scheduled.

## Test plan

- [x] `make ci-fast` — green (6015 unit tests, coverage, spell, lint,
      mypy, build).
- [x] `make docs` (mkdocs strict) — green.
- [x] Parity gate — `data/eval/runs/v5_parity_2026-07-05.json`
      `overall_pass = true`.
- [x] Broad regression sweep (unit + integration providers/ml) —
      728 pass, 13 skip.
- [x] Manual FLAN-T5 + BART smoke through the migrated backends —
      identical output to pre-migration.
- [ ] Nightly parity regression (scheduled): `tests/e2e/test_v5_parity_regression.py -m nightly`.
- [ ] Post-merge: watch `preload-ml-models` runs in the pipeline image
      Dockerfile for any HF cache surprises.

## Related

- `docs/wip/ISSUE-382-TRANSFORMERS-V5-DEEP-ANALYSIS-2026-07-05.md`
- `docs/wip/ISSUE-382-TRANSFORMERS-V5-EXECUTION-PLAN.md`
- `docs/wip/ISSUE-382-AI-PROVIDER-AUDIT-2026-07-05.md`
- `docs/adr/ADR-068-bart-led-as-ml-production-baseline.md`
- Follow-up: #1142
- Closes: #382
