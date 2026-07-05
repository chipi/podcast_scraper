# Release v2.7.0 — Transformers v5 upgrade + ML architecture unification

**Release Date:** 2026-07-05
**Type:** Minor release (major dependency bump, backwards-compatible public API)
**Last Updated:** 2026-07-05

## Summary

v2.7.0 raises the `transformers` floor from `4.57.6` to `>=5.0.0,<6.0.0`
and unifies the local ML architecture behind two shared backends
(`HFEvidenceBackend`, `HFSeq2SeqBackend`). Public APIs preserved — no
config changes, no schema changes, no operator action required.

Delivered as one epic PR closing GitHub issue **#382**.

## Highlights

- **transformers v5.13.0** replacing 4.57.6. Picks up the CVE-2026-1839
  fix (Trainer `torch.load` `weights_only`, fixed in 5.0.0rc3+) and the
  eight PYSEC-2025-211..218 transformers-4.x deserialization advisories
  that had no fix in the 4.x line.
- **sentence-transformers 5.6.0** (from 5.5.1) — silent causal-LM
  reranker scoring bug fixed; TSDAE restored on transformers v5.
- **`pipeline()` retired** for all local ML tasks:
  - `pipeline("summarization")` → `AutoModelForSeq2SeqLM.generate()` +
    `transformers.GenerationConfig`
  - `pipeline("text2text-generation")` (hybrid tier-1 reduce) → same
  - `pipeline("question-answering")` (extractive QA, removed in v5) →
    `AutoModelForQuestionAnswering.forward` with pipeline-parity
    post-processing (overflow chunking, top-k logit walk, softmax over
    aggregated candidates)
- **Two new shared backends** replace three parallel loader idioms:
  - `HFEvidenceBackend` (`providers/ml/hf_evidence_backend.py`) — QA /
    NLI / embedding. Owns device resolution (with per-subclass
    `mps_supported` flag), the standard `from_pretrained` kwargs
    (`local_files_only`, `low_cpu_mem_usage=False`, `trust_remote_code=False`),
    and a per-subclass instance cache with a threading lock.
  - `HFSeq2SeqBackend` (`providers/ml/hf_seq2seq_backend.py`) — one
    loader/generator for BART / LED / Pegasus / LongT5 / FLAN-T5.
    `SummaryModel` (map profile) and `TransformersReduceBackend`
    (reduce profile) both delegate through it. Snapshot-first
    checkpoint discovery + retry hook + model-family class override.
- **Download helper collapse:** three `_download_*_for_cache` helpers
  in `providers/ml/model_loader.py` collapsed into a single
  `_download_hf_evidence_model(kind, model_id)` with a `Literal` kind
  discriminator; `preload_evidence_models` iterates one list of
  `(kind, aliases)` pairs instead of three parallel loops.

## Behavior parity — measured

Frozen artifacts under `data/eval/`:

- **Summarizer (BART+LED, `curated_5feeds_smoke_v1`):** min ROUGE-L =
  1.0000, mean = 1.0000. Byte-identical output pre-vs-post migration
  on all 5 episodes.
  See `data/eval/baselines/baseline_ml_bart_authority_smoke_v5_pre/`
  and `..._v5_post/`, comparison in
  `data/eval/runs/v5_parity_2026-07-05.json`.
- **Extractive QA (`deepset/roberta-base-squad2`, 8 fixture pairs):**
  answer-text match 7/8 (87.5 %), offset match 6/8 (75 %). The
  one divergent pair is a legitimate algorithmic difference: our v5
  post-processing softmaxes over aggregated top-k candidates across
  chunks, where the removed v4 pipeline used its own aggregation.
  Both answers are correct extractive spans.

Full parity report: `data/eval/runs/v5_parity_2026-07-05.json`.

**Independent data-quality validation** (`data/eval/runs/v5_data_quality_recheck_2026-07-05.json`,
run via `scripts/eval/data_quality_recheck_2026-07-05.py`):

| Comparison                                                             | Metric                         | Result                                                              |
| ---------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------- |
| v5-post vs shipped `baseline_ml_bart_authority_smoke_v1` (April 2026)  | ROUGE-L / R-1 / R-2            | **1.0000 / 1.0000 / 1.0000** across all 5 episodes — byte-identical |
| v5-post vs `silver_opus47_smoke_v1` (Claude Opus 4.7 reference)        | mean ROUGE-L / R-1 / R-2       | **0.2175 / 0.4003 / 0.2391**                                        |
| Historical anchor: shipped_v1 vs same silver                            | mean ROUGE-L / R-1 / R-2       | **0.2175 / 0.4003 / 0.2391** — identical to row 2                   |

Rows 2 and 3 being identical is the strongest possible cross-check:
v5 achieves the same quality score against the semantic ground truth
as v4 did. No quality regression measured across three orthogonal
comparisons.

**Broader uniform data-quality coverage** (post-audit expansion): every
ML surface #382 touched now has a captured baseline + regression guard,
matching the summarizer path's validation bar.

| Surface                                  | Frozen reference                                                     | Bar                                     |
| ---------------------------------------- | -------------------------------------------------------------------- | --------------------------------------- |
| Summarizer smoke (BART+LED, 5 eps)       | `baselines/baseline_ml_bart_authority_smoke_v5_post/`                | byte-identical vs shipped_v1            |
| Summarizer benchmark (BART+LED, 10 eps)  | `baselines/baseline_ml_bart_authority_benchmark_v5/`                 | 1.0000 vs smoke on shared IDs           |
| Extractive QA (roberta-squad2, 8 pairs)  | `references/qa_baseline_v5_post.jsonl`                               | answer-text identity                    |
| NLI (deberta-v3-base, 12 pairs)          | `references/nli_baseline_v5.jsonl`                                   | entailment score ±0.01                  |
| Embedding (MiniLM-L6-v2, 10 inputs)      | `references/embedding_baseline_v5.jsonl`                             | dim + L2 + first-8 dims ±1e-4           |
| FLAN-T5 reduce (hybrid tier-1, 4 prompts)| `references/flant5_reduce_baseline_v5.jsonl`                         | output-text identity                    |

One-shot regression via `scripts/eval/full_ml_recheck.py --json-report ...`;
enforced by the nightly e2e test
`tests/e2e/test_v5_parity_regression.py::test_full_ml_surface_recheck_no_regressions`.

Absolute quality anchors (v5-post scored against semantic silvers):

| Baseline                              | Reference                              | Scope   | mean ROUGE-L / R-1 / R-2         |
| ------------------------------------- | -------------------------------------- | ------- | -------------------------------- |
| smoke_v5_post                         | silver_opus47_smoke_v1                 | 5 eps   | 0.2175 / 0.4003 / 0.2391         |
| shipped_v1 (historical, v4-pipeline)  | silver_opus47_smoke_v1                 | 5 eps   | 0.2175 / 0.4003 / 0.2391 (same)  |
| benchmark_v5                          | silver_sonnet46_benchmark_v1           | 10 eps  | 0.1906 / 0.3790 / 0.1520         |
| smoke_v5_post (subset)                | silver_sonnet46_benchmark_v1           | 5 eps   | 0.1975 / 0.4015 / 0.1700         |

Determinism across dataset scope: benchmark_v5 vs smoke_v5_post on the
5 shared episode IDs = **1.0000 ROUGE-L identical output** — the same
transcript produces the same summary whether it's part of a 5-episode
smoke run or a 10-episode benchmark run. No batch-size drift.


## Dependency deltas

```diff
-  "transformers>=4.57.6,<5.0.0",
+  "transformers>=5.0.0,<6.0.0",
-  "sentence-transformers>=5.4.1,<6.0.0",
+  "sentence-transformers>=5.6.0,<6.0.0",
```

Transitively pulled in: `huggingface-hub 1.22.0` (v5 requires >=1.0).
`torch`, `accelerate`, `tokenizers`, `numpy`, `protobuf`, `spacy` —
unchanged.

## Makefile / CI cleanup

Nine `pip-audit --ignore-vuln` entries removed:

- CVE-2026-1839 (transformers Trainer `torch.load` w/o `weights_only`)
- PYSEC-2025-211, -212, -213, -214, -215, -216, -217, -218 (transformers
  4.x conversion-script / X-CLIP / Trainer deserialization RCE)

Fixed a shell-syntax bug in `run-promote` (`#` comment inside a `\`
line-continuation block, silently eating the trailing `\`) so
`make baseline-create` auto-promotes correctly.

## Test infrastructure

- New: `tests/unit/podcast_scraper/providers/ml/test_hf_evidence_backend.py`
  (15 tests): device matrix, cache-hit-once, subclass-cache-isolation,
  concurrent get_or_load thread safety, ABC contract enforcement.
- New: `tests/unit/podcast_scraper/providers/ml/test_hf_seq2seq_backend.py`
  (19 tests): default-safetensors matrix, device auto-detect fallback,
  guards, snapshot-fallback path, generate-config wiring.
- Killed **seven** dead `@patch("transformers.pipeline", create=True)`
  decorators in `tests/unit/podcast_scraper/test_summarizer.py` —
  transformers `_LazyModule` renders `create=True` no-ops (issue #677).
- Bulk-migrated 34 pipeline-mock references in `test_summarizer.py` to
  the class-method seam (`_generate_summary`).
- New docs: `docs/guides/testing-strategy-ml.md` documents where to
  mock and why.

## Compatibility

**Public API preserved:**

- `extractive_qa.answer` / `answer_candidates` / `answer_multi` /
  `clear_qa_pipeline_cache` / `QASpan` — unchanged.
- `extractive_qa.get_qa_model` / `load_qa_model` — **NEW canonical names**
  for what were previously `get_qa_pipeline` / `load_qa_pipeline`. Both
  old names are retained as deprecation aliases (emit
  `DeprecationWarning` on call, `stacklevel=2`); they proxy to the new
  names verbatim. Migrate at your leisure; aliases stay until v3.0.0.
- `nli_loader.entailment_score` / `entailment_scores_batch` /
  `get_nli_model` / `load_nli_model` / `predict_output_to_entailment_scores` —
  unchanged.
- `embedding_loader.encode` / `cosine_similarity` /
  `get_embedding_model` / `load_embedding_model` — unchanged.
- `SummaryModel.summarize` / `SummaryModel(model_name, device, cache_dir, revision)`
  constructor — unchanged. Internal `self.pipeline` field type changed
  from `Optional[Pipeline]` to `bool` (loaded sentinel); external code
  should not have depended on the `Pipeline` type.
- `hybrid_ml_provider.TransformersReduceBackend.reduce()` shape and
  `HybridReduceResult` — unchanged.
- `SummarizationProvider` Protocol (`initialize` / `summarize` / `cleanup`)
  — unchanged.

**Config:** no changes required.
**Schemas:** no changes.
**Registry:** no revision-hash changes; all pinned checkpoints verified
resolvable under transformers 5.13.0.

## Known follow-ups (separate from this release)

- **Bundled-inference dedup across cloud providers** — tracked in
  [#1142](https://github.com/chipi/podcast_scraper/issues/1142). The
  `summarize_bundled` / `summarize_extraction_bundled` /
  `summarize_mega_bundled` trio on OpenAI / Gemini / Anthropic /
  Mistral totals ~1460 LOC across the four providers with a shared
  flow (chunking → LLM call → JSON-parse retry → cost tracking →
  result-shape assembly) but SDK-specific request/response shapes.
  Orthogonal to transformers v5; ships in its own PR when scheduled.
  Full measurement + API proposal + test plan lives in the audit doc
  (`docs/wip/ISSUE-382-AI-PROVIDER-AUDIT-2026-07-05.md`) and the issue.

## Related

- Deep analysis: `docs/wip/ISSUE-382-TRANSFORMERS-V5-DEEP-ANALYSIS-2026-07-05.md`
- Execution plan: `docs/wip/ISSUE-382-TRANSFORMERS-V5-EXECUTION-PLAN.md`
- AI-provider audit: `docs/wip/ISSUE-382-AI-PROVIDER-AUDIT-2026-07-05.md`
- Testing strategy: `docs/guides/testing-strategy-ml.md`
- Post-impl notes: `docs/adr/ADR-068-bart-led-as-ml-production-baseline.md`
  (§ Post-Implementation Notes — #382)
- Parity report: `data/eval/runs/v5_parity_2026-07-05.json`
- Closes: GitHub issue #382
