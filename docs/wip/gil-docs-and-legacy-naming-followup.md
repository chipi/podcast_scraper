# GIL provider-only path: doc + “legacy” naming follow-up

**Status:** WIP — no doc fixes applied in this note; inventory for a later pass.

## What changed in code (context)

- GIL grounding in `build_artifact` uses **only** the provider path:
  `find_grounded_quotes_via_providers` with `quote_extraction_provider` and
  `entailment_provider`.
- If callers omit those instances, **`create_gil_evidence_providers`** in
  `src/podcast_scraper/gi/deps.py` builds them from `Config` (same wiring idea as
  the old `metadata_generation` block), optionally reusing `summary_provider`.
- The previous **direct-ML fallback** (`find_grounded_quotes` inside the pipeline)
  and **provider-exception → legacy** behavior were removed.
- Metric **`gi_evidence_path_legacy`** was removed (`metrics.py`, JSONL emitter,
  run summary helpers, tests).
- `find_grounded_quotes` **remains** in `gi/grounding.py` for unit/integration
  tests that hit `extractive_qa` + `nli_loader` directly.

## Documentation likely stale (update when convenient)

These still describe the old “legacy path” or “provider vs legacy” metrics:

| File | Issue |
| ---- | ----- |
| `docs/guides/GROUNDED_INSIGHTS_GUIDE.md` | § around “legacy path” and “On exception … fall back to the legacy path” — replace with: providers resolved from config when not passed; on failure, stub/degraded artifact (except `GILGroundingUnsatisfiedError` when strict). |
| `docs/rfc/RFC-049-grounded-insight-layer-core.md` | Flow text: “If either provider is not supplied, … legacy path” — should describe `create_gil_evidence_providers` / `summary_provider` instead. |
| `docs/NON_FUNCTIONAL_REQUIREMENTS.md` | Row on GIL evidence: “provider vs legacy” metrics — drop legacy; mention `gi_evidence_path_provider` only (or clarify single path). |
| `docs/TESTING_STRATEGY.md` | Bullet mentioning `test_pipeline.py` “legacy” vs provider — reword to provider path + mocked `create_gil_evidence_providers` where relevant. |

**Optional:** grep `gi_evidence_path_legacy` in docs after edits (should be zero).

## Other “legacy” strings in repo (unrelated to GIL evidence)

Do **not** conflate with GIL; different meanings:

- **`src/podcast_scraper/evaluation/experiment_config.py`** — “legacy mode” =
  glob / `episodes_glob` vs dataset-based eval.
- **`tests/integration/ml_model_cache_helpers.py`** — “legacy path” = old
  Whisper model directory on disk.
- **`config.py`**, **`cli.py`**, **`kg/pipeline.py`**, **`ml_provider.py`**, etc. —
  migration, deprecated params, or historical labels.

A full grep for `legacy` under `src/` and `docs/` is still useful before closing
this WIP.

## Suggested follow-up checklist

1. Update the four doc files above; run `make fix-md`, `make lint-markdown`,
   `make docs`.
2. Skim RFC-049 for any other “fallback to direct ML” language.
3. If product/NFR owners care, align `NON_FUNCTIONAL_REQUIREMENTS.md` with the
   single evidence path story.
