# Real-Episode Validation Guide

Unit tests prove code paths route correctly. They do **not** prove the
feature works against real provider responses, real transcripts, or the
real end-to-end pipeline. This guide captures the "test with real episodes
before push" pattern introduced in PR #646, which caught four silently
shipped bugs that all unit tests had missed:

- `llm_pipeline_mode: mega_bundled` referenced dispatch arms that weren't
  wired for GIL/KG skip (would have cost *more* than staged).
- DeepSeek mega-bundle timed out at default HTTP timeout on 5+ min audio.
- `#632` research tiered Gemini/OpenAI/Mistral/Grok as "not capable" for
  mega-bundle; real-episode retest showed all 6 produce comparable results.
- `single_feed_uses_corpus_layout=True` was silently dead code on the CLI
  path — wrapping lived in `service.run()` but the CLI calls
  `workflow.run_pipeline(cfg)` directly.

All four were caught by harnesses living in `scripts/validate/`. None
by unit tests.

## The rule (from CLAUDE.md)

> Before pushing any change that touches a production pipeline stage
> (summarization dispatch, GI/KG extraction, transcription preprocessing,
> audio pipeline, any new `llm_pipeline_mode` value, any profile default),
> the last step before commit is:
>
> 1. Run one real episode end-to-end with the changed code path, using the
>    real provider API keys available in `.env`. There is no "I don't have
>    keys" — check `.env` first.
> 2. Measure the claim numerically. If the change is "fewer LLM calls",
>    count them. If it's "cheaper transcription", measure file size and $.
>    If it's "better KG", count nodes. Unit tests don't measure claims.
> 3. Inspect one artifact by eye. Open the produced `gi.json` / `kg.json`
>    / `metadata.json` / cleaned audio and confirm it looks sane.
> 4. Only then push.

## When to build a real-episode harness

Build one every time you are about to ship:

- A new enum value on `Config` (`llm_pipeline_mode`, `gi_insight_source`,
  `kg_extraction_source`, `transcription_provider`, …).
- A new Config field that feeds into subprocess / external-API behaviour
  (timeouts, preprocessing params, prompt templates, profile values).
- A new profile default that changes how production traffic is routed.
- A change to a pipeline stage that has multiple callers (CLI +
  `service.run` + library).

A unit test that `patch.object(service, "workflow")` is not a substitute —
the bug was in the path that bypasses `service.run()` entirely.

## How to build one

Two templates in `scripts/validate/`:

### `validate_phase3c.py` — dispatch / cost / artifact counts

Use when the change routes through provider methods and you want to prove
call-count / cost / artifact deltas. Pattern:

```python
# 1. Pick transcripts representative of production audio.
TRANSCRIPTS_DIR = _REPO_ROOT / ".test_outputs" / "_validate_phase3c" / "transcripts"

# 2. Build the provider through the real factory.
from podcast_scraper.summarization.factory import create_summarization_provider
provider = create_summarization_provider(cfg)
provider.initialize()

# 3. Instrument every provider method AND underlying HTTP client so you
#    count calls + tokens regardless of which method fired.
counts, tokens = _instrument(provider)

# 4. Drive the real pipeline code (not mocks).
summary_meta, _ = _generate_episode_summary(...)
gi_payload = gi_build_artifact(..., prefilled_insights=...)
kg_payload = kg_build_artifact(..., prefilled_partial=...)

# 5. Assert numerical gates that reflect the feature's claim.
#    (e.g. mega_bundled claims 1 LLM call — assert counts["total"] == 1)
```

### `validate_layout_644.py` — subprocess CLI invocation

Use when the change affects the CLI surface (arg parsing, profile loading,
output directory layout). Pattern:

```python
# 1. Spawn the real CLI as a subprocess with controlled args.
p = subprocess.run(
    [".venv/bin/python", "-m", "podcast_scraper.cli",
     "--output-dir", str(output_dir),
     "--max-episodes", "1",
     "--no-transcribe-missing", "--no-generate-metadata",
     "<rss-url>"],
    capture_output=True, text=True, timeout=300)

# 2. Inspect the resulting on-disk tree.
assert (output_dir / "feeds" / "rss_<slug>" / "run_<ts>").exists()

# 3. Contrast with the opposite state to prove the flag made the difference.
```

### `validate_post_reingestion.py` — re-ingested corpus

Use when you need to validate that a large re-ingested corpus produced the
expected artifact quality. Pattern: run every explore-expansion CLI
command, aggregate `gi.json` / `kg.json` counts across the full corpus,
check soft gates (grounded-insight percentage, quotes-per-insight, etc.).

## Commit the harness with the PR

Real-episode harnesses are **reusable baselines** — the next change to
the same stage has a ready comparison. Commit them to `scripts/validate/`
alongside the code they validate. Don't delete them after the PR merges.

## Cost budget

Real-episode runs hit cloud APIs. Per harness:

- `validate_phase3c.py` on 2 transcripts × 10 configs ≈ **$0.10** total.
- `validate_layout_644.py` uses `--no-transcribe-missing` → **$0** cost.
- `validate_post_reingestion.py` depends on corpus size; CLI commands
  themselves are free, only the re-ingestion is paid.

If a full real-episode harness would cost more than ~$1 per run, trim the
matrix (fewer transcripts, fewer modes, smallest model per provider).

## When unit tests are the right answer

After a real-episode harness exposes a bug and you fix it, **also add a
unit test that locks the wiring in**. The harness caught the bug; the unit
test prevents the regression. Both belong in CI, but they cover different
failure modes:

| Failure mode | Unit test | Real-episode |
| --- | :-: | :-: |
| Method returns wrong shape | ✓ | ✓ |
| Dispatch routes to wrong method | ✓ (if mock matches real shape) | ✓ |
| Provider API response format changed | ✗ | ✓ |
| YAML → Config → dispatch → subprocess wiring has a silent drop | ✗ | ✓ |
| Cost claim doesn't hold on real traffic | ✗ | ✓ |
| Timeout / rate-limit / 503 handling breaks | ✗ | ✓ |
| Entity / topic quality on production audio | ✗ | ✓ |

Pair them. Neither alone is enough for a production-facing change.

## Related

- `CLAUDE.md` — *Final validation before push: real episodes, not just
  unit tests* section.
- `scripts/validate/validate_phase3c.py` — canonical dispatch harness.
- `scripts/validate/validate_layout_644.py` — canonical CLI harness.
- `scripts/validate/validate_post_reingestion.py` — corpus-level harness.
- `docs/guides/UNIT_TESTING_GUIDE.md` — unit test rules (complementary).
- `docs/architecture/TESTING_STRATEGY.md` — unit + integration + e2e + validation.
