# ADR-088: macOS Local CI Process Safety for ML Workloads

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-074](../rfc/RFC-074-process-safety-ml-workloads-macos.md)
- **Related ADRs**: [ADR-033](ADR-033-stratified-ci-execution.md) (CI layering context)

## Context & Problem Statement

On macOS, parallel **ML model cache probing** and heavy **`readdir()`** traffic against large Hugging
Face caches can interact badly with **APFS kernel locking**, producing **uninterruptible (UE) Python
processes**, pileups when agents retry **`make`**, and in worst cases filesystem metadata damage.
Parse-time **`$(shell …)`** Makefile probes that import **`spacy.load()`** or Transformers made
**every** `make` invocation (including **`make help`**) a potential ML I/O storm.

## Decision

1. **No ML-heavy Makefile parse-time probes** — Model cache checks run **only inside recipes** that
   need them (for example **`make ci`** runs a bounded Python probe before **`_ci_body`**), not as
   **`:= $(shell …)`** at parse time. **`make help`** and lightweight targets must not load Whisper,
   Transformers, or spaCy models.

2. **Lightweight cache checks** — Prefer filesystem presence checks and
**`spacy.util.get_installed_models()`** over **`spacy.load()`** for “is NER installed?” style gates.

3. **`cleanup-processes` before heavy test or CI targets** — Makefile invokes **`cleanup-processes`**
   ahead of **`ci`**, **`test-*`**, and related targets to reduce orphaned **`pytest`** / probe
   processes; patterns avoid killing unrelated long-lived **`serve`** commands (see Makefile comments
   on removed overly broad **`pkill`** regex).

4. **`check-zombie` and `check-spotlight` diagnostics** — Operators and agents can detect UE-state
   PIDs and Spotlight interference without guessing.

5. **Agent or human policy** — Do **not** run multiple **`make ci`** / **`make ci-fast`** /
   **`make test`** concurrently on macOS; after a hung **`make`**, run **`make cleanup-processes`**.
   (Also codified in **`.cursorrules`**.)

6. **Pre-commit timeout** — The hook uses a bounded wall-clock so a wedged subprocess cannot hold
   the developer machine indefinitely (details in RFC-074 and live hook script).

## Rationale

- **Bounds blast radius** — Serialises the riskiest local workflows without changing Linux CI
  behavior materially.
- **Keeps developer machines usable** — Same priority as green CI.
- **Aligns with stratified CI** — Fast vs full gates remain valid; this ADR is about **local process
  economics**, not removing **`stack-test`**.

## Alternatives Considered

1. **Run all ML in Docker on macOS dev** — Helpful optional path; not required as the only fix;
   Makefile hygiene is the baseline.
2. **Disable parallel pytest everywhere** — Too slow on Linux; macOS-specific discipline is enough.
3. **Ignore agent-driven pileup** — Rejected; agent retry patterns were part of incident timelines.

## Consequences

- **Positive**: Routine `make` is safe for quick iteration; `make ci` still validates ML caches
  explicitly.
- **Negative**: Contributors must learn **`cleanup-processes`** / **`check-zombie`** when things
  go wrong.
- **Neutral**: RFC-074 may remain **Draft** for narrative depth; this ADR is the **accepted** decision
  record for what shipped in Makefile + hooks + rules.

## Implementation Notes

- **Makefile**: `ci:` recipe-time cache probe comment block, `cleanup-processes`, `check-zombie`,
  `_ci_body` includes **`stack-test-ml-ci`**
- **Tests**: `tests/integration/ml_model_cache_helpers.py` (filesystem-first transformers cache
  check vs tokenizer load)
- **Policy**: `.cursorrules` process-safety section

## References

- [RFC-074: Process safety for ML workloads on macOS](../rfc/RFC-074-process-safety-ml-workloads-macos.md)
- [Architecture — Process Safety](../architecture/ARCHITECTURE.md) (summary bullet)
