# Log Hygiene Audit — 2026-04-27

**Issue:** [#679](https://github.com/chipi/podcast_scraper/issues/679) sub-task C
**Branch:** `feat/rfc-081-phase-1-prep`
**Author:** Marko + Claude
**Status:** Audit complete; sign-off checklist pending operator review.

## Why this exists

[RFC-081](../rfc/RFC-081-pre-prod-environment-and-control-plane.md)
Phase 1 ships container logs to **Grafana Cloud Loki** via the
Grafana Agent. Before any logs leave the host (Codespace or future
VPS), we need to confirm they don't carry:

- API keys / tokens
- Operator config dumps (which contain provider keys)
- Personally identifiable info (User-Agent / Referer / IPs)
- Copyrighted-text fragments (transcript bodies, full episode titles
  in unusual quantity, RSS metadata verbatim)
- Stack traces with sensitive local-variable context
- Unstructured `f"…{arbitrary_dict}"` dumps that are landmines

This is the audit. Issue [#680](https://github.com/chipi/podcast_scraper/issues/680)
(Grafana Cloud wiring) is **explicitly gated on this checklist closing**.

## Scope

Read-only survey across three log-emitting surfaces. Test files
(`tests/**`) excluded — log hygiene there is not the publish gate.

| Surface | Status |
|---|---|
| FastAPI api stdout (server + routes) | Clean with one risky pattern |
| Pipeline subprocess stdout/stderr (CLI + workflow + providers) | Partially clean — two action items |
| Viewer Nginx access logs | Clean (no `access_log` directive → default no-logging) |

## Findings

### API surface (FastAPI)

**Confirmed leaks: none.**

**Risky patterns (audit periodically):**

- `logger.exception()` calls capture full traceback with local
  variables (Python's default). File paths + line numbers + type info
  are fine; risk surfaces if a frame happens to contain a bearer
  token or `api_key` variable. Affected:
  - [`src/podcast_scraper/server/pipeline_jobs.py:497`](../../src/podcast_scraper/server/pipeline_jobs.py#L497)
  - [`src/podcast_scraper/server/routes/index_stats.py:114`](../../src/podcast_scraper/server/routes/index_stats.py#L114)
  - [`src/podcast_scraper/server/routes/index_rebuild.py:60`](../../src/podcast_scraper/server/routes/index_rebuild.py#L60)
- Reverse-proxy headers are *available* but not logged by default.
  Confirm Nginx config never grows an `access_log` directive in
  production. See [`docker/viewer/nginx.conf`](../../docker/viewer/nginx.conf).

**Clean patterns** (good examples to mirror):

- [`src/podcast_scraper/server/operator_config_security.py`](../../src/podcast_scraper/server/operator_config_security.py)
  rejects PUTs that contain `*_api_key`, `password`, `secret` keys.
  Prevents accidental config serialisation leaks.
- 380+ call sites use `format_exception_for_log()` /
  `redact_for_log()` to strip Bearer tokens / `sk-*` keys / `api_key`
  patterns / passwords from exception strings before logging.
- Server routes log targeted fields (paths, counts, IDs), not
  arbitrary config objects.

### Pipeline subprocess surface

**Confirmed leaks (action required):**

1. **Episode titles logged at INFO level — verbatim from RSS**

   File: [`src/podcast_scraper/workflow/episode_processor.py`](../../src/podcast_scraper/workflow/episode_processor.py)
   Lines: 320, 340, 1613, 1754 (`logger.info(f"[{idx}] no transcript
   for: {episode.title}")` and similar).

   **Why this is a leak class:** episode titles come from third-party
   RSS feeds. They may contain copyrighted text fragments, PII (guest
   names), or advertiser content that we are not licensed to
   redistribute. These INFO logs persist to `.viewer/jobs/<id>.log`
   and stream back via `/api/jobs/<id>/log`.

   **Recommended fix:** truncate to ~50 chars + log title hash for
   correlation. Helper goes in `src/podcast_scraper/utils/log_safe.py`
   or alongside the existing redaction utils.

2. **`multi_feed_batch` JSON dump may carry unfiltered incident data**

   File: [`src/podcast_scraper/workflow/corpus_operations.py:333`](../../src/podcast_scraper/workflow/corpus_operations.py#L333)
   Pattern: `logger.info("multi_feed_batch: %s", json.dumps(payload, ...))`.

   Error messages in `payload` flow through `redact_for_log()` (good),
   but `batch_incidents` dict may carry incident descriptions written
   by upstream paths that don't redact.

   **Recommended fix:** audit a sample of `corpus_incident.jsonl`
   files for unredacted content; if clean, confirm with a regression
   test; if not, add explicit redaction at the dump site.

**Risky patterns (audit + harden):**

- Provider modules (`gemini`, `deepseek`, `grok`, `anthropic`,
  `openai`, `mistral`) use `logger.<...>(..., exc_info=True)` at
  DEBUG level. Full tracebacks at DEBUG are fine for local dev but
  landmines if `log_level: DEBUG` ever flips on in pre-prod via
  `viewer_operator.yaml`. **Recommended:** add a server-side
  validator that rejects DEBUG log levels for the api when running
  in pre-prod.

- `src/podcast_scraper/service.py:146` — `logger.error(..., exc_info=True)`
  on feed failure. Exception is pre-redacted (line 143), but
  traceback frames still capture. Low risk because feed-level
  exceptions are usually network/file errors, but provider API
  exceptions could leak bearer tokens in frames.

**Clean patterns:**

- Transcript / response text is consistently logged by length, not
  content (`response_text[:200]` etc.).
- `corpus_multi_feed_summary` is a structured log with counts +
  metadata, not bodies.

### Viewer Nginx access surface

**Status: clean.**

[`docker/viewer/nginx.conf`](../../docker/viewer/nginx.conf) has no
explicit `access_log` directive, so Nginx uses its default behaviour
which is **no request logging**. Reverse-proxy headers (`X-Real-IP`,
`X-Forwarded-For`, `X-Forwarded-Proto`) are set on upstream requests
but not written to disk by Nginx itself.

**Gate:** if a future PR ever adds `access_log /path/to/log;`, it
must come with an explicit log-format declaration that redacts
User-Agent, Referer, and any cookies. Add a CI lint guard to flag
new `access_log` directives for review.

### Cross-cutting findings

**Structured-logging adoption:** mixed. Most server / workflow code
uses parameterised `logger.info("msg %s", var)` (good). Some
f-string log lines exist (`logger.info(f"[{idx}] ...")`); they're
limited in scope but each is a future audit cost.

**Exception handling:** `format_exception_for_log()` /
`redact_for_log()` are widely adopted (380+ call sites). The
redaction patterns cover Bearer tokens / `sk-*` / `api_key` / passwords.

**JSON-structured stdout:** none currently. Logs are emitted as
plain text. Loki will ingest these as strings; structured-extraction
would require either:

- Promtail pipeline stages with regex parsers (Loki-side cost), or
- Switch to a structured-logging emitter at the source (e.g.,
  `structlog` + JSON renderer; api-only or full repo).

For Phase 1 this is acceptable — Loki strings are searchable and the
volume is hobby-scale. **Defer the structured-logging migration to a
follow-up RFC** if log-volume / query-cost grows enough to justify it.

## Sign-off checklist (gates issue [#680](https://github.com/chipi/podcast_scraper/issues/680))

The operator must confirm each item before Grafana Agent / Loki
log-shipping turns on. Items 1, 3, 4 are blockers; items 2, 5, 6, 7
are audits.

### Blockers

- [ ] **1. Episode-title redaction shipped.** Add a helper that
  truncates to N chars + appends a hash; replace the 4 call sites in
  `episode_processor.py`. Confirm `.viewer/jobs/*.log` no longer
  contains verbatim third-party titles. Estimated effort: 1 hour.
- [ ] **3. log_level pinned to INFO or higher in pre-prod.** Add a
  startup-time validator in api that refuses DEBUG when an env var
  like `PODCAST_ENV=preprod` is set. Avoids accidental
  DEBUG-mode-in-production trace leakage. Estimated effort: 30 min.
- [ ] **4. Verify `logger.exception()` frames are safe in api.**
  Either spot-check by triggering each path or add a custom log
  handler that strips locals from tracebacks. The 3 affected files
  are listed above. Estimated effort: 1-2 hours depending on the
  approach chosen.

### Audits (no code change required, just confirm)

- [ ] **2. Spot-check `corpus_incident.jsonl`** files for unredacted
  user/operator content. Take 10 recent files from `.test_outputs/`
  or developer corpus dirs. Pass = no API keys, no full transcript
  bodies, no operator config dumps.
- [ ] **5. Confirm Nginx `access_log` remains disabled** in
  production deployment. Add a CI lint guard for `access_log`
  directives in `docker/viewer/nginx.conf` so future PRs that
  re-enable it require an explicit redact-format.
- [ ] **6. Spot-check 10 recent `.viewer/jobs/*.log`** files for:
  - Full transcript bodies at INFO level (should not appear)
  - Full operator config dumps (should not appear)
  - Stack traces with `api_key=...` or `sk-...` substrings
- [ ] **7. Test config-validation flow.** PUT a `viewer_operator.yaml`
  containing `openai_api_key: "test"` to `/api/operator-config` and
  confirm rejection. Confirms the static guard against accidental
  config persistence.

When all 7 items are checked off, log-shipping in #680 can light up.
Until then, Grafana Agent ships **metrics only** (no Loki sink).

## Confidence + caveats

- **Audit was read-only.** No source files were modified. The
  recommended fixes (items 1, 3, 4) are sketched but not implemented
  — done-pile work for a separate PR after operator review.
- **Confidence on completeness: 75%.** Used `grep` + spot-reads
  across the obvious leak surfaces. Edge cases (e.g., a third-party
  library auto-logging at INFO with our config object as context)
  weren't exhaustively swept. The sign-off checklist's audit items
  (2, 5, 6, 7) are the safety net for things this scan missed.
- **Loki is forgiving but not infinite.** Even a clean log stream
  benefits from structured emission long-term. The deferred
  `structlog` migration should land before this stack grows past
  hobby scale.
