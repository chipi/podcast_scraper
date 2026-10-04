# Post-deploy list — deploy of 2026-10-04

What this deploy ships, what to do around it, and how to check each change live. Every prod WRITE
needs per-instance operator approval; migrations are dry-run → read the frozen set → apply →
verify. Previous list: `POST-DEPLOY-2026-10-02.md` (its 10-03 order was executed; m0018/m0019
applied, the 6 KG-failed episodes repaired).

## What ships (since the deployed `sha-67fea1b`)

| Commit(s) | Change | Live check |
|---|---|---|
| `b55eefd9e` | perf_cache single-flight: a cold digest is built once, not once per retry | P2 |
| `272ca19b8`, `55467df42` | 149 blocking handlers + `/api/usage` run in the threadpool | P3 |
| `4ca10301b` | obs reads the operator api with the read key it derives itself | P4 |
| `6ab9a27f0`, `6ccfe996d` | GI speaker path mints person ids without the title; m0020 re-merges titled ids minted since m0017 | P5, P6 |
| `d850cd33c`, `2f59fde71`, `33163facb`, `2381fdb35`, `3571b7d4c`, `9f10ec837` | analytics page views, Google button, Sign in with Apple (configuration-gated), app-exit reasons | P7 (operator's) |
| `dcdec0b76`, `d47a4e35c`, `841faedba`, `5850f310b`, `78fbeb6ad`, `810151a7b`, `6de7af4e5` | native app / iOS / Android build lanes; `6de7af4e5` also makes a failed docker install fail the build | none on the server; build log only |
| `448151fab` … `11e75c7ab` (#2276) | per-voice naming decision trace in `<transcript>.speakers.diagnostics.json` → `decision_trace`: roster + helpers + both LLM answers (detection raw, resolver verdicts). Observer only — no naming decision changes (gate: 2,319 episodes, old vs new, 0 differ) | N1–N6 |

## Before the deploy

| # | Step | Note |
|---|---|---|
| D0 | Cut-off: push the #2276 commits + docs to main (rebased) | done when this file is on main |
| D1 | Jobs: `70e6aef3` (deepen) RUNNING; QUEUED `2e300536` (deepen), `0c5c2b5f`, nightly `b3d7c810`, DeepMind rederive `851d28e2` (state at 2026-10-04 ~11:40 UTC) | operator decides: let the running one finish or cancel; cancel or keep the queued ones. A deploy restarts the pipeline container — a running job dies mid-episode |
| D2 | `deploy-all-prod` at the pushed sha | operator approval link |

## After the deploy — carried over (P)

| # | Check / step | How | Expected |
|---|---|---|---|
| P1 | Deploy log, all surfaces at one sha | workflow run | green; stack test green |
| P2 | Operator live smoke passes FIRST time | smoke job | the 10-03 smoke failed once on the cold digest; single-flight should make the first run pass |
| P3 | API latency + errors after the threadpool move | VictoriaMetrics / obs `prod_health`, `prod_recent_errors`: p50/p95 per route vs the day before; 5xx rate; `/api/usage` time | no rise; `/api/usage` no longer stalls other routes |
| P4 | obs `prod_recent_runs` returns data end to end | obs MCP tool call | runs listed (was 404 / auth failure) |
| P5 | m0020 | `upgrade run --to 0020 --dry-run` → read the frozen set → apply (`--yes`, approval) → `upgrade verify` | verify clean; then delete its 13 GB snapshot (approval, path-specific) |
| P6 | Titled person ids stay merged | first new GI artifact with a titled speaker | `person:<name>` without the title |
| P7 | Login surfaces (operator's PR) | Google button renders in Google Sans; Sign in with Apple only if configured; one page view per navigation | as the PR describes |

## After the deploy — new for #2276 (N)

| # | Check | How | Expected |
|---|---|---|---|
| N1 | The first ingested episode's sidecar carries the trace | `jq '.decision_trace | {version, degraded}'` on its `.speakers.diagnostics.json` | `version: 1`, `degraded: false` |
| N2 | Both LLM answers are in it | `.decision_trace.inputs.detection` (`detector: VLLMProvider`, `raw`, `corroboration_rejected`), `.inputs.llm_resolution` (`raw`, `verdicts[].outcome`) | present; `raw` is the model's JSON |
| N3 | Completeness | every voice published by name has a step that SET that name (script used for the gate, in the doc) | 0 without |
| N4 | No errors from the new paths | VictoriaLogs / GlitchTip since deploy: `speaker_detection_report`, `last_speaker_detection_raw`, `naming_trace`, `AttributeError` in `episode_processor` / `processing` / `resolution` | none |
| N5 | Size and time | sidecar size and per-episode naming-stage time vs pre-deploy | trace ≲ 12 KB (replay max 12,181 B); no measurable stage-time change |
| N6 | Reprocess path | the next `rederive_only` / `relabel_only` job's sidecar | `decision_trace` present; `inputs.detection` may be `null` (reprocess without detection) — recorded, not an error |
| N7 | Deployed code = gated code | `roster_replay --trace-out` from the IMAGE (no staging) over `/app/output` | `traced_roster_differs` 0, `traced_only_error` 0, `trace_degraded` 0; keep `traces.jsonl` for phase 2 |

## After the checks

| # | Step |
|---|---|
| R1 | Resume / re-queue the deepen shows that did not run; `scripts/audit/show_review.py` on each as it finishes |
| R2 | #2276 phase 2: join replayed traces to the gold labels (dev 101 / val 500) — per-rung precision, failure chains per show. Offline, no deploy needed |
| R3 | #2276: the LLM-rung analysis waits for ingest-time traces (N1–N2) to accumulate — offline replays carry only surviving LLM names |

## Not covered by this list

- Nightly-only checks of the 10-02 list (B8, B9, B13, B14) still stand on the next nightly.
- Load: the trace adds per-episode dict copies and a ≤12 KB sidecar block; not measured under a
  full prod ingest — N5 is the first measurement.
- The order-dependent unit-test flakes seen in the #2276 work (`test_run_manifest`,
  `test_summary_poison_guard`, `test_metadata_generation` dry run) are not investigated; CI may hit
  them.
