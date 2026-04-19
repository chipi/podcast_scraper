# PRD-030: Viewer operator surface — feeds, config, and optional `serve` jobs

- **Status**: Draft
- **Tracking**: [GitHub #626](https://github.com/chipi/podcast_scraper/issues/626)
- **Related RFC**: [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) (Draft)
- **Related UX specs**:
  - [VIEWER_IA.md](../uxs/VIEWER_IA.md) — status bar operational chrome; no placeholder UI
  - [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) — status bar tokens, dialogs/tabs, `data-testid`
  - [UXS-006](../uxs/UXS-006-dashboard.md) — Dashboard → Pipeline (jobs, hygiene, end-of-day clarity)

## Summary

Operators using **`podcast serve`** and the GI/KG viewer need a **single operational place** in the shell (next to corpus path) to:

1. **Curate RSS URLs** in a canonical list file (CLI `--rss-file` parity).
2. **View and edit non-secret pipeline options** (providers, timeouts, and other knobs) in the **same config file the CLI would use**, with a **round trip to the UI** — structured where possible, raw YAML when needed — **without putting API keys or tokens in that file** (secrets **only via environment**).
3. **Run optional pipeline jobs** from the server (Phase 2) and **understand what ran, what failed, and what is stale** — including hygiene for **many jobs**, **stuck subprocesses**, and a **clear end-of-state** at the end of a session or day.

MVP scope is **opt-in**, **default-off** HTTP surfaces + viewer chrome only when capability flags are true. **RFC-065** (CLI live monitor) remains a **separate** surface from HTTP job control.

## Background & Context

Feed URLs and YAML config today are edited out-of-band. That invites **path drift** between the corpus root in the SPA and the files the CLI reads. Multi-feed layout already assumes **one corpus parent** ([RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md)).

**Product decisions (locked for this PRD — see #626 thread):**

- **Config file location:** Prefer the path passed to **`podcast serve --config-file`** when set; otherwise fall back to a **corpus-relative default** file (see RFC-077 for basename and precedence).
- **Secrets:** The operator-editable config file **must not** contain API keys or tokens; **secrets only via environment**; the UI and API **reject** writes that introduce forbidden keys.

## Goals

- **G1 (Feeds)**: Operator can view/edit/save the canonical per-corpus feed list file when `feeds_api` is enabled.
- **G2 (Config)**: Operator can view/edit/save **non-secret** pipeline options in the resolved operator config file when `operator_config_api` (name TBD in RFC) is enabled; changes persist to the **same file** the CLI would load per precedence rules.
- **G3 (Hygiene / jobs)**: When job APIs exist, the product makes it possible to see **terminal vs stuck vs stale** work, **cancel** or **mark** jobs where safe, and **reconcile** so operators are not left guessing “what is still running” at end of day.
- **G4**: **`GET /api/health`** stays small; heavy job state lives on job endpoints / Dashboard — not health polling.
- **G5**: **IA**: Feeds + config entry points live in **operational** chrome (status bar and/or adjacent dialogs), not the left query column ([VIEWER_IA](../uxs/VIEWER_IA.md)).

## Non-Goals

- **NG1**: Storing or displaying **API keys in the operator config file** or round-tripping them through the viewer — out of scope; use env.
- **NG2**: Full **schema-driven form for every `Config` field** in v1 — optional progressive enhancement; v1 may ship **validated YAML subset** + a small set of high-value controls (RFC sizes the slice).
- **NG3**: **Production auth** for `serve` — out of scope; document localhost-only posture.
- **NG4**: Replacing **RFC-065** terminal monitor — distinct product path.

## Personas

- **Local operator**: Adjusts feeds and provider/timeouts for the next run; wants **confidence** no orphan jobs are burning CPU overnight.
- **Power user**: Comfortable with YAML; wants raw edit with **validation errors** surfaced in UI.
- **Reviewer**: Validates phase split, secret policy, and job hygiene acceptance criteria on **#626**.

## User Stories

- _As an operator, I can open **Feeds** and **Config** from the status bar (when enabled), save, and run the CLI or server job using the same files I just edited._
- _As an operator, I cannot save a config file that contains secret keys — I am told to use environment variables instead._
- _As an operator, after many runs, I can see which jobs are **done**, **failed**, or **stale**, and take **reconcile / cancel** actions so the system is clear at end of day._

## Functional Requirements

### FR1: Feed list file (MVP — unchanged intent)

- **FR1.1**–**FR1.4**: As in prior draft — `GET`/`PUT` feeds list; `feeds_api` on health; `podcast serve` opt-in; line file format matches CLI `--rss-file` reader.

### FR2: Operator config file (MVP or 1b — RFC phases)

- **FR2.1**: Server resolves **config path** = `--config-file` if `serve` was started with it, else **default under corpus root** (RFC basename).
- **FR2.2**: **`GET`** returns file content (or structured JSON mirror) suitable for editing **non-secret** fields.
- **FR2.3**: **`PUT`** validates: **reject** body/files containing forbidden secret keys (RFC lists mechanism — denylist vs `Config` secret field inventory).
- **FR2.4**: `GET /api/health` exposes **`operator_config_api`: true** iff routes mounted (same strict client rule as `feeds_api`).
- **FR2.5**: Viewer: **Config** affordance **next to** Feeds (same dialog with tabs, or adjacent buttons — UXS-001); help text names resolved path and secret policy.

### FR3: Jobs (Phase 2 — RFC detail)

- **FR3.1**–**FR3.3**: Job create/list/detail; subprocess contract; refresh on terminal state — per RFC-077.

### FR4: Job hygiene and stale processes (Phase 2)

- **FR4.1**: Every job has a **persistent record** (id, command summary, start time, PID when running, terminal status, error tail pointer) queryable from API and **Dashboard → Pipeline**.
- **FR4.2**: **Stale** definition (e.g. no heartbeat / wall-clock timeout / PID exited but record not updated) is documented; UI shows **Stale** and offers **Mark failed** or **Reconcile** per RFC safety rules.
- **FR4.3**: **Cancel** sends termination to child process when still running (grace period, then force — RFC).
- **FR4.4**: **End-of-day clarity**: a single view or filter “non-terminal” + export or copy summary for operators (lightweight in v1 — can be table filter + doc link).

### FR5: Documentation

- **FR5.1**: [VIEWER_IA](../uxs/VIEWER_IA.md), [UXS-001](../uxs/UXS-001-gi-kg-viewer.md), [SERVER_GUIDE](../guides/SERVER_GUIDE.md) updated with feeds + config + (Phase 2) job hygiene.
- **FR5.2**: [UXS-006](../uxs/UXS-006-dashboard.md) for Pipeline job table, stale/cancel affordances, filters.

## Success Metrics

- **MVP**: Feeds round-trip + config round-trip for a **non-secret** sample config; **PUT** rejected when a secret key is present.
- **Phase 2**: Integration or E2E covers at least one **stale** and one **cancel** path; operator doc explains end-of-day **reconcile** workflow.

## Dependencies

- [PRD-003](PRD-003-user-interface-config.md)
- [PRD-025](PRD-025-corpus-intelligence-dashboard-viewer.md)
- [RFC-008](../rfc/RFC-008-config-model.md) — `Config` validation surface
- [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md), [RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md), [ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)

## Constraints & Assumptions

- **Secrets only in env** — operator file is not a secrets store.
- **Health** remains non-chatty.

## Related Work

- [#626](https://github.com/chipi/podcast_scraper/issues/626), [#606](https://github.com/chipi/podcast_scraper/issues/606), [#50](https://github.com/chipi/podcast_scraper/issues/50)
- [RFC-065](../rfc/RFC-065-live-pipeline-monitor.md) — CLI monitor
- [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md)

## Release Checklist

- [ ] PRD / RFC reviewed; #626 sign-off before Phase 1 code
- [ ] Feeds MVP + tests + docs
- [ ] Operator config MVP + tests + docs
- [ ] Phase 2 jobs + hygiene + UXS-006

## Revision history

| Date | Change |
| ---- | ------ |
| 2026-04-19 | Initial Draft |
| 2026-04-19 | Expanded: operator config (no secrets in file; path = serve `--config-file` or corpus default); job hygiene / stale processes; IA with Feeds+Config |
