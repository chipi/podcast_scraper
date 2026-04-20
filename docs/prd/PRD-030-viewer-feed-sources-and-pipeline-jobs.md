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

1. **Curate feeds** in the canonical structured corpus file **`feeds.spec.yaml`** (root `{ feeds: [...] }`), with CLI **`--feeds-spec`** parity and the same path the viewer Feeds tab reads and writes (RFC-077 / #626). Legacy **`--rss-file`** line lists remain supported for migration but are not the canonical multi-feed source.
2. **View and edit non-secret pipeline options** in the **same operator YAML file the CLI would use** (`--config-file` / `viewer_operator.yaml`), with a **round trip to the UI**: optional packaged **`profile:`** preset ([GitHub #593](https://github.com/chipi/podcast_scraper/issues/593)) plus **thin overrides** — structured where possible, raw YAML for overrides — **without putting API keys or tokens in that file** (secrets **only via environment**). **Feed URLs** are edited in the canonical **feeds list** (G1), not duplicated in operator YAML (see RFC-077).
3. **Run optional pipeline jobs** from the server (Phase 2) and **understand what ran, what failed, and what is stale** — including hygiene for **many jobs**, **stuck subprocesses**, and a **clear end-of-state** at the end of a session or day.

MVP scope is **opt-in**, **default-off** HTTP surfaces + viewer chrome only when capability flags are true. **RFC-065** (CLI live monitor) remains a **separate** surface from HTTP job control.

## Background & Context

Feed URLs and YAML config today are edited out-of-band. That invites **path drift** between the corpus root in the SPA and the files the CLI reads. Multi-feed layout already assumes **one corpus parent** ([RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md)).

**Product decisions (locked for this PRD — see #626 thread):**

- **Config file location:** Prefer the path passed to **`podcast serve --config-file`** when set; otherwise fall back to a **corpus-relative default** file (see RFC-077 for basename and precedence).
- **Secrets:** The operator-editable config file **must not** contain API keys or tokens; **secrets only via environment**; the UI and API **reject** writes that introduce forbidden keys.

## Goals

- **G1 (Feeds)**: Operator can view/edit/save the canonical per-corpus **`feeds.spec.yaml`** (structured `{ feeds: [...] }` — JSON shape in the viewer editor, YAML on disk) when `feeds_api` is enabled; acceptance requires **one** Feeds surface aligned with the CLI (**`--feeds-spec`** / same resolved path for jobs).
- **G2 (Config)**: Operator can view/edit/save **non-secret** pipeline options (optional **`profile:`** + overrides) in the resolved operator config file when `operator_config_api` is enabled; changes persist to the **same file** the CLI would load per precedence rules. **`GET /api/operator-config`** exposes packaged preset names for the profile picker and may **create** a minimal **`profile: cloud_balanced`** file when it is missing or whitespace-only and that preset is packaged; **`PUT`** rejects top-level feed-list keys (use Feeds API / **`feeds.spec.yaml`**) and secret keys.
- **G3 (Hygiene / jobs)**: When job APIs exist, the product makes it possible to see **terminal vs stuck vs stale** work, **cancel** or **mark** jobs where safe, and **reconcile** so operators are not left guessing “what is still running” at end of day.
- **G4**: **`GET /api/health`** stays small; heavy job state lives on job endpoints / Dashboard — not health polling.
- **G5**: **IA**: Feeds + **config** (status bar label **Operator YAML** / Corpus sources tab) entry points live in **operational** chrome (status bar and/or adjacent dialogs), not the left query column ([VIEWER_IA](../uxs/VIEWER_IA.md)).

## Non-Goals

- **NG1**: Storing or displaying **API keys in the operator config file** or round-tripping them through the viewer — out of scope; use env.
- **NG2**: Full **schema-driven form for every `Config` field** in v1 — optional progressive enhancement; v1 may ship **validated YAML subset** + a small set of high-value controls (RFC sizes the slice).
- **NG3**: **Production auth** for `serve` — out of scope; document localhost-only posture.
- **NG4**: Replacing **RFC-065** terminal monitor — distinct product path.
- **NG5**: **Viewer does not** create new packaged presets under `config/profiles/`, corpus-local profile forks, or “save as new preset” flows — defer to a future RFC if needed.

## Operator file shape (recommended)

- **`profile: <name>`** (optional): loads defaults from packaged `config/profiles/<name>.yaml` before applying other keys in the same file (merge order: preset defaults, then explicit keys in operator YAML override).
- **Overrides**: a **small** set of additional YAML keys (timeouts, `max_episodes`, provider deltas, paths). Avoid pasting an entire preset into the operator file — use **`profile:`** plus deltas.
- **Feeds**: use **FR1** / corpus **`feeds.spec.yaml`** (and **`--feeds-spec`** for CLI and server-spawned jobs when that file exists); do not put `rss`, `rss_url`, `rss_urls`, or `feeds` at the root of operator YAML when using the viewer operator workflow (API rejects them on **PUT**).

## Personas

- **Local operator**: Adjusts feeds and provider/timeouts for the next run; wants **confidence** no orphan jobs are burning CPU overnight.
- **Power user**: Comfortable with YAML; wants raw edit with **validation errors** surfaced in UI.
- **Reviewer**: Validates phase split, secret policy, and job hygiene acceptance criteria on **#626**.

## User Stories

- _As an operator, I can open **Feeds** and **Config** from the status bar (when enabled), save, and run the CLI or server job using the same files I just edited._
- _As an operator, I cannot save a config file that contains secret keys — I am told to use environment variables instead._
- _As an operator, after many runs, I can see which jobs are **done**, **failed**, or **stale**, and take **reconcile / cancel** actions so the system is clear at end of day._

## Functional Requirements

### FR1: Structured feeds file (MVP)

- **FR1.1**: `GET`/`PUT /api/feeds` round-trips the corpus **`feeds.spec.yaml`** document (JSON API body uses **`feeds`**: array of URL strings or objects with **`url`** plus optional allowlisted per-feed overrides; same validation as **`load_feeds_spec_file`**).
- **FR1.2**: `feeds_api` on **`GET /api/health`** when routes are mounted; `podcast serve` opt-in unchanged.
- **FR1.3**: Pipeline jobs subprocess argv includes **`--feeds-spec`** with the resolved absolute path when the file exists, plus **`--config`** to the operator YAML (same merge semantics as manual CLI).
- **FR1.4**: Migration from legacy **`rss_urls.list.txt`** / **`--rss-file`**: operators may convert lines to `{ feeds: ["url", ...] }` once (see **`config/examples/feeds.spec.example.*`** and [MIGRATION_GUIDE](../api/MIGRATION_GUIDE.md)); line lists remain CLI-supported but are not the canonical Feeds API contract.

### FR2: Operator config file (MVP or 1b — RFC phases)

- **FR2.1**: Server resolves **config path** = `--config-file` if `serve` was started with it, else **default under corpus root** (RFC basename).
- **FR2.2**: **`GET`** returns file content plus **`available_profiles`** (sorted packaged preset names for the profile picker; RFC-077).
- **FR2.3**: **`PUT`** validates: **reject** forbidden secret keys **and** top-level feed-source keys (`rss`, `rss_url`, `rss_urls`, `feeds` — RFC-077); atomic write to the resolved path only.
- **FR2.4**: `GET /api/health` exposes **`operator_config_api`: true** iff routes mounted (same strict client rule as `feeds_api`).
- **FR2.5**: Viewer: **Config** affordance (**Operator YAML** tab in Corpus sources — UXS-001) **next to** Feeds; help text names resolved path, secret policy, and shallow validation limits.

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
- [GitHub #593](https://github.com/chipi/podcast_scraper/issues/593) — YAML **`profile:`** preset merge in `Config`

## Constraints & Assumptions

- **Secrets only in env** — operator file is not a secrets store.
- **Health** remains non-chatty.

## Related Work

- [#626](https://github.com/chipi/podcast_scraper/issues/626), [#593](https://github.com/chipi/podcast_scraper/issues/593), [#606](https://github.com/chipi/podcast_scraper/issues/606), [#50](https://github.com/chipi/podcast_scraper/issues/50)
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
| 2026-04-20 | Profiles (#593): operator file = optional `profile:` + thin overrides; feeds only in feed list; NG5 fork preset out of scope; GET `available_profiles`; PUT rejects feed keys at root |
| 2026-04-21 | G5/FR2.5: align “Config” wording with Operator YAML tab; preset list mirrors Config cwd+repo roots |
| 2026-04-20 | Structured corpus feeds: **`feeds.spec.yaml`**, **`--feeds-spec`**, FR1 rewrite; G1/G2 acceptance = one path for viewer + CLI + jobs; deprecate canonical **`rss_urls.list.txt`** |
