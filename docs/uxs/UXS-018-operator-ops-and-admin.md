# UXS-018: Operator Ops and Admin tabs

- **Status**: Active (written 2026-10-05 from the shipped components; the code was the source)
- **Surface:** `web/gi-kg-viewer` — main tabs **Ops** (`OpsView`) and **Admin** (`UsersAdminView`,
  `RankingConfigAdminView`, `GraphAnalyticsAdminView`)
- **Inherits:** [UXS-001](UXS-001-gi-kg-viewer.md) (operator design system — tokens, type, density)
- **Related:** #1128 (viewer roles, user management), ADR-113 (resilience), ADR-142 (LLM gateway),
  #11 B2 (discovery ranking)

These are the operator's control surfaces. Until this spec they had unit tests but no UXS and no
browser coverage. This document describes what shipped; where shipped behaviour looks wrong it says
so under **Open questions** rather than specifying a fix.

## Access

| Role (server-assigned) | Ops tab | Admin tab |
| --- | --- | --- |
| listener | no shell at all (`no-access-message`) | — |
| creator | hidden | hidden |
| admin | `main-tab-ops` | `main-tab-admin` |
| auth disabled (local dev) | shown | hidden |

The tabs are hidden in the UI, and the server enforces the same rule: every `/admin/*`,
ranking-config and graph-events route returns **403** to a non-admin. Hiding a tab is a convenience,
never the guard.

## Ops tab (`ops-view`)

Header **Prod ops** with **Refresh** (`ops-refresh`), disabled while loading and reading
**Refreshing…**. A failed summary shows `ops-error` (`text-danger`). Four blocks, top to bottom:

**1. Source cards.** One card per source (`ops-source-{name}`), always in this order: health,
version, runs, deploys, cost, logs, errors, alerts, traces. Each card has a status
(`ops-status-{name}`) and one summary line:

| Status | Colour | Meaning |
| --- | --- | --- |
| `live` | `text-success` | the source answered |
| `unconfigured` | `text-muted` | not wired for this deploy (the line reads "not configured") |
| `failed` | `text-danger` | wired but erroring (the line carries the error) |

Summary lines: `status: …`, `{code_version} · corpus {sha}`, `N recent runs`,
`N deploys · X% fail`, `$0.0000 (24h)` (or `n/a`), `N error lines (1h)`, `N unresolved`,
`N firing / M`, `N recent traces`. Grid: 1 / 2 / 4 columns at base / `sm` / `lg`.

**2. Prod LLM gateway** (`llm-gateway-panel`). The status (`llm-gateway-status`) reads `live`,
`not configured` or `unreachable`. The body is exactly one of:

- the error (`llm-gateway-error`);
- "VictoriaMetrics not wired for this deploy.";
- "No per-key spend recorded yet.";
- a table (`llm-gateway-keys`, row `llm-key-{alias}`) of key, spend, budget and burn.

Spend under one cent shows two significant figures. A burn of **90 % or more** is `text-danger`.

**3. Resilience** (`resilience-panel`). The status (`resilience-status`) reads **all clear**
(`text-success`) or **backing off** (`text-danger`). The rest of the panel:

- **Reset breakers** (`resilience-reset`) appears only while a breaker is open. It force-closes all
  of them, then re-reads the state.
- One chip per LLM provider (`resilience-breaker-{provider}`). An open breaker is `text-danger` and
  shows its remaining cooldown in seconds.
- Open RSS feeds are listed (`resilience-rss-open`).
- A fuse line (`resilience-fuses`) gives the per-episode and per-run call limits. These are "not
  resettable — fix & rerun".

**4. LLM token & cost usage** (`usage-panel`).

- **Group by** chips (`usage-groupby-{dim}`): provider·model (the default), model, operation,
  episode_id. The active chip uses `bg-accent`, and each click re-fetches.
- When telemetry exists but no token events were recorded, the panel shows `usage-uninstrumented`:
  "cost is unknown, not zero". An absent measurement is never shown as zero.
- Otherwise a totals line (`usage-total`: calls · in · out · cached · cost) and a table
  (`usage-table`, top 20 rows `usage-row`).

Each panel fetches on its own, so one failing source never blanks the others.

## Admin tab

Three sections stacked in one scroll column (`space-y-8`): Users, Discovery ranking, Analytics.

### Users (`users-admin`)

- **Add user** form. Email is required (`new-user-email`); name is optional; role defaults to
  creator. **Add user** (`create-user-button`) is disabled until an email is typed. A created user
  joins the table, and the form resets.
- **Table** sorted by email. Each row (`user-row-{email}`) has:
  - the name (or the email) over the email, with **· you** on the admin's own row;
  - the role select (`role-select-{email}`): listener, creator, admin;
  - the status pill (`active-toggle-{email}`): **Active** (`text-success`) or **Inactive**
    (`text-danger`), which toggles on click;
  - **Delete** (`delete-user-{email}`), behind a confirmation: "Delete {email}? This cannot be
    undone."
- **Self-lockout.** The admin's own row has role, status and Delete disabled. The server rejects the
  same changes.
- While a row's request is in flight, that row's controls are disabled.
- Any failure shows in `users-admin-error` (`role="alert"`). The table keeps its last good state.

### Discovery ranking (`ranking-config-admin`)

- One card per signal (`ranking-signal-{name}`) with:
  - an enabled checkbox (`ranking-enabled-{name}`); a disabled signal's card is dimmed to 60 %;
  - a weight (`ranking-weight-{name}`, step 0.1);
  - one input per signal-specific param (`ranking-param-{name}-{key}`); numeric text is stored as a
    number.
- **Save** (`ranking-config-save`) writes the whole registry and shows **Saved ✓**
  (`ranking-config-saved`). Any later edit clears it. The server merges onto defaults, so a bad save
  cannot empty ranking.
- The loading state is `ranking-config-loading`; the error state is `ranking-config-error`.

### Analytics — Graph (`graph-analytics-admin`)

- **Refresh** (`graph-analytics-refresh`). The totals line reads `{events} events · {users} users`.
- A size table (`ga-size-{nodes|edges|trail}`): min, avg, p50, p95 and max per redraw.
- Two lists sorted by count, highest first: **Actions** (`ga-actions`) and **Node taps by kind**
  (`ga-taps`). Empty, they read "No events yet." and "No taps yet.".
- **Breakage — N** (`ga-breakage`), listed by reason. With none, it reads "None recorded."
  (`text-grounded`).
- **Sessions** (`ga-sessions`): one row per session (`ga-session-{id}`) reading
  `{user} · N ev · size a–b`. With none, it reads "No sessions yet.".
  - Picking a row shows its step-by-step timeline (`ga-timeline`) in plain words: "tapped topic",
    "navigated → person (trail 3)", "redraw · N nodes / M edges", "re-centre (…)", "broke: …".
  - **Replay ▶** (`ga-replay`) loads that session into the **Graph** tab's replay player and
    switches to Graph.
- The loading state is `graph-analytics-loading`; the error state is `graph-analytics-error`.

## Accessibility

- Errors use `role="alert"` (users, ranking, analytics).
- Every control is a real `button`, `select` or `input` with a visible label. Status is never colour
  alone: the pill, status words and chips all carry text.
- Disabled self-row controls use the native `disabled` attribute, so assistive tech announces them
  as unavailable.

## Decided

- **User deletion keeps the browser's native `window.confirm`** (operator, 2026-10-06). The consumer
  app uses an in-app dialog with Cancel focused first (UXS-014). The operator viewer deliberately
  does not need to match it.

## Open questions (shipped behaviour that may not be intended)

- **Ops reads live production data in every environment the API is configured for.** The e2e API
  inherits the dev observability config and returns real gateway spend. The Playwright specs mock
  these routes so they do not depend on it.

## Testing

`e2e/ops-and-admin.spec.ts` owns this surface:

- **Users and Discovery ranking** run against the live e2e API, with unique identities per run.
  Ranking is restored after its test.
- **Ops and graph analytics** are mocked. Their live data is either production telemetry (Ops) or
  empty in a fresh corpus (analytics).

`auth-roles.spec.ts` keeps the role matrix.
