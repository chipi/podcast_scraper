# Viewer UI batch plan — #709 / #695 / #694

**Branch:** `feat/viewer-ui-709-695-694`
**Status:** Active (planning complete, implementation not started)
**Scope decision (2026-06-18):** UI trio only. #382 (transformers v5) and #538
(CodeQL TD-001) are explicitly **out** — #382 is a risky ML migration with a
known blocker (v5 removed the `question-answering` pipeline used by extractive
QA; `pyproject.toml` pins `<5` for this reason) and belongs on its own branch.
TD-001 (#538) parks itself per its Option-C recommendation.

---

## 1. Diagnosis — where the space is cramped

The viewer is Vue 3 + Pinia + Tailwind + semantic `--ps-*` tokens, **no UI
library**, native `<dialog>` modals.

- Shell has **4 frozen main tabs** (Digest / Library / Graph / Dashboard);
  adding a 5th is heavyweight (VIEWER_IA + UXS + keyboard).
- 3-column layout; center strip is **~400px** at the 1024px baseline.
- **The real bottleneck:** the Configuration modal (`StatusBar.vue`, the
  "Corpus & API" dialog) is **512×448px** with **4 nested top-tabs**
  (Feeds / Job Profile / Job Configuration / Health).

Two of the three features want to pour more into that 512px modal:

- **#694** — a feed row today is **one line** (URL + "has extra fields" badge).
  The issue wants 5 must-fields + a collapsible Advanced block **per row**.
- **#709** — wants a table (Name | Cron | Enabled | Next run) + cron preview +
  per-row toggle.
- **#695** — the outlier: lives **dashboard-side** (PipelineJobsCard /
  HistoryStrip / ExplorePanel), which has room. Clean modal feature.

## 2. Decisions (locked 2026-06-18)

1. **Restructure the Configuration modal — stays a popup, grows bigger.** Widen
   to `~min(60rem,96vw) × min(40rem,88vh)` and replace the top tab-strip with a
   **left sub-nav rail** (Feeds / Job Profile / Job Configuration / **Scheduled**
   / Health). **Popup is the UXS-correct choice**, not just convenient: the app
   has **no URL routing** (VIEWER_IA:77), so a "dedicated page" would require
   either a routing model (violates the no-routing principle) or a 5th main tab
   (the 4 are frozen — VIEWER_IA:207–244). A dedicated page is the documented
   escalation *only* via a deliberate VIEWER_IA revision if config ever outgrows
   a comfortable modal; we are nowhere near that.
2. **#709 lives inside the Configuration surface** as the new "Scheduled" rail
   item (it edits `scheduled_jobs:` in `viewer_operator.yaml` — keep it with the
   config it edits). UXS-aligned: scheduled feed sweeps are a config concern, and
   the canon mandates a single canonical place per concern (UXS-001:347–359).
3. **Build a small shared-primitive layer** rather than inlining per feature.
4. **Consolidate config OUT of the Dashboard** (the "crammed dashboard" the
   operator sensed). See §2a.

## 2a. UXS review — consolidation principle + Dashboard bleed

**The "all config in Configuration" goal is the existing documented rule**, not
a new preference: UXS-001:347–359 + VIEWER_IA:265–270 require a *single canonical
place per concern* and *no secondary/duplicate config surfaces*; the Operator-YAML
tab even rejects feed keys at the boundary (422/400).

**Dashboard config-bleed audit — exactly one real bleed found:**

- **Index-rebuild actions are scattered across 4 places:** `StatusBar` bolt +
  `StatusBar` index dialog (Configuration — correct), **`DashboardView` →
  `IndexStatusCard` "Update index" / "Full rebuild" buttons** (bleed), and
  **`BriefingCard` `@rebuild-index` emit** (bleed). All call the same
  `indexStats.requestIndexRebuild()` store method — no split-brain, just UI
  scatter.
- **Fix (FOLDED INTO THIS BRANCH — confirmed 2026-06-18):** make Dashboard
  `IndexStatusCard` **status-only** (last rebuild, vector count, "rebuild
  recommended" warning); route the rebuild *action* to the Configuration
  surface. Reconsider the `BriefingCard` emit. Keep the `StatusBar` bolt (it's a
  status indicator with a quick action — acceptable shortcut).

**Everything else is clean:** the Pipeline "Run" button sends **zero inline
params** (runs use last-saved config — correct separation; no run-param UI to
move), and feeds / profile / operator-YAML live *only* in the Config modal.

**#695 correctly stays dashboard-side:** job logs are *observability*, not
config — the consolidation principle does not pull the log viewer into
Configuration.

**UXS-001 drift to reconcile:** UXS-001 documents the config modal as **2 tabs**
(Feeds + Operator YAML), but the code ships **4** (Feeds / Job Profile / Job
Configuration / Health). The restructure PR must reconcile the doc to reality +
the new rail.

**Mandatory UXS deliverables (canon: docs/uxs/index.md:89–99) when this lands:**
update `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md`, add/update Playwright specs +
helpers, and revise the relevant UXS — **UXS-001** (config-modal restructure +
drift fix), **UXS-006** (Dashboard, if `IndexStatusCard` changes), and a
**VIEWER_IA** note on the consolidation. No speculative/placeholder chrome —
each surface ships with its implementation (VIEWER_IA:19–23).

## 3. Shared primitives (build first — each tested in isolation)

| Primitive | Why / who uses it | Reuse notes |
| --- | --- | --- |
| `AppDialog.vue` | #695 modal; **de-dupes 4 existing `<dialog>` impls** (StatusBar ×3, TranscriptViewerDialog) | Mirror StatusBar pattern: native `<dialog>`, `backdrop:bg-black/40`, inner flex wrapper (never flex the `<dialog>`), `@click` backdrop-close, Esc native, `showModal()`/`close()` via template ref |
| `ToggleSwitch.vue` | #709 enable/disable | No toggle exists today; current pattern is copy-pasted `aria-pressed` buttons. Wrap that into one accessible component |
| `RelativeTime.vue` | #709 `next_run_at`; also job-history surfaces | `formatRelativeRunAge` (utils/dashboardTime.ts) exists for text; wrap it in `HoverRichTip` to add absolute-UTC-on-hover |
| `utils/clipboard.ts` | #695 copy log | Extract the duplicated `copyTextToClipboard` from EpisodeDetailPanel + NodeDetail |
| tiny table rows | #709 schedule table | No table abstraction; use `DiagnosticRow`-style rows or a minimal `<tbody>` — do **not** pull in a table lib |

**Already reusable (do not rebuild):** `CollapsibleSection` (Advanced block),
`DiagnosticRow` (the "overriding default = N" hint chip), `HoverRichTip`,
native `<input type="date">` (in DateChip), `useFilterChipPopover`, and the
**3s→10s adaptive tail-polling pattern** in `PipelineJobExplorePanel.vue`
(fingerprint-based backoff + `usePageVisible`).

## 4. Per-feature plan

### #695 — in-app log viewer modal — ✅ DONE (2026-06-18)

Shipped in the working tree: `utils/clipboard.ts` (+ de-duped EpisodeDetailPanel
/ NodeDetail), `shared/AppDialog.vue` (reusable controlled modal — also the base
for the later config restructure), `stores/pipelineJobLog.ts`,
`dashboard/PipelineJobLogDialog.vue` (hosted once in `DashboardView`), and the
three surfaces switched to "View log" buttons. Tests: 3 vitest files +
`e2e/pipeline-job-log-viewer.spec.ts`; `E2E_SURFACE_MAP.md` updated. Full viewer
vitest (2097) + vue-tsc green.

Honest caveats (not regressions — scope calls):

- **`pipelineJobRunDetailsText.ts` left unchanged** — it has no clickable
  affordance (it only embeds `log_relpath` as text in copied run details), so
  there was nothing to "switch to the modal". The real affordance switch is the
  three `.vue` surfaces.
- **Search-within-log not implemented** (issue "Should", not "Must") — rely on
  native Cmd+F for now, per the issue's own fallback.
- **Live→terminal-while-open auto-stop** is heuristic: the dialog has only the
  status snapshot at open, so polling backs off (3→10s) and hard-stops after the
  tail is byte-stable for ~8 polls. The main acceptance concern (no thrash when
  scrolling *already-terminal* jobs) is fully satisfied via snapshot status.

Original design notes:

- New `PipelineJobLogDialog.vue` built on `AppDialog`.
- Body via existing `fetchPipelineJobLogTail(corpusPath, jobId, maxBytes)`.
- Reuse `PipelineJobExplorePanel`'s polling (3–5s while running/queued, stop at
  terminal state); manual Refresh + Copy (new clipboard util) + tail-size
  selector (default 64K). Search-within optional (or rely on native Cmd+F).
- Switch all four surfaces from download-link → "View log" opening the modal:
  `PipelineJobsCard`, `PipelineJobHistoryStrip`, `PipelineJobExplorePanel`,
  `pipelineJobRunDetailsText.ts`. Keep "Download full log" inside the modal.
- **Schema gap to reconcile:** issue references `truncated_head`, but the TS
  response is `{ text, truncated }`. Confirm backend
  (`PipelineJobLogTailResponse` in `server/schemas.py`) before building the
  "showing last N KB" hint — either add the field or scope the hint to
  `truncated`.

### Configuration modal restructure (enables #694 + #709)

- `StatusBar.vue` "Corpus & API" dialog: widen to `w-[min(60rem,96vw)]`,
  convert the top tab-strip to a left sub-nav rail; preserve all existing
  `data-testid`s where possible (Playwright `stack-jobs-flow.spec.ts` and the
  feeds specs depend on them — add new ones, don't rename silently).
- Add "Scheduled" rail item.

### #694 — per-feed overrides (drill-in within the widened modal)

- Feeds pane: clicking a feed opens a focused per-feed editor (back button)
  rather than cramming fields into the row.
- Must-fields: `max_episodes`, `episode_order` (newest/oldest), `episode_offset`,
  `episode_since`, `episode_until` (native date inputs).
- Advanced block via `CollapsibleSection` (retry / delay / circuit-breaker /
  conditional-GET / episode-retry / user_agent) — can stay a raw-YAML textarea
  if individual inputs feel like over-engineering.
- Hint chip (`DiagnosticRow`) "(overriding viewer_operator default = N)".
- **Round-trip rule:** empty per-feed input = field **absent** (never write
  `max_episodes: null`). Preserve unknown keys.
- API unchanged (`PUT /api/feeds`, `RssFeedEntry` already allowlists fields).

### #709 — scheduled jobs (new "Scheduled" rail item)

- `GET /api/scheduled-jobs` already live (#708 shipped — `scheduled_jobs.py`).
- Table: Name | Cron | `ToggleSwitch` (enabled) | `RelativeTime` (next_run_at).
  Disabled → `—`; distinguish "disabled" vs "invalid cron" badge.
- Enable/disable writes back via `PUT /api/operator-config` (already triggers
  `scheduler.reload()`); rewrite only the matching entry's `enabled:` field,
  preserve surrounding comments.
- Cron preview (next 3 fire times) via `cron-parser` while editing.

## 5. Sequencing

Build start (confirmed 2026-06-18): **#695 vertical slice** — extract
`AppDialog` as part of building the log modal, rather than building all
primitives up front. The other primitives (`ToggleSwitch`, `RelativeTime`) get
extracted when their first consumer (#709) is built.

1. **#695 vertical slice:** `utils/clipboard.ts` → `AppDialog.vue` (extracted
   from the `TranscriptViewerDialog` pattern) → `PipelineJobLogDialog.vue`
   (reusing `PipelineJobExplorePanel`'s tail-polling) → switch all four surfaces
   (`PipelineJobsCard`, `PipelineJobHistoryStrip`, `PipelineJobExplorePanel`,
   `pipelineJobRunDetailsText.ts`) → vitest + Playwright spec.
2. **Index-rebuild consolidation** (folded in) — ✅ DONE (2026-06-19).
   `indexStats.requestOpenIndexDialog()` + `dialogOpenNonce` signal; `StatusBar`
   watches it and opens the "Vector index" dialog (now `status-bar-index-dialog`
   with `index-dialog-update` / `index-dialog-full-rebuild` testids).
   `IndexStatusCard` is status-only with a **Manage in Configuration →**
   (`index-status-manage`) button; `BriefingCard` rebuild actions renamed to
   **Open index controls** (`open-index-controls` event) and route to the same
   dialog. Rewrote `dashboard-index-rebuild-mocks.spec.ts`; updated UXS-006 +
   `E2E_SURFACE_MAP.md`. Full vitest (2097) + 3 e2e + vue-tsc green.
   **Dead code removed (2026-06-19):** deleted `CorpusDataWorkspace.vue`,
   `DashboardOverviewSection.vue` (both unmounted — App mounts only
   `DashboardView`; carried a third copy of the rebuild buttons), and the
   now-orphaned `MetricsPanel.vue`. Kept `utils/metrics.ts` (still used by the
   `indexStats` store + has its own test) and the `openCorpusDataWorkspace` e2e
   helper (misnamed but actively opens `DashboardView`). vue-tsc + 2097 vitest
   green after removal.
3. **Config modal restructure** — ✅ DONE (2026-06-19). `StatusBar.vue`
   "Corpus & API" dialog widened to `w-[min(60rem,96vw)]` / `h-[min(36rem,88vh)]`;
   top tab-strip → **left sub-nav rail** (`<nav aria-label="Configuration
   sections">`), all `data-testid`s preserved. Reconciled UXS-001 drift
   (documented as 2 tabs/"Corpus sources" → actual 4 sections/"Corpus & API").
   Updated UXS-001 (+ revision row), `E2E_SURFACE_MAP.md`; added a rail-IA
   assertion to `status-bar-feeds-operator-mocks.spec.ts` (5/5 pass). vue-tsc +
   2097 vitest green. **Scheduled** rail item is NOT added yet (no placeholder
   UI — it ships with #709 in step 5). AppDialog conversion deferred: kept the
   native `<dialog>` to keep the diff tight; can unify later.
4. **#694** drill-in editor — ✅ DONE (2026-06-19). `utils/feedOverrides.ts`
   (split/build/round-trip; empty = inherit; collapse to bare URL; preserve
   unknown keys) + `shell/FeedOverrideEditor.vue` (structured max_episodes /
   episode_order / offset / since / until + collapsible Advanced raw-JSON +
   override hint chip). Wired into StatusBar Feeds Manage as a **Configure**
   drill-in (`sources-dialog-feeds-row-configure-{idx}`). Tests: feedOverrides
   (10) + FeedOverrideEditor (7) vitest + `feed-overrides-mocks.spec.ts` e2e
   (max_episodes on one feed only). UXS-001 + `E2E_SURFACE_MAP.md` updated.
   vue-tsc + 2114 vitest green. Hint chip's "= N" is best-effort (parses an
   explicit `max_episodes:` from the operator YAML body; preset-derived defaults
   show the no-N variant).
5. **#709** Scheduled rail item — ✅ DONE (2026-06-19). New primitives
   `shared/ToggleSwitch.vue` + `shared/RelativeTime.vue` (+ `utils/relativeTime.ts`);
   `utils/cronPreview.ts` (cron-parser v5 — validate + next-3 preview);
   `utils/scheduledJobsYaml.ts` (`toggleScheduledJobEnabled` line-rewrite,
   comment-preserving); `api/scheduledJobsApi.ts`; `shell/ScheduledJobsSection.vue`
   (Name/Cron/Enabled/Next-run table, invalid-cron badge, greyed disabled rows).
   Wired as the **Scheduled** rail item; toggle handled in StatusBar
   (`onScheduledToggle` → operator-config GET/rewrite/PUT → scheduler reload +
   section refresh via nonce). **Added dep: `cron-parser@^5`.** Tests:
   relativeTime (8) / ToggleSwitch (4) / RelativeTime (2) / cronPreview (4) /
   scheduledJobsYaml (6) / ScheduledJobsSection (4) vitest +
   `scheduled-jobs-mocks.spec.ts` e2e. UXS-001 + `E2E_SURFACE_MAP.md` +
   `SERVER_GUIDE.md` (struck the "No Scheduled tab" note) updated. Full suite
   **2142 vitest** + vue-tsc + 13 config-dialog/dashboard e2e green.
   (Cron preview was later also added under the Job Configuration editor — see §7.)

## 7. Polish pass — "do it best" (✅ DONE 2026-06-20)

Five quality items chosen with the operator (time/effort not a constraint).
End state: **vue-tsc clean · 2158 vitest · 32 e2e across all touched surfaces**.

1. **AppDialog consolidation** — all five modals now share `shared/AppDialog.vue`
   (Configuration, Vector-index, Artifact-list, Transcript, pipeline-job-log).
   `AppDialog` gained `bodyClass` (rail layouts) + `closeTestid` (distinct close
   testids per dialog: `sources-dialog-close`, `index-dialog-close`,
   `artifact-list-close`, `transcript-viewer-close`, `pipeline-job-log-close`).
   The "de-dupes 4 dialogs" rationale is now realized. TranscriptViewer unit test
   updated for the shared `<h2>` (no more `#transcript-viewer-title` id).
2. **#709 cron preview in the Job Configuration editor** — new
   `shell/CronSchedulePreview.vue` + `parseScheduledJobsFromYaml`; live next-run
   preview + invalid-cron flag under the operator YAML textarea (the issue's
   literal location). Spec `cron-preview-mocks.spec.ts`.
3. **#695 in-log search** — `utils/textSearch.ts` + a find bar in the log dialog
   (highlight, match count, prev/next, Enter/Shift+Enter).
4. **#694 Advanced structured inputs** — `FEED_ADVANCED_FIELDS` schema +
   grouped typed inputs (`feed-override-adv-{key}`) replacing the raw-JSON-only
   Advanced; a raw-JSON box remains for unmodelled keys (preserved verbatim).
5. **#695 status-aware terminal-stop** — the log dialog now polls `GET /api/jobs`
   alongside the tail and stops auto-refresh on the real running→terminal
   transition (replacing the byte-stable heuristic).

**Bug caught + fixed in the polish e2e sweep:** the default `app-dialog-close`
testid collided once multiple AppDialogs co-existed in the DOM (strict-mode
match) — resolved by the distinct per-dialog close testids above.

## 6. Cross-cutting / gotchas

- **UXS rule:** restructuring the dialog (tab-strip → sub-nav rail) and adding
  the Scheduled section requires a **UXS-001 + VIEWER_IA revision** (canonical
  IA source). Do this in the same PR as the restructure.
- **UXS deliverables (canon docs/uxs/index.md:89–99):** update
  `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` + specs whenever a surface contract
  changes; revise **UXS-006** (Dashboard) for the IndexStatusCard change and
  **UXS-001/VIEWER_IA** for the config restructure.
- **Tests:** vitest (`*.test.ts`, happy-dom) per component; Playwright specs in
  `web/gi-kg-viewer/e2e/`. Mirror `dashboard.spec.ts` (page.route mocks).
- **Pre-push (per project rules):** `npm run build` locally (vue-tsc strict is
  invisible to ci-ui-fast); `make ci-ui-full` (touches testids / chip surfaces /
  adds Playwright + `tests/stack-test/`); `make docs` if any `*.md` changed.
- Strike the "No Scheduled tab in the viewer" note in `SERVER_GUIDE.md` when
  #709 lands.
