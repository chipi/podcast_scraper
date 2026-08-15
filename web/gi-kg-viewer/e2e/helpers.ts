import { expect, type Page, type TestInfo } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'

/**
 * Mocked-API sign-in for tests that route `**\/api/**` themselves instead of
 * booting a real backend. Mirrors the ``signInAs`` helper first written
 * inline in ``auth-roles.spec.ts`` — every mocked-API spec since the
 * ``#1128`` auth gate landed must set up ``/api/app/auth/status`` or the
 * app boots to ``<LoginView>`` and every ``getByTestId(...)`` on the
 * shell times out (the failure mode found in ci-ui-full 2026-07-18:
 * 152/158 mocked-API specs timing out at 15 s).
 *
 * Roles — canonical vocabulary from ``src/podcast_scraper/server/app_roles.py``,
 * validated against ``stores/auth.ts`` and the ``v-if`` gates in ``App.vue``:
 *
 * - ``listener`` — Learning Player only. Renders ``<NoAccessView>`` in the
 *   viewer. Use this only when a spec asserts the no-access flow.
 * - ``creator`` — viewer base shell: digest / library / graph. Does NOT
 *   see Dashboard / Ops / Admin (those are ``v-if="auth.isAdmin"`` in
 *   ``App.vue`` lines 675/682/696). Use this for the majority of
 *   mocked-API specs.
 * - ``admin`` — creator + Dashboard / Ops / Admin tabs. Use this for specs
 *   that click any admin-only surface.
 *
 * The generic ``API_FALLBACK`` route below matches any host-rooted
 * ``/api/`` path (but NOT the viewer's own ``/src/api/*.ts`` module URLs)
 * so boot calls (``/api/health``, ``/api/artifacts``, ``/api/app/preferences``,
 * …) do not hang while the spec's own more-specific ``page.route(...)``
 * mocks take precedence per Playwright's LIFO route ordering.
 */
type MockRole = 'admin' | 'creator' | 'listener'

const MOCK_ROLE_USER = (role: MockRole) => ({
  user_id: `u_${role}`,
  email: `${role}@x.io`,
  name: role[0]!.toUpperCase() + role.slice(1),
  role,
  disabled: false,
})

/** Host-rooted `/api/` only — must NOT match viewer's own `/src/api/*.ts` module URLs. */
const MOCK_API_FALLBACK = /^https?:\/\/[^/]+\/api\//

/**
 * Sign in as `role`, and — unless told otherwise — stub the ENTIRE API surface.
 *
 * The `liveApi` opt-out is the migration path for #1619. This suite's Playwright `webServer` has
 * only ever run Vite, with no backend behind it, so `mockSignIn` installs a catch-all that answers
 * every `/api/**` request with `{}` and each spec overrides the handful it cares about. That is why
 * 59 of 68 spec files "mock": they are not mocking a few endpoints, they are running against no
 * server at all.
 *
 * It no longer has to be that way — the same fixture-bootstrapped API the consumer suite uses
 * serves every endpoint this app calls. Pass `{ liveApi: true }` to let requests through to it
 * (`e2e/run-local-stack.sh` starts it), then delete that spec's own `page.route` overrides. The
 * default stays stubbed so the migration can proceed one spec at a time; when few remain, flip it.
 *
 * Auth itself stays stubbed either way: role-based access is what these specs exercise, and
 * driving a real OAuth round-trip per test would be slower and prove nothing about the viewer.
 */
export async function mockSignIn(
  page: Page,
  role: MockRole,
  opts: { liveApi?: boolean } = {},
): Promise<void> {
  /* Fallback FIRST (least specific). Playwright routes are LIFO so any
   * per-test ``page.route(...)`` set up after this call wins. */
  if (!opts.liveApi) {
    await page.route(MOCK_API_FALLBACK, (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }),
    )
  }
  /* Auth gate: enabled=true + the role's user. The App.vue gate reads
   * ``auth.enabled`` and ``auth.canUseViewer``/``auth.isAdmin`` to pick
   * between LoginView / NoAccessView / shell (see App.vue:611-696). */
  await page.route('**/api/app/auth/status', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ enabled: true, user: MOCK_ROLE_USER(role) }),
    }),
  )
}

/** Shell `<h1>` product title; v2 lives in a child span (accessible name includes it). */
export const SHELL_HEADING_RE = /Podcast Intelligence Platform/i

/**
 * The corpus root **as the server resolved it** — for `{ liveApi: true }` specs.
 *
 * Never hardcode a corpus path in a migrated spec: it is the repo-relative
 * `tests/fixtures/app-validation-corpus/v3` when the API runs natively, and `/corpus` when it
 * runs from the container `e2e/run-local-stack.sh` starts (which bind-mounts the same corpus
 * there). Asking the server keeps a spec correct under both.
 */
export async function liveCorpusRoot(page: Page): Promise<string> {
  const resp = await page.request.get('/api/corpus/feeds')
  if (!resp.ok()) throw new Error(`liveCorpusRoot: /api/corpus/feeds returned ${resp.status()}`)
  const body = (await resp.json()) as { path?: string }
  if (!body.path) throw new Error('liveCorpusRoot: /api/corpus/feeds returned no `path`')
  return body.path
}

/**
 * A real per-run `.../metadata` directory inside the live corpus, discovered from the API.
 *
 * The run directory is named for the run that produced it (`run_20260101_000000` in the v3
 * fixture), so it is derived from an artifact's `relative_path` rather than spelled out — a
 * rebuilt corpus renames it and a hardcoded spec would silently stop exercising the hint.
 */
export async function liveFeedMetadataDir(page: Page): Promise<string> {
  const root = await liveCorpusRoot(page)
  const resp = await page.request.get(`/api/artifacts?path=${encodeURIComponent(root)}`)
  if (!resp.ok()) throw new Error(`liveFeedMetadataDir: /api/artifacts returned ${resp.status()}`)
  const body = (await resp.json()) as { artifacts?: { relative_path?: string }[] }
  const rel = (body.artifacts ?? []).find((a) => a.relative_path?.includes('/metadata/'))
    ?.relative_path
  if (!rel) throw new Error('liveFeedMetadataDir: no artifact under a `metadata/` directory')
  return `${root}/${rel.slice(0, rel.lastIndexOf('/'))}`
}

/**
 * Guard for specs that WRITE to the one corpus the stack serves (#1619).
 *
 * `status-bar-feeds-operator-mocks`, `feed-overrides-mocks`, `scheduled-jobs-mocks` and
 * `cron-preview-mocks` each rewrite `feeds.spec.yaml` / `viewer_operator.yaml` through the operator
 * API. `test.describe.configure({ mode: 'serial' })` keeps tests inside one file from seeding over
 * each other, but Playwright has no cross-FILE mutex — two of these files running in different
 * workers would corrupt each other's fixture silently, and the failure would look like a flaky
 * assertion rather than a collision.
 *
 * `e2e/run-local-stack.sh` defaults to `--workers=1`, which makes the collision impossible. This
 * turns the remaining case — someone overriding it — into an immediate, explicit failure instead
 * of a mysterious one.
 *
 * Call once at the top of such a spec's `beforeEach`.
 */
export function requireSerialCorpusAccess(testInfo: TestInfo): void {
  const workers = testInfo.config.workers
  if (workers > 1) {
    throw new Error(
      `${testInfo.titlePath[0] ?? 'this spec'} writes to the shared corpus and cannot run with ` +
        `${workers} workers — two write-path specs in different workers overwrite each other's ` +
        `seeded state. Re-run with --workers=1 (the default in e2e/run-local-stack.sh).`,
    )
  }
}

/** One entry of `GET /api/corpus/feeds` — the fields migrated specs assert on. */
export type LiveFeed = {
  feed_id: string
  display_title: string
  episode_count: number
}

/** One entry of `GET /api/corpus/episodes` — the fields migrated specs assert on. */
export type LiveEpisode = {
  metadata_relative_path: string
  feed_id: string
  feed_display_title: string
  episode_id: string
  episode_title: string
  summary_title: string | null
  summary_preview: string | null
  summary_bullets_preview: string[] | null
  publish_date: string | null
}

/**
 * The live corpus's feed catalogue, as the server lists it.
 *
 * Migrated specs pick a subject from here instead of naming `Mock Show`, so a re-recorded corpus
 * changes the data under the test without changing the test.
 */
export async function liveFeeds(page: Page): Promise<LiveFeed[]> {
  const resp = await page.request.get('/api/corpus/feeds')
  if (!resp.ok()) throw new Error(`liveFeeds: /api/corpus/feeds returned ${resp.status()}`)
  const body = (await resp.json()) as { feeds?: LiveFeed[] }
  const feeds = body.feeds ?? []
  if (feeds.length === 0) throw new Error('liveFeeds: the live corpus lists no feeds')
  return feeds
}

/**
 * The first episode the live Library lists, optionally scoped to one feed.
 *
 * Ordering is the server's (newest first), so this is "whatever the Library shows first" — the
 * same row a user would click.
 */
export async function liveFirstEpisode(
  page: Page,
  opts: { feedId?: string } = {},
): Promise<LiveEpisode> {
  const q = new URLSearchParams({ limit: '1' })
  if (opts.feedId) q.set('feed_id', opts.feedId)
  const resp = await page.request.get(`/api/corpus/episodes?${q.toString()}`)
  if (!resp.ok()) throw new Error(`liveFirstEpisode: /api/corpus/episodes returned ${resp.status()}`)
  const body = (await resp.json()) as { items?: LiveEpisode[] }
  const first = (body.items ?? [])[0]
  if (!first) throw new Error('liveFirstEpisode: the live corpus lists no episodes')
  return first
}

/** One row of `GET /api/corpus/digest` — the fields migrated specs assert on. */
export type LiveDigestRow = {
  feed_id: string
  feed_display_title: string
  episode_id: string
  episode_title: string
  summary_preview: string | null
  publish_date: string | null
  cil_digest_topics: {
    topic_id: string
    label: string
    in_topic_cluster: boolean
    topic_cluster_compound_id: string | null
  }[]
}

/**
 * The Digest's "Recent" rows, as the server assembles them.
 *
 * Note the distinction that matters for #1619: `rows` is catalogue data and is populated, while
 * `topics` (the retrieval-grounded topic bands) is a separate, search-driven field the v3 corpus
 * leaves empty. Specs about rows can migrate; specs about bands cannot.
 */
export async function liveDigestRows(page: Page): Promise<LiveDigestRow[]> {
  const resp = await page.request.get('/api/corpus/digest')
  if (!resp.ok()) throw new Error(`liveDigestRows: /api/corpus/digest returned ${resp.status()}`)
  const body = (await resp.json()) as { rows?: LiveDigestRow[] }
  const rows = body.rows ?? []
  if (rows.length === 0) throw new Error('liveDigestRows: the live digest has no rows')
  return rows
}

/**
 * Sign in as an ISOLATED mock identity, unique per (spec, project).
 * Same shape as ``web/learning-player/e2e/helpers.ts::signInIsolated`` —
 * the player and viewer tier-3 walks share this pattern.
 *
 * The mock OAuth provider honours the ``?as=`` hint (dev/e2e only) and
 * self-completes as ``e2e-<hint>`` — so parallel specs don't share one
 * mock user (which would race on the shared per-user server files).
 * ``who`` should be the spec's name.
 *
 * Server-side redirect only: no UI click, no post-goto timing race. The
 * ``user-menu-button`` testid becoming visible is the post-auth marker
 * — it is the always-visible avatar in the authenticated header
 * (``UserMenu.vue``) and does NOT render on the ``NoAccessView`` screen
 * a role-gated user would land on. The ``user-menu-signout`` marker
 * used by the learning-player's version of this helper does NOT work
 * for the viewer — the viewer's Sign out lives inside a click-to-open
 * menu (``v-if="open"``), so it isn't rendered until the avatar is
 * clicked. The avatar button IS.
 *
 * ``make serve`` must be started with:
 *   APP_OAUTH_PROVIDER=mock
 *   APP_SIGNUP_MODE=open
 *   APP_ADMIN_EMAILS=e2e-<hint>@e2e.local  (or the exact email the
 *     mock provider synthesizes — see ``MockOAuthProvider.exchange_code``
 *     in ``src/podcast_scraper/server/app_oauth.py``)
 *   APP_SEED_USERS_FILE=config/dev-seed-users.json
 * Otherwise the ``access_policy`` gate returns 403 or the
 * ``NoAccessView`` renders.
 */
export async function signInIsolated(
  page: Page,
  who: string,
  testInfo: TestInfo,
): Promise<void> {
  const id = `${who}-${testInfo.project.name}`.toLowerCase().replace(/[^a-z0-9-]/g, '')
  /* ``?grant=creator`` promotes a new/listener identity to ``creator`` at
   * callback (see ``app_auth.py::app_auth_login``) — enough for the base
   * shell (digest/library/graph) that tier-3 walks exercise. Only
   * ``creator`` is grantable this way; ``admin`` still requires an
   * explicit ``APP_ADMIN_EMAILS`` entry and is never needed for these
   * walks. */
  await page.goto(
    `/api/app/auth/login?as=${encodeURIComponent(id)}&grant=creator`,
  )
  await expect(page.getByTestId('user-menu-button')).toBeVisible()
}

/**
 * Sign in as the fixed ADMIN identity — needed by tests that click the
 * Dashboard / Ops / Admin nav buttons, which are gated on
 * ``auth.isAdmin`` in ``App.vue`` (see ``main-tab-dashboard`` etc.).
 *
 * The ``ada-admin`` hint maps to ``ada-admin@e2e.local`` (see the
 * ``MockOAuthProvider`` synth rule). ``make serve-for-validation``
 * bakes that email into ``APP_ADMIN_EMAILS`` — the callback then
 * lands the session in the ``admin`` role. ``?grant=creator`` is
 * ignored for admin-listed emails (the admin role wins).
 *
 * All admin-gated tests share this ONE identity because parallel
 * mutations against admin-only surfaces would race. The v4 dashboard
 * walk / P5.x / P7.x tab-switch patterns only READ (nav click + chip
 * pick), so a shared admin identity is safe.
 */
export async function signInAsAdmin(page: Page): Promise<void> {
  await page.goto('/api/app/auth/login?as=ada-admin')
  await expect(page.getByTestId('user-menu-button')).toBeVisible()
}

/** Header nav (Digest / Library / Graph / Dashboard) — scope clicks to avoid substring clashes (e.g. main-tab Library vs “Load into graph”). */
export function mainViewsNav(page: Page) {
  return page.getByRole('navigation', { name: 'Main views' })
}

/**
 * **Dashboard** tab — waits for the briefing card (`data-testid="briefing-card"`).
 */
export async function openCorpusDataWorkspace(page: Page): Promise<void> {
  await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
  await page.getByTestId('briefing-card').waitFor({ state: 'visible' })
}

/** Bottom status bar corpus path field (`data-testid="status-bar-corpus-path"`). */
export function statusBarCorpusPathInput(page: Page) {
  return page.getByTestId('status-bar-corpus-path')
}

/**
 * Offline graph load: force /api/health to fail, open **Graph**, then load the CI
 * fixture via the status bar **Files** / hidden file input.
 */
/**
 * Closes the first-run graph gesture overlay when it is visible so pointer and
 * keyboard tests can reach Cytoscape (overlay is above `.graph-canvas`).
 */
export async function dismissGraphGestureOverlayIfPresent(page: Page): Promise<void> {
  const btn = page.getByTestId('graph-gesture-overlay-dismiss')
  try {
    await btn.waitFor({ state: 'visible', timeout: 3000 })
  } catch {
    return
  }
  // ``force`` avoids flakes when a parent (e.g. ``<html>`` during transition) briefly intercepts hits.
  await btn.click({ force: true })
}

export async function loadGraphViaFilePicker(page: Page): Promise<void> {
  await page.route('**/api/health**', async (route) => {
    await route.abort('failed')
  })

  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

  await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

  const fileInput = page.getByTestId('status-bar-local-file-input')
  await fileInput.setInputFiles(GI_SAMPLE_FIXTURE)

  await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
}

// ---------------------------------------------------------------------------
// #1209 — viewer e2e harness helpers.
// See docs/guides/E2E_TESTING_GUIDE.md §Handoff FSM invariants for the
// contract these codify.
// ---------------------------------------------------------------------------

/**
 * #1209 H4 — invariant-poll the graph handoff FSM until it reaches
 * ``state === 'ready'``, or the timeout elapses.
 *
 * Prefer this over ``page.waitForTimeout(1500)`` for FSM-driven waits.
 * Under parallel-worker CPU contention the wall-clock is noisy (Tier-2
 * P2.5 flake, 2026-07-18); polling the FSM state directly is deterministic
 * and returns as soon as the transition completes. The FSM's own
 * stuck-detector fires at ``STUCK_TIMEOUT_MS = 15_000``, so a 10-second
 * poll timeout leaves 5 s of headroom.
 */
export async function waitForFsmReady(
  page: Page,
  opts: { timeoutMs?: number } = {},
): Promise<void> {
  await page.waitForFunction(
    () => {
      const w = window as unknown as { __GIKG_FSM__?: { state: string } }
      return w.__GIKG_FSM__?.state === 'ready'
    },
    undefined,
    { timeout: opts.timeoutMs ?? 10_000 },
  )
}

/**
 * #1209 H4 — invariant-poll the graph handoff FSM until it reaches
 * a specific target state. General form of ``waitForFsmReady``. Use
 * ``waitForFsmReady`` unless you specifically need a non-``ready`` state
 * (e.g. asserting the FSM sits in ``loading_fetch`` before an artifact
 * arrives).
 */
export async function waitForFsmState(
  page: Page,
  targetState: string,
  opts: { timeoutMs?: number } = {},
): Promise<void> {
  await page.waitForFunction(
    (s) => {
      const w = window as unknown as { __GIKG_FSM__?: { state: string } }
      return w.__GIKG_FSM__?.state === s
    },
    targetState,
    { timeout: opts.timeoutMs ?? 10_000 },
  )
}

/**
 * #1209 H3 — reset USERPREFS-1 to an empty payload so tests don't leak
 * per-user state across walks.
 *
 * Real bite (Tier-3 walk, 2026-07-17): V-G1 as admin flipped
 * graph-load-mode to Top-down + persisted. V4 as creator inherited
 * Top-down and hit "no cluster compound" → assertion failure.
 * Every walk that touches USERPREFS-1-synced state (theme, panels,
 * lens flags, graph load mode, graph-legend collapsed) should call this
 * in ``test.beforeEach`` after ``mockSignIn`` so the server-side
 * per-user prefs start clean.
 *
 * Uses PUT (replace) with an empty preference map rather than DELETE — the
 * server API doesn't expose DELETE (PUT is idempotent + safe).
 *
 * #1619: the body must be ``{ preferences: {} }``, not ``{}``. The bare object was rejected with
 * **422 `body.preferences: Field required`** — this helper had never run against a real server,
 * because every spec that called it was mocked, so the wrong shape went unnoticed. That is the
 * class of defect the migration exists to find.
 */
export async function resetUserPreferences(page: Page): Promise<void> {
  const resp = await page.request.put('/api/app/preferences', { data: { preferences: {} } })
  if (!resp.ok() && resp.status() !== 401) {
    // 401 = not signed in (caller may reset before sign-in for a clean
    // starting state; that's fine). Any other non-2xx is a real bug.
    throw new Error(`resetUserPreferences: PUT /api/app/preferences returned ${resp.status()}`)
  }
}

/**
 * #1209 H1 — read the per-user graph_events.jsonl log(s) from a corpus.
 *
 * Under auth, graph analytics events land at
 * ``<CORPUS>/.app/users/u_<hash>/graph_events.jsonl`` — one file per
 * per-user hash. Multiple test runs on the same corpus accumulate into
 * these files, so filter by ``session_id`` (from
 * ``__GIKG_ANALYTICS__.sessionId``) to isolate a specific run's events.
 *
 * Callers use this from Node.js (Playwright test bodies), not the
 * browser. Signature intentionally takes fs+path as args so the helper
 * itself doesn't force a filesystem import into every consumer.
 */
export function readGraphEventsLog(args: {
  corpusPath: string
  sessionId?: string
  fs: { readFileSync: (p: string, enc: string) => string; existsSync: (p: string) => boolean; readdirSync: (p: string) => string[] }
  path: { join: (...parts: string[]) => string }
}): string[] {
  const usersDir = args.path.join(args.corpusPath, '.app', 'users')
  if (!args.fs.existsSync(usersDir)) return []
  const lines: string[] = []
  for (const uidDir of args.fs.readdirSync(usersDir)) {
    const p = args.path.join(usersDir, uidDir, 'graph_events.jsonl')
    if (!args.fs.existsSync(p)) continue
    const content = args.fs.readFileSync(p, 'utf-8')
    for (const line of content.split(/\r?\n/)) {
      if (!line) continue
      if (!args.sessionId) {
        lines.push(line)
      } else if (line.includes(`"${args.sessionId}"`)) {
        lines.push(line)
      }
    }
  }
  return lines
}
