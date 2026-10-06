import { expect, test, type Page, type Route } from '@playwright/test'
import { signInAsAdmin } from './helpers'

/**
 * UXS-018 — the operator's Ops and Admin tabs, which until 2026-10-05 had unit tests and no browser
 * coverage (the role matrix lives in `auth-roles.spec.ts`).
 *
 * - **Users** and **Discovery ranking** run against the LIVE e2e API: these are the account- and
 *   ranking-mutating surfaces, so a test the server could disagree with is the only kind worth
 *   having. Identities are unique per run; ranking is restored after its test.
 * - **Ops** and **graph analytics** are MOCKED. The e2e API inherits the dev observability config,
 *   so `/api/ops/*` returns real production telemetry (gateway spend, homelab sources) — data a test
 *   must not depend on. Graph analytics is empty in a fresh corpus, so its populated states (sessions,
 *   timeline, replay) need a fixture.
 */
const RUN = Date.now().toString(36)
const ADMIN = 'ada-admin@e2e.local'

async function openAdmin(page: Page): Promise<void> {
  await signInAsAdmin(page)
  await page.goto('/')
  await page.getByTestId('main-tab-admin').click()
  await expect(page.getByTestId('users-admin')).toBeVisible()
}

async function serverUser(page: Page, email: string): Promise<Record<string, unknown> | undefined> {
  const users = (await (await page.request.get('/api/app/admin/users')).json()) as Array<Record<string, unknown>>
  return users.find((u) => u.email === email)
}

test.describe('Admin › Users (live API)', () => {
  test.describe.configure({ mode: 'serial' })
  const email = `uxs018-${RUN}@e2e.local`

  test('the admin row is marked "you" and its controls are locked', async ({ page }) => {
    await openAdmin(page)
    const self = page.getByTestId(`user-row-${ADMIN}`)
    await expect(self).toContainText('· you')
    await expect(page.getByTestId(`role-select-${ADMIN}`)).toBeDisabled()
    await expect(page.getByTestId(`active-toggle-${ADMIN}`)).toBeDisabled()
    await expect(page.getByTestId(`delete-user-${ADMIN}`)).toBeDisabled()
  })

  test('Add user creates the account on the server with the chosen role', async ({ page }) => {
    await openAdmin(page)
    const add = page.getByTestId('create-user-button')
    await expect(add).toBeDisabled() // nothing typed yet
    await page.getByTestId('new-user-email').fill(email)
    const form = page.getByTestId('users-admin').locator('form')
    await form.getByRole('combobox').selectOption('listener')
    await expect(add).toBeEnabled()
    await add.click()

    const row = page.getByTestId(`user-row-${email}`)
    await expect(row).toBeVisible()
    await expect(page.getByTestId(`role-select-${email}`)).toHaveValue('listener')
    await expect(page.getByTestId(`active-toggle-${email}`)).toHaveText('Active')
    await expect(page.getByTestId('new-user-email')).toHaveValue('') // the form resets
    await expect.poll(async () => (await serverUser(page, email))?.role).toBe('listener')
  })

  test('changing the role and deactivating persist on the server', async ({ page }) => {
    await openAdmin(page)
    await page.getByTestId(`role-select-${email}`).selectOption('creator')
    await expect.poll(async () => (await serverUser(page, email))?.role).toBe('creator')

    const status = page.getByTestId(`active-toggle-${email}`)
    await status.click()
    await expect(status).toHaveText('Inactive')
    await expect.poll(async () => (await serverUser(page, email))?.disabled).toBe(true)
    await status.click()
    await expect(status).toHaveText('Active')
    await expect.poll(async () => (await serverUser(page, email))?.disabled).toBe(false)

    // Survives a reload: the table is the server's, not the page's.
    await page.reload()
    await page.getByTestId('main-tab-admin').click()
    await expect(page.getByTestId(`role-select-${email}`)).toHaveValue('creator')
  })

  test('a failed change says so and leaves the row as it was', async ({ page }) => {
    await openAdmin(page)
    await page.route('**/api/app/admin/users/**', async (route: Route) => {
      if (route.request().method() === 'PATCH') {
        await route.fulfill({ status: 500, body: 'boom' })
        return
      }
      await route.continue()
    })
    await page.getByTestId(`role-select-${email}`).selectOption('admin')
    await expect(page.getByTestId('users-admin-error')).toBeVisible()
    await expect.poll(async () => (await serverUser(page, email))?.role).toBe('creator')
  })

  test('Delete asks first: dismiss keeps the user, confirm removes it from the server', async ({ page }) => {
    await openAdmin(page)
    const messages: string[] = []
    page.once('dialog', async (d) => {
      messages.push(d.message())
      await d.dismiss()
    })
    await page.getByTestId(`delete-user-${email}`).click()
    expect(messages[0]).toBe(`Delete ${email}? This cannot be undone.`)
    await expect(page.getByTestId(`user-row-${email}`)).toBeVisible()
    expect(await serverUser(page, email)).toBeTruthy()

    page.once('dialog', (d) => d.accept())
    await page.getByTestId(`delete-user-${email}`).click()
    await expect(page.getByTestId(`user-row-${email}`)).toHaveCount(0)
    await expect.poll(async () => serverUser(page, email)).toBeUndefined()
  })
})

test.describe('Admin › Discovery ranking (live API)', () => {
  test.describe.configure({ mode: 'serial' })

  test('toggling a signal, its weight and a param saves to the server; editing clears "Saved"', async ({
    page,
  }) => {
    await signInAsAdmin(page)
    const original = await (await page.request.get('/api/app/ranking-config')).json()
    try {
      await openAdmin(page)
      const sigs = original.signals as Array<{ name: string; enabled: boolean; weight: number; params: Record<string, unknown> }>
      const withParam = sigs.find((s) => Object.keys(s.params ?? {}).length > 0)!
      const key = Object.keys(withParam.params)[0]!
      const name = withParam.name

      await page.getByTestId(`ranking-enabled-${name}`).setChecked(!withParam.enabled)
      await page.getByTestId(`ranking-weight-${name}`).fill('2.5')
      await page.getByTestId(`ranking-param-${name}-${key}`).fill('7')
      await page.getByTestId('ranking-config-save').click()
      await expect(page.getByTestId('ranking-config-saved')).toHaveText('Saved ✓')

      const stored = (await (await page.request.get('/api/app/ranking-config')).json()) as typeof original
      const s = (stored.signals as typeof sigs).find((x) => x.name === name)!
      expect(s.enabled).toBe(!withParam.enabled)
      expect(s.weight).toBe(2.5)
      expect(s.params[key]).toBe(7) // numeric text is stored as a NUMBER

      // A disabled signal is dimmed; any edit clears the saved mark.
      if (withParam.enabled) {
        await expect(page.getByTestId(`ranking-signal-${name}`)).toHaveClass(/opacity-60/)
      }
      await page.getByTestId(`ranking-weight-${name}`).fill('2.6')
      await expect(page.getByTestId('ranking-config-saved')).toHaveCount(0)
    } finally {
      const r = await page.request.put('/api/app/ranking-config', { data: original })
      expect(r.ok(), `restoring ranking-config returned ${r.status()}`).toBe(true)
    }
  })
})

/* ---- graph analytics (mocked) -------------------------------------------------------------- */

const SUMMARY = {
  total_events: 42,
  users: 3,
  by_action: { graph_node_tap: 20, graph_redraw: 15, graph_rail_nav: 7 },
  node_taps_by_kind: { topic: 12, person: 8 },
  size: {
    samples: 15,
    nodes: { min: 10, avg: 55.5, max: 120, p50: 50, p95: 110 },
    edges: { min: 5, avg: 80.2, max: 300, p50: 70, p95: 260 },
    trail: { min: 1, avg: 2.1, max: 6, p50: 2, p95: 5 },
  },
  breakage: { count: 0, by_reason: {} },
}
const SESSIONS = [{ session_id: 's-1', user_id: 'u_alpha', count: 4, size_min: 10, size_max: 60 }]
const TIMELINE = [
  { action: 'graph_redraw', nodes: 10, edges: 5 },
  { action: 'graph_node_tap', kind: 'topic' },
  { action: 'graph_rail_nav', to_kind: 'person', trail_size: 2 },
  { action: 'graph_broke', reason: 'empty-canvas' },
]

async function mockAnalytics(page: Page, summary: object = SUMMARY, sessions: object[] = SESSIONS): Promise<void> {
  const json = (body: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
  await page.route('**/api/app/graph-events/summary**', (r) => r.fulfill(json(summary)))
  await page.route('**/api/app/graph-events/sessions**', (r) => r.fulfill(json({ sessions })))
  await page.route('**/api/app/graph-events/session/**', (r) => r.fulfill(json({ events: TIMELINE })))
}

test.describe('Admin › Analytics — Graph (mocked)', () => {
  test('totals, size table, lists sorted by count, breakage, and a session timeline in words', async ({ page }) => {
    await mockAnalytics(page)
    await openAdmin(page)
    const ga = page.getByTestId('graph-analytics-admin')
    await expect(ga.getByTestId('ga-total')).toHaveText('42')
    await expect(ga.getByTestId('ga-users')).toHaveText('3')
    await expect(ga.getByTestId('ga-size-nodes')).toContainText('120')
    await expect(ga.getByTestId('ga-actions').locator('li')).toHaveText([
      /graph_node_tap\s*20/,
      /graph_redraw\s*15/,
      /graph_rail_nav\s*7/,
    ])
    await expect(ga.getByTestId('ga-taps').locator('li').first()).toHaveText(/topic\s*12/)
    await expect(ga.getByTestId('ga-breakage')).toHaveText('None recorded.')

    await ga.getByTestId('ga-session-s-1').click()
    await expect(ga.getByTestId('ga-timeline').locator('li')).toHaveText([
      /redraw · 10 nodes \/ 5 edges/,
      /tapped topic/,
      /navigated → person \(trail 2\)/,
      /broke: empty-canvas/,
    ])
  })

  test('Replay ▶ loads the session into the Graph tab\'s replay player', async ({ page }) => {
    await mockAnalytics(page)
    await openAdmin(page)
    await page.getByTestId('graph-analytics-admin').getByTestId('ga-replay').click()
    await expect(page.getByTestId('replay-playpause')).toBeVisible()
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
  })

  test('empty analytics read as empty, not as broken', async ({ page }) => {
    await mockAnalytics(
      page,
      { ...SUMMARY, total_events: 0, users: 0, by_action: {}, node_taps_by_kind: {}, breakage: { count: 0, by_reason: {} } },
      [],
    )
    await openAdmin(page)
    const ga = page.getByTestId('graph-analytics-admin')
    await expect(ga.getByTestId('ga-actions')).toHaveText('No events yet.')
    await expect(ga.getByTestId('ga-taps')).toHaveText('No taps yet.')
    await expect(ga.getByTestId('ga-sessions')).toHaveText('No sessions yet.')
  })
})

/* ---- Ops (mocked) --------------------------------------------------------------------------- */

const OPS = {
  target: 'test',
  live: ['health', 'version', 'cost'],
  unconfigured: ['alerts'],
  failed: ['logs'],
  sources: {
    health: { ok: true, data: { status: 'ok' } },
    version: { ok: true, data: { code_version: '2.7.0', corpus_git_sha: 'abc123' } },
    cost: { ok: true, data: { estimated_cost_usd: 1.23456 } },
    alerts: { ok: false, configured: false },
    logs: { ok: false, configured: true, error: 'VictoriaLogs 502' },
  },
}
const GATEWAY = {
  configured: true,
  reachable: true,
  keys: [
    { key_alias: 'podcast-prod', spend_usd: 22.8, max_budget_usd: 25, burn_ratio: 0.912 },
    { key_alias: 'sandbox', spend_usd: 0.0042, max_budget_usd: 5, burn_ratio: 0.00084 },
  ],
}
const BREAKERS_OPEN = {
  any_open: true,
  llm_breakers: {
    openai: { open: true, recent_failures: 5, cooldown_remaining_seconds: 41.2, trips_total: 2 },
    anthropic: { open: false, recent_failures: 0, cooldown_remaining_seconds: 0, trips_total: 0 },
  },
  rss: { circuit_breaker_open_feeds: ['feed-a'] },
  fuses: { llm_max_calls_per_episode: 40, llm_max_calls_per_run: 800 },
}
const BREAKERS_CLEAR = {
  ...BREAKERS_OPEN,
  any_open: false,
  llm_breakers: { openai: { open: false, recent_failures: 0, cooldown_remaining_seconds: 0, trips_total: 2 } },
  rss: { circuit_breaker_open_feeds: [] },
}
const USAGE = (groupBy: string) => ({
  group_by: groupBy.split(','),
  uninstrumented: false,
  total: { calls: 12, input_tokens: 34000, output_tokens: 5600, cached_input_tokens: 1000, estimated_cost_usd: 0.4321 },
  groups: [{ provider: 'openai', model: 'gpt-x', operation: 'summarize', episode_id: 'e1', calls: 12, input_tokens: 34000, output_tokens: 5600, cached_input_tokens: 1000, estimated_cost_usd: 0.4321 }],
})

async function mockOps(page: Page, opts: { gateway?: object; usageUninstrumented?: boolean } = {}): Promise<{
  resets: string[]
  usageGroupBys: string[]
}> {
  const resets: string[] = []
  const usageGroupBys: string[] = []
  let breakers: object = BREAKERS_OPEN
  const json = (body: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
  await page.route('**/api/ops/summary**', (r) => r.fulfill(json(OPS)))
  await page.route('**/api/ops/llm-gateway**', (r) => r.fulfill(json(opts.gateway ?? GATEWAY)))
  await page.route('**/api/resilience**', (r) => r.fulfill(json(breakers)))
  await page.route('**/api/ops/resilience/reset**', async (r) => {
    resets.push(new URL(r.request().url()).searchParams.get('scope') ?? '')
    breakers = BREAKERS_CLEAR
    await r.fulfill(json({ ok: true }))
  })
  await page.route('**/api/usage**', async (r) => {
    const g = new URL(r.request().url()).searchParams.get('group_by') ?? ''
    usageGroupBys.push(g)
    await r.fulfill(json(opts.usageUninstrumented ? { ...USAGE(g), uninstrumented: true } : USAGE(g)))
  })
  return { resets, usageGroupBys }
}

async function openOps(page: Page): Promise<void> {
  await signInAsAdmin(page)
  await page.goto('/')
  await page.getByTestId('main-tab-ops').click()
  await expect(page.getByTestId('ops-view')).toBeVisible()
}

test.describe('Ops tab (mocked)', () => {
  test('source cards keep their order and say live / unconfigured / failed with a one-line summary', async ({
    page,
  }) => {
    await mockOps(page)
    await openOps(page)
    const cards = page.getByTestId('ops-view').locator('[data-testid^="ops-source-"]')
    await expect(cards).toHaveCount(9)
    await expect(cards.first()).toHaveAttribute('data-testid', 'ops-source-health')
    await expect(cards.last()).toHaveAttribute('data-testid', 'ops-source-traces')
    await expect(page.getByTestId('ops-status-health')).toHaveText('live')
    await expect(page.getByTestId('ops-source-health')).toContainText('status: ok')
    await expect(page.getByTestId('ops-source-version')).toContainText('2.7.0 · corpus abc123')
    await expect(page.getByTestId('ops-source-cost')).toContainText('$1.2346 (24h)')
    await expect(page.getByTestId('ops-status-alerts')).toHaveText('unconfigured')
    await expect(page.getByTestId('ops-source-alerts')).toContainText('not configured')
    await expect(page.getByTestId('ops-status-logs')).toHaveText('failed')
    await expect(page.getByTestId('ops-source-logs')).toContainText('VictoriaLogs 502')
    // Not in any list → failed, never silently "live".
    await expect(page.getByTestId('ops-status-traces')).toHaveText('failed')
  })

  test('LLM gateway: per-key spend, sub-cent precision, and a burn of 90% or more in danger', async ({ page }) => {
    await mockOps(page)
    await openOps(page)
    await expect(page.getByTestId('llm-gateway-status')).toHaveText('live')
    const hot = page.getByTestId('llm-key-podcast-prod')
    await expect(hot).toContainText('$22.80')
    await expect(hot).toContainText('91.2%')
    await expect(hot.locator('td').last()).toHaveClass(/text-danger/)
    const cool = page.getByTestId('llm-key-sandbox')
    await expect(cool).toContainText('$0.0042')
    await expect(cool.locator('td').last()).not.toHaveClass(/text-danger/)
  })

  test('LLM gateway not configured says so instead of a table', async ({ page }) => {
    await mockOps(page, { gateway: { configured: false, reachable: false, keys: [] } })
    await openOps(page)
    await expect(page.getByTestId('llm-gateway-status')).toHaveText('not configured')
    await expect(page.getByTestId('llm-gateway-panel')).toContainText('VictoriaMetrics not wired for this deploy.')
    await expect(page.getByTestId('llm-gateway-keys')).toHaveCount(0)
  })

  test('Resilience: open breakers back off; Reset closes all of them and the panel reads all clear', async ({
    page,
  }) => {
    const calls = await mockOps(page)
    await openOps(page)
    await expect(page.getByTestId('resilience-status')).toHaveText('backing off')
    await expect(page.getByTestId('resilience-breaker-openai')).toContainText('openai · 42s')
    await expect(page.getByTestId('resilience-rss-open')).toContainText('feed-a')
    await expect(page.getByTestId('resilience-fuses')).toContainText('40/episode · 800/run')
    await page.getByTestId('resilience-reset').click()
    await expect(page.getByTestId('resilience-status')).toHaveText('all clear')
    await expect(page.getByTestId('resilience-reset')).toHaveCount(0)
    expect(calls.resets).toEqual(['all'])
  })

  test('Usage: totals and rows; a group-by chip re-fetches by that dimension', async ({ page }) => {
    const calls = await mockOps(page)
    await openOps(page)
    await expect(page.getByTestId('usage-total')).toContainText('12 calls')
    await expect(page.getByTestId('usage-total')).toContainText('$0.4321')
    await expect(page.getByTestId('usage-row')).toHaveCount(1)
    await expect(page.getByTestId('usage-row')).toContainText('openai · gpt-x')
    await page.getByTestId('usage-groupby-operation').click()
    await expect(page.getByTestId('usage-row')).toContainText('summarize')
    expect(calls.usageGroupBys).toContain('operation')
  })

  test('Usage with telemetry but no token events reads "unknown, not zero"', async ({ page }) => {
    await mockOps(page, { usageUninstrumented: true })
    await openOps(page)
    await expect(page.getByTestId('usage-uninstrumented')).toContainText('cost is unknown, not zero')
    await expect(page.getByTestId('usage-total')).toHaveCount(0)
  })
})
