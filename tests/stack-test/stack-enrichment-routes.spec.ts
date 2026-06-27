import { expect, test } from '@playwright/test'

/**
 * RFC-088 enrichment HTTP surface — smoke spec against the live stack.
 *
 * Verifies the six read endpoints and the auto-disable re-enable
 * endpoint are mounted under ``enable_jobs_api`` and return well-shaped
 * payloads for a fresh corpus (no enrichment run yet). The companion
 * spec ``stack-enrichment-job-flow.spec.ts`` drives an actual job and
 * asserts the dynamic state.
 */

const CORPUS = '/app/output'

test.describe('stack test — RFC-088 enrichment HTTP routes', () => {
  test('GET /api/enrichment/status returns no-status shape for fresh corpus', async ({
    request,
  }) => {
    const r = await request.get('/api/enrichment/status', { params: { path: CORPUS } })
    expect(r.status()).toBe(200)
    const body = (await r.json()) as { available?: boolean; status?: string; reason?: string }
    // Fresh corpus: no status file → available=false. After a run elsewhere
    // in the session, available may be true; either is acceptable here.
    expect(typeof body).toBe('object')
    if (body.available === false) {
      expect(body.reason).toBeTruthy()
    }
  })

  test('GET /api/enrichment/health returns enrichers map', async ({ request }) => {
    const r = await request.get('/api/enrichment/health', { params: { path: CORPUS } })
    expect(r.status()).toBe(200)
    const body = (await r.json()) as { enrichers?: Record<string, unknown> }
    expect(body.enrichers).toBeDefined()
    expect(typeof body.enrichers).toBe('object')
  })

  test('GET /api/enrichment/metrics returns window + per_enricher', async ({ request }) => {
    const r = await request.get('/api/enrichment/metrics', { params: { path: CORPUS } })
    expect(r.status()).toBe(200)
    const body = (await r.json()) as { window?: string; per_enricher?: Record<string, unknown> }
    expect(body.window).toBe('24h')
    expect(body.per_enricher).toBeDefined()
  })

  test('GET /api/enrichment/metrics?window=1h echoes the window', async ({ request }) => {
    const r = await request.get('/api/enrichment/metrics', {
      params: { path: CORPUS, window: '1h' },
    })
    expect(r.status()).toBe(200)
    const body = (await r.json()) as { window?: string }
    expect(body.window).toBe('1h')
  })

  test('GET /api/enrichment/run-summary returns no-run for fresh corpus', async ({ request }) => {
    const r = await request.get('/api/enrichment/run-summary', { params: { path: CORPUS } })
    expect(r.status()).toBe(200)
    const body = (await r.json()) as { available?: boolean; status?: string }
    // Either no run yet (available=false) or the most recent one — both shapes ok.
    expect(typeof body).toBe('object')
  })

  test('GET /api/enrichment/events returns events array', async ({ request }) => {
    const r = await request.get('/api/enrichment/events', {
      params: { path: CORPUS, limit: 5 },
    })
    expect(r.status()).toBe(200)
    const body = (await r.json()) as { events?: unknown[]; count?: number }
    expect(Array.isArray(body.events)).toBe(true)
    expect(typeof body.count).toBe('number')
  })

  test('POST /api/enrichment/health/{id}/re-enable accepts empty body for unknown id', async ({
    request,
  }) => {
    // For an enricher with no prior health record, re-enable seeds a fresh
    // record with auto_disabled=false. Idempotent — safe to assert success.
    const r = await request.post('/api/enrichment/health/smoke_test_enricher/re-enable', {
      params: { path: CORPUS },
    })
    expect(r.status()).toBe(200)
    const body = (await r.json()) as { enricher_id?: string; auto_disabled?: boolean }
    expect(body.enricher_id).toBe('smoke_test_enricher')
    expect(body.auto_disabled).toBe(false)
  })
})
