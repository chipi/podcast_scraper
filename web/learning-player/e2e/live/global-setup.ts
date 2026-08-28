/**
 * Live-smoke warmup — pay the deployed app's cold start ONCE, before any test.
 *
 * 2026-08-27: the post-deploy smoke ran ~3 min after the containers restarted and the
 * episode page missed its 15s expect budget three times in a row (retries all landed in
 * the same cold window) — a false red on a healthy deploy; the identical suite passed
 * minutes later. The heavy first-hit costs are server-side (catalog scan + slug index +
 * search model load), so one round of warmup requests absorbs them for the whole suite.
 *
 * Never fails the run: warmup errors are logged and swallowed — a down app should be
 * reported by the tests themselves, with their proper assertions and traces.
 */
import type { FullConfig } from '@playwright/test'

export default async function globalSetup(_config: FullConfig): Promise<void> {
  const baseURL = process.env.LIVE_BASE_URL || 'https://closelistening.app'
  const username = process.env.PLAYER_PREVIEW_USER || 'marko'
  const password = process.env.PLAYER_PREVIEW_PASS || ''
  if (!password) return // gated specs will skip anyway; nothing to warm

  const auth = 'Basic ' + Buffer.from(`${username}:${password}`).toString('base64')
  const started = Date.now()
  try {
    const episodes = await fetch(`${baseURL}/api/app/episodes?page_size=15`, {
      headers: { Authorization: auth },
      signal: AbortSignal.timeout(60_000),
    })
    const list = (await episodes.json()) as {
      items?: Array<{ slug: string; status: string; has_bridge: boolean }>
    }
    const ep = list.items?.find((e) => e.status === 'ready' && e.has_bridge)
    const warmups = [
      `${baseURL}/api/app/podcasts`,
      `${baseURL}/api/app/theme-clusters?limit=3`,
      ...(ep ? [`${baseURL}/api/app/episodes/${ep.slug}`] : []),
    ]
    await Promise.allSettled(
      warmups.map((u) =>
        fetch(u, { headers: { Authorization: auth }, signal: AbortSignal.timeout(60_000) })
      )
    )
    console.log(`[live-smoke warmup] done in ${Date.now() - started}ms (${warmups.length + 1} requests)`)
  } catch (err) {
    console.log(`[live-smoke warmup] non-fatal: ${String(err)} (${Date.now() - started}ms)`)
  }
}
