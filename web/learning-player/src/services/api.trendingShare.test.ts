/**
 * How long a trending answer is shared (2026-10-09).
 *
 * `getTrending` used to keep every answer for the whole session, on the reasoning that trending is
 * corpus-wide. `scope: 'mine'` made that false: "Your trends" stayed as first loaded however much
 * you followed or listened, and — the key carries no user — a second account on the same device got
 * the first account's. The cross-surface e2e spec caught it once its navigation really left Discover.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { _resetTrendingForTests, getTrending } from './api'

function stubFetch(): ReturnType<typeof vi.fn> {
  const f = vi.fn(
    async () =>
      new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
  )
  vi.stubGlobal('fetch', f)
  return f
}

const trendingCalls = (f: ReturnType<typeof vi.fn>) =>
  f.mock.calls.filter(([u]) => String(u).includes('/trending')).length

beforeEach(() => _resetTrendingForTests())
afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('getTrending — sharing', () => {
  it('mine: every call after the first has settled asks the server again', async () => {
    const f = stubFetch()
    await getTrending('topic', 'mine', 20, '3m')
    await getTrending('topic', 'mine', 20, '3m')
    expect(trendingCalls(f)).toBe(2)
  })

  it('mine: concurrent callers still share one request', async () => {
    const f = stubFetch()
    await Promise.all([getTrending('topic', 'mine', 20, '3m'), getTrending('topic', 'mine', 20, '3m')])
    expect(trendingCalls(f)).toBe(1)
  })

  it('corpus: shared for five minutes, then asked again', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    const f = stubFetch()
    await getTrending('topic', 'corpus', 20, '3m')
    await getTrending('topic', 'corpus', 20, '3m')
    expect(trendingCalls(f)).toBe(1)
    vi.setSystemTime(Date.now() + 5 * 60_000 + 1)
    await getTrending('topic', 'corpus', 20, '3m')
    expect(trendingCalls(f)).toBe(2)
  })

  it('a failure is not shared: the next call retries', async () => {
    const f = vi.fn(async () => new Response('boom', { status: 500 }))
    vi.stubGlobal('fetch', f)
    await expect(getTrending('topic', 'corpus', 20, '3m')).rejects.toThrow()
    await expect(getTrending('topic', 'corpus', 20, '3m')).rejects.toThrow()
    expect(trendingCalls(f)).toBe(2)
  })
})
