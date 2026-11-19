import { describe, expect, it, vi } from 'vitest'
import { dedupeInFlight } from './inFlightDedupe'

describe('dedupeInFlight', () => {
  it('shares one promise for concurrent callers with the same key', async () => {
    let runs = 0
    let release!: () => void
    const gate = new Promise<void>((r) => {
      release = r
    })
    const p1 = dedupeInFlight('k1', async () => {
      runs += 1
      await gate
      return 42
    })
    const p2 = dedupeInFlight('k1', async () => {
      runs += 1
      await gate
      return 99
    })
    await Promise.resolve()
    expect(runs).toBe(1)
    release()
    const [a, b] = await Promise.all([p1, p2])
    expect(a).toBe(42)
    expect(b).toBe(42)
  })

  it('does not dedupe sequential calls', async () => {
    const fn = vi.fn(async (n: number) => n)
    const a = await dedupeInFlight('seq', () => fn(1))
    const b = await dedupeInFlight('seq', () => fn(2))
    expect(a).toBe(1)
    expect(b).toBe(2)
    expect(fn).toHaveBeenCalledTimes(2)
  })
})
