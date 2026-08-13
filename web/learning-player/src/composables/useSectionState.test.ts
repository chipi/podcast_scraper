import { describe, expect, it, vi } from 'vitest'
import { useSectionState } from './useSectionState'

/**
 * Direct tests for the #1591 primitive. It is used by seven sections, so its behaviour under
 * failure is the behaviour of the whole Home page — but until now it was only covered indirectly
 * through HomeView, which tests the composition rather than the contract.
 */
describe('useSectionState', () => {
  it('starts in loading, before anything is fetched', () => {
    const s = useSectionState<string[]>([])
    expect(s.phase.value).toBe('loading')
    expect(s.isLoading.value).toBe(true)
    expect(s.data.value).toEqual([])
  })

  it('records data and becomes ready on success', async () => {
    const s = useSectionState<string[]>([])
    await s.load(async () => ['a', 'b'])
    expect(s.phase.value).toBe('ready')
    expect(s.isReady.value).toBe(true)
    expect(s.data.value).toEqual(['a', 'b'])
  })

  it('a successful EMPTY result is ready, not an error', async () => {
    // The distinction the whole issue turns on: "the system has nothing" and "the request failed"
    // are different states that used to render identically.
    const s = useSectionState<string[]>([])
    await s.load(async () => [])
    expect(s.phase.value).toBe('ready')
    expect(s.isError.value).toBe(false)
  })

  it('a rejection becomes an error and does NOT collapse into the initial value', async () => {
    // `.catch(() => [])` — collapsing failure into emptiness — is the defect this replaces.
    const s = useSectionState<string[]>(['seed'])
    await s.load(async () => {
      throw new Error('boom')
    })
    expect(s.phase.value).toBe('error')
    expect(s.isError.value).toBe(true)
    expect(s.isReady.value).toBe(false)
    // Previous data is left alone rather than being wiped by the failure.
    expect(s.data.value).toEqual(['seed'])
  })

  it('does not reject — callers use `await load()` without try/catch', async () => {
    // Every call site is `void load()` or `await load()`. If this rethrew, each would need its own
    // handler and the ones using `void` would raise unhandled rejections.
    const s = useSectionState<string[]>([])
    await expect(
      s.load(async () => {
        throw new Error('boom')
      }),
    ).resolves.toBeUndefined()
  })

  it('retrying after a failure recovers', async () => {
    const s = useSectionState<string[]>([])
    const fetcher = vi
      .fn<() => Promise<string[]>>()
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce(['recovered'])

    await s.load(fetcher)
    expect(s.phase.value).toBe('error')

    await s.load(fetcher)
    expect(s.phase.value).toBe('ready')
    expect(s.data.value).toEqual(['recovered'])
  })

  it('returns to loading while a retry is in flight', async () => {
    const s = useSectionState<string[]>([])
    await s.load(async () => {
      throw new Error('boom')
    })
    expect(s.phase.value).toBe('error')

    let release!: (v: string[]) => void
    const pending = s.load(() => new Promise<string[]>((res) => (release = res)))
    // The error must clear immediately, or a retry looks like it did nothing.
    expect(s.phase.value).toBe('loading')
    release(['ok'])
    await pending
    expect(s.phase.value).toBe('ready')
  })
})
