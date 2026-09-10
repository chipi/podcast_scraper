import { afterEach, describe, expect, it } from 'vitest'
import { isOffline, useOnline } from './useOnline'

// The composable is a module singleton wired to window events; leave it online for the next test.
afterEach(() => window.dispatchEvent(new Event('online')))

describe('useOnline', () => {
  it('reports online by default (and when the environment cannot tell)', () => {
    expect(useOnline().isOnline.value).toBe(true)
    expect(isOffline()).toBe(false)
  })

  it('flips to offline on the offline event and back on online', () => {
    const { isOnline } = useOnline()
    window.dispatchEvent(new Event('offline'))
    expect(isOnline.value).toBe(false)
    window.dispatchEvent(new Event('online'))
    expect(isOnline.value).toBe(true)
  })

  it('shares one reactive flag across callers (singleton listener)', () => {
    const a = useOnline()
    const b = useOnline()
    window.dispatchEvent(new Event('offline'))
    expect(a.isOnline.value).toBe(false)
    expect(b.isOnline.value).toBe(false)
  })
})
