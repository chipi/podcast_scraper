// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useGraphAnalyticsStore } from './graphAnalytics'
import * as api from '../api/graphAnalyticsApi'

vi.mock('../api/graphAnalyticsApi', () => ({ postGraphEvents: vi.fn() }))
const post = vi.mocked(api.postGraphEvents)

describe('useGraphAnalyticsStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    vi.useFakeTimers()
  })
  afterEach(() => vi.useRealTimers())

  it('buffers a tracked event (action + payload + ts) without posting immediately', () => {
    const s = useGraphAnalyticsStore()
    s.track('node_tap', { id: 'topic:a' })
    expect(s.buffer).toHaveLength(1)
    expect(s.buffer[0]).toMatchObject({ action: 'node_tap', id: 'topic:a' })
    expect(typeof s.buffer[0].ts).toBe('number')
    expect(post).not.toHaveBeenCalled()
  })

  it('ignores an empty action', () => {
    const s = useGraphAnalyticsStore()
    s.track('')
    expect(s.buffer).toHaveLength(0)
  })

  it('flush posts the batch and clears the buffer', () => {
    const s = useGraphAnalyticsStore()
    s.track('a')
    s.track('b')
    s.flush()
    expect(post).toHaveBeenCalledTimes(1)
    expect(post.mock.calls[0][0].map((e) => e.action)).toEqual(['a', 'b'])
    expect(s.buffer).toHaveLength(0)
  })

  it('flush is a no-op when the buffer is empty', () => {
    useGraphAnalyticsStore().flush()
    expect(post).not.toHaveBeenCalled()
  })

  it('auto-flushes on the timer', () => {
    const s = useGraphAnalyticsStore()
    s.track('a')
    expect(post).not.toHaveBeenCalled()
    vi.advanceTimersByTime(10_000)
    expect(post).toHaveBeenCalledTimes(1)
  })
})
