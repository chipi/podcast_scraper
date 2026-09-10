import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { ApiError } from '../services/api'
import * as outbox from '../services/outbox'
import { useInterestsStore } from './interests'

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

describe('interests store', () => {
  it('ensureLoaded() pulls the followed tokens once', async () => {
    const spy = vi.spyOn(api, 'getUserInterests').mockResolvedValue(['tc:ai', 'person:jane'])
    const s = useInterestsStore()
    await s.ensureLoaded()
    await s.ensureLoaded() // cached — no second fetch
    expect(s.has('person:jane')).toBe(true)
    expect(s.has('topic:absent')).toBe(false)
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('toggle() follows then unfollows, server response authoritative', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'addInterest').mockResolvedValue(['topic:ai'])
    vi.spyOn(api, 'removeInterest').mockResolvedValue([])
    const s = useInterestsStore()
    await s.toggle('topic:ai')
    expect(s.has('topic:ai')).toBe(true)
    expect(api.addInterest).toHaveBeenCalledWith('topic:ai')
    await s.toggle('topic:ai')
    expect(s.has('topic:ai')).toBe(false)
    expect(api.removeInterest).toHaveBeenCalledWith('topic:ai')
  })

  it('a TRANSIENT follow failure keeps the optimistic flip and QUEUES it (#2004 #7)', async () => {
    // Previously toggle() swallowed the error and left ids unchanged — a silent dead button offline.
    // Now the flip persists and the write queues to replay, exactly like favourites.
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'addInterest').mockRejectedValue(new Error('offline')) // transient (not ApiError)
    const enq = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const s = useInterestsStore()
    await s.ensureLoaded()
    await s.toggle('topic:ai')
    expect(s.has('topic:ai')).toBe(true)
    expect(enq).toHaveBeenCalledWith({ op: 'interest.add', token: 'topic:ai' })
  })

  it('a TRANSIENT unfollow failure keeps the optimistic flip and QUEUES it', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['topic:ai'])
    vi.spyOn(api, 'removeInterest').mockRejectedValue(new Error('offline'))
    const enq = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const s = useInterestsStore()
    await s.ensureLoaded()
    await s.toggle('topic:ai')
    expect(s.has('topic:ai')).toBe(false)
    expect(enq).toHaveBeenCalledWith({ op: 'interest.remove', token: 'topic:ai' })
  })

  it('a REFUSAL (4xx) reverts the optimistic flip and does NOT queue', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'addInterest').mockRejectedValue(new ApiError(400, 'bad request'))
    const enq = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const s = useInterestsStore()
    await s.ensureLoaded()
    await s.toggle('topic:ai')
    expect(s.has('topic:ai')).toBe(false) // reverted — a refusal is an answer
    expect(enq).not.toHaveBeenCalled()
  })
})
