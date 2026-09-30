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

describe('replaceAll — the picker PUTs an absolute set (iOS-F1)', () => {
  it('adopts the written set so every surface reading the store agrees', () => {
    const s = useInterestsStore()
    expect(s.ids).toEqual([])
    s.replaceAll(['tc:ai', 'thc:risk'])
    expect(s.ids).toEqual(['tc:ai', 'thc:risk'])
    expect(s.has('tc:ai')).toBe(true)
  })

  it('marks the store LOADED, so a later ensureLoaded() cannot clobber it with a stale fetch', async () => {
    // `ensureLoaded` short-circuits on `loaded`. If `replaceAll` left it false the next caller
    // would refetch and could overwrite a just-saved set with whatever the server had a moment ago.
    const spy = vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const s = useInterestsStore()
    s.replaceAll(['tc:ai'])
    await s.ensureLoaded()
    expect(spy).not.toHaveBeenCalled()
    expect(s.ids).toEqual(['tc:ai'])
  })

  it('a quick follow-then-unfollow ends unfollowed, on the screen AND in request order', async () => {
    // The flaky trending.spec toggle-and-back (2026-09-30): the add's response arrived after the
    // remove's and re-lit the button. The add is held back here to force that ordering.
    let releaseAdd!: () => void
    const order: string[] = []
    vi.spyOn(api, 'addInterest').mockImplementation(async () => {
      order.push('add:start')
      await new Promise<void>((r) => (releaseAdd = r))
      order.push('add:end')
      return ['topic:ai']
    })
    vi.spyOn(api, 'removeInterest').mockImplementation(async () => {
      order.push('remove:start')
      return []
    })
    const s = useInterestsStore()
    const first = s.toggle('topic:ai')
    expect(s.has('topic:ai')).toBe(true) // optimistic
    const second = s.toggle('topic:ai')
    expect(s.has('topic:ai')).toBe(false) // optimistic, second tap
    await Promise.resolve()
    releaseAdd()
    await Promise.all([first, second])
    expect(s.has('topic:ai'), 'the earlier add response re-lit a follow the user undid').toBe(false)
    // The server sees the user's order: the remove is not sent until the add has finished.
    expect(order).toEqual(['add:start', 'add:end', 'remove:start'])
  })
})
