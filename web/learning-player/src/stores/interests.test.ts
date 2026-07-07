import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
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

  it('toggle() leaves ids unchanged when the follow write rejects (no optimistic drift)', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'addInterest').mockRejectedValue(new Error('signed out'))
    const s = useInterestsStore()
    await s.ensureLoaded()
    await s.toggle('topic:ai') // add path rejects → state must not flip
    expect(s.has('topic:ai')).toBe(false)
  })

  it('toggle() leaves ids unchanged when the unfollow write rejects', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['topic:ai'])
    vi.spyOn(api, 'removeInterest').mockRejectedValue(new Error('transient'))
    const s = useInterestsStore()
    await s.ensureLoaded()
    await s.toggle('topic:ai') // remove path rejects → token still followed
    expect(s.has('topic:ai')).toBe(true)
  })
})
