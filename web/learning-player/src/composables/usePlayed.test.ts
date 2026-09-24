import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { usePlayed } from './usePlayed'
import {
  __resetPositions,
  hydratePositions,
  localPosition,
  recordPosition,
} from '../services/playbackPositions'
import { useCompletedStore } from '../stores/completed'

vi.mock('../services/api', () => ({
  getCompleted: vi.fn(async () => []),
  markCompleted: vi.fn(async (slug: string) => [slug]),
  unmarkCompleted: vi.fn(async () => []),
}))

/**
 * "Played" has two sources and the app only ever read one (operator 2026-09-23).
 *
 * The join is the SERVER's — `/api/app/completed` merges the hand-marked list with the finish
 * record — so these do not re-test the merge. What they pin is the reason this composable exists
 * at all: the window where the server cannot know yet, because the finish happened offline.
 */
describe('usePlayed', () => {
  beforeEach(async () => {
    setActivePinia(createPinia())
    __resetPositions()
    await hydratePositions('u1')
  })

  it('counts an episode the server calls completed', () => {
    const completed = useCompletedStore()
    completed.slugs = ['ep-marked']
    expect(usePlayed().isPlayed('ep-marked')).toBe(true)
  })

  it('counts an episode finished OFFLINE, before the server has heard about it', () => {
    // The flight case. `/completed` is empty because the PUT never left the device; the device's
    // own record is the only thing that knows, and it is right.
    recordPosition('ep-flight', 1800, true, false)
    expect(useCompletedStore().slugs).toEqual([])
    expect(usePlayed().isPlayed('ep-flight')).toBe(true)
  })

  it('does NOT count an episode merely started offline', () => {
    recordPosition('ep-part', 90, false, false)
    expect(usePlayed().isPlayed('ep-part')).toBe(false)
  })

  it('does not count an episode it has never seen', () => {
    expect(usePlayed().isPlayed('ep-unknown')).toBe(false)
  })

  it('un-playing clears the DEVICE finish flag, so the toggle can come back', async () => {
    // Without this the toggle is one-way: the server drops its half, the local half keeps
    // answering true, and the row stays marked however many times you tap it.
    recordPosition('ep-flight', 1800, true, false)
    const played = usePlayed()
    expect(played.isPlayed('ep-flight')).toBe(true)

    await played.togglePlayed('ep-flight')

    expect(played.isPlayed('ep-flight')).toBe(false)
    expect(localPosition('ep-flight')?.finished).toBe(false)
  })

  it('un-playing keeps the resume point', async () => {
    // "I did not finish it" is not "I was never here" -- throwing the position away would send the
    // listener back to zero on an episode they are 30 minutes into.
    recordPosition('ep-flight', 1800, true, false)
    await usePlayed().togglePlayed('ep-flight')
    expect(localPosition('ep-flight')?.seconds).toBe(1800)
  })

  it('re-marks a position it had un-played, without inventing a finish', async () => {
    recordPosition('ep-flight', 1800, true, false)
    const played = usePlayed()
    await played.togglePlayed('ep-flight')
    await played.togglePlayed('ep-flight')

    // Back on via the completed set -- the marker the user just asked for -- and NOT by rewriting
    // the local finish, which would claim the audio reached the end when it did not.
    expect(played.isPlayed('ep-flight')).toBe(true)
    expect(useCompletedStore().slugs).toContain('ep-flight')
    expect(localPosition('ep-flight')?.finished).toBe(false)
  })
})
