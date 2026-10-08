// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAuthStore } from '../stores/auth'
import { useUserPreferencesStore } from '../stores/userPreferences'
import { GUIDED_SNOOZE_MS, GUIDED_SNOOZED_PREF, GUIDED_START_PREF, useGuidedStart } from './useGuidedStart'

describe('useGuidedStart (operator 2026-10-08)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) }))
    useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
  })

  it('"Not now" snoozes the guide, and it comes back after three days', () => {
    const g = useGuidedStart()
    const t0 = 1_800_000_000_000
    g.snooze(t0)
    expect(g.isSnoozed(t0 + 1000)).toBe(true)
    expect(g.isSnoozed(t0 + GUIDED_SNOOZE_MS - 1)).toBe(true)
    expect(g.isSnoozed(t0 + GUIDED_SNOOZE_MS)).toBe(false)
    expect(useUserPreferencesStore().get(GUIDED_SNOOZED_PREF)).toBe(t0)
  })

  it('a "dismissed: true" from before the snooze counts as long ago, not forever', () => {
    void useUserPreferencesStore().set(GUIDED_SNOOZED_PREF, true)
    localStorage.setItem('lp.interests.dismissed', '1')
    expect(useGuidedStart().isSnoozed()).toBe(false)
  })

  it('restart clears the snooze and marks the run to start from step 1', async () => {
    const g = useGuidedStart()
    g.snooze()
    await g.restart()
    expect(g.isSnoozed()).toBe(false)
    expect(g.state.value).toBe('restart')
    expect(useUserPreferencesStore().get(GUIDED_START_PREF)).toBe('restart')
    expect(localStorage.getItem('lp.interests.dismissed')).toBeNull()
  })
})
