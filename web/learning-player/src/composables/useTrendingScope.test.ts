import { beforeEach, describe, expect, it, vi } from 'vitest'

// Plain mutable holders (no reactivity needed): each test creates a fresh useTrendingScope() and
// reads the computed once, so it evaluates against the current holder values.
const state = vi.hoisted(() => ({ authed: false, pref: undefined as unknown }))
const setSpy = vi.hoisted(() => vi.fn())

vi.mock('../stores/auth', () => ({
  useAuthStore: () => ({
    get isAuthenticated() {
      return state.authed
    },
  }),
}))
vi.mock('../stores/userPreferences', () => ({
  useUserPreferencesStore: () => ({ get: () => state.pref, set: setSpy }),
}))

import { TRENDING_SCOPE_PREF, useTrendingScope } from './useTrendingScope'

describe('useTrendingScope (#2030)', () => {
  beforeEach(() => {
    state.authed = false
    state.pref = undefined
    setSpy.mockClear()
  })

  it('is corpus when signed out, even if a "mine" pref is stored', () => {
    state.authed = false
    state.pref = 'mine'
    expect(useTrendingScope().scope.value).toBe('corpus')
  })

  it('is MINE when signed in with no stored lens — on by default (operator 2026-10-07)', () => {
    state.authed = true
    state.pref = undefined
    expect(useTrendingScope().scope.value).toBe('mine')
  })

  it('is corpus once the listener switches it off, and that sticks', () => {
    state.authed = true
    state.pref = 'corpus'
    expect(useTrendingScope().scope.value).toBe('corpus')
  })

  it('is mine when signed in and the lens is set to mine', () => {
    state.authed = true
    state.pref = 'mine'
    expect(useTrendingScope().scope.value).toBe('mine')
  })

  it('setScope persists the lens under the app-level preference key', () => {
    state.authed = true
    useTrendingScope().setScope('mine')
    expect(setSpy).toHaveBeenCalledWith(TRENDING_SCOPE_PREF, 'mine')
  })

  it('setScope writes nothing when signed out — there is no "mine" and no per-user pref', () => {
    // The Search results' switch now writes the shared lens (2026-10-09), and a signed-out visitor
    // can still pick "All" there.
    state.authed = false
    useTrendingScope().setScope('corpus')
    expect(setSpy).not.toHaveBeenCalled()
  })
})
