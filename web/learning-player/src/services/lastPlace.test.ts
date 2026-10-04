import { describe, expect, it } from 'vitest'
import { outcomeOf, planRestore, RESTORE_WINDOW_MS, type LastPlace } from './lastPlace'

/** #2278 — the restore rules, as plain values. */
const NOW = 1_800_000_000_000
const episode = { slug: 'ep-1', url: 'https://x/a.mp3', title: 'Ep', position: 754 }
const saved = (over: Partial<LastPlace> = {}): LastPlace => ({
  path: '/person/person:aaron-levie',
  userId: 'u1',
  at: NOW - 10 * 60_000,
  episode,
  ...over,
})
const base = {
  now: NOW,
  userId: 'u1',
  landedOnHome: true,
  routeIsRestorable: true,
  playerEmpty: true,
}

describe('planRestore', () => {
  it('brings back the screen and the episode after a cold launch onto Home', () => {
    expect(planRestore({ ...base, saved: saved() })).toEqual({
      route: '/person/person:aaron-levie',
      episode,
    })
  })

  it('restores nothing for ANOTHER account — never open someone else’s last screen', () => {
    expect(planRestore({ ...base, saved: saved({ userId: 'u2' }) })).toEqual({ route: null, episode: null })
  })

  it('restores nothing signed out', () => {
    expect(planRestore({ ...base, userId: null, saved: saved() })).toEqual({ route: null, episode: null })
  })

  it('honours the 12h window at its edge, and not past it', () => {
    expect(planRestore({ ...base, saved: saved({ at: NOW - RESTORE_WINDOW_MS }) }).route).not.toBeNull()
    expect(planRestore({ ...base, saved: saved({ at: NOW - RESTORE_WINDOW_MS - 1 }) })).toEqual({
      route: null,
      episode: null,
    })
  })

  it('ignores a record from the future (clock change) rather than trusting it', () => {
    expect(planRestore({ ...base, saved: saved({ at: NOW + 60_000 }) })).toEqual({ route: null, episode: null })
  })

  it('leaves the route alone when the boot did not land on Home — a WebView reload keeps its URL', () => {
    expect(planRestore({ ...base, landedOnHome: false, saved: saved() })).toEqual({ route: null, episode })
  })

  it('does not restore a route the caller says is not restorable (public, Home, unknown)', () => {
    expect(planRestore({ ...base, routeIsRestorable: false, saved: saved() }).route).toBeNull()
  })

  it('never replaces an episode that is already loaded', () => {
    expect(planRestore({ ...base, playerEmpty: false, saved: saved() }).episode).toBeNull()
  })

  it('leaves the episode to its own page — PlayerView loads it and resumes from the server', () => {
    const plan = planRestore({ ...base, saved: saved({ path: '/episode/ep-1?t=12' }) })
    expect(plan).toEqual({ route: '/episode/ep-1?t=12', episode: null })
  })

  it('restores the episode on another episode’s page', () => {
    expect(planRestore({ ...base, saved: saved({ path: '/episode/ep-2' }) }).episode).toEqual(episode)
  })

  it('handles a record with no episode', () => {
    expect(planRestore({ ...base, saved: saved({ episode: null }) })).toEqual({
      route: '/person/person:aaron-levie',
      episode: null,
    })
  })
})

describe('outcomeOf', () => {
  it('names what came back, for app_launch.restored', () => {
    expect(outcomeOf({ route: '/a', episode })).toBe('both')
    expect(outcomeOf({ route: '/a', episode: null })).toBe('route')
    expect(outcomeOf({ route: null, episode })).toBe('episode')
    expect(outcomeOf({ route: null, episode: null })).toBe('none')
  })
})
