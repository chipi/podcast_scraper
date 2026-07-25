import { describe, expect, it } from 'vitest'
import apiSrc from '../services/api.ts?raw'
import playerStoreSrc from '../stores/player.ts?raw'
import playerViewSrc from '../views/PlayerView.vue?raw'
import indexHtml from '../../index.html?raw'

/**
 * Guardrail (#1311) — codified mobile-readiness invariants that must not regress. These are static
 * source checks that run in the normal unit suite (so CI gates them), guarding the Capacitor-shell
 * prerequisites established in the mobile epic (#1298). If one fails, a change re-broke a mobile
 * invariant — fix the change, don't weaken the check.
 */

// All .vue components, loaded as raw source.
const components = import.meta.glob('../**/*.vue', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

describe('mobile invariants (guardrail #1311)', () => {
  it('API base resolves from VITE_API_BASE_URL (reaches the API in the capacitor:// shell)', () => {
    // A bare origin-relative base fails in the WebView (origin = capacitor://localhost).
    expect(apiSrc, 'services/api.ts must resolve the base from VITE_API_BASE_URL').toMatch(
      /VITE_API_BASE_URL/,
    )
  })

  it('index.html sets viewport-fit=cover (enables env(safe-area-inset-*))', () => {
    expect(indexHtml).toMatch(/viewport-fit=cover/)
  })

  it('no raw vh units in components — use dvh (iOS 100vh over-report clips sheets/footers)', () => {
    const offenders: string[] = []
    for (const [path, src] of Object.entries(components)) {
      const matches = src.match(/\[[^\]]*vh\]/g) ?? [] // Tailwind arbitrary values e.g. max-h-[85vh]
      for (const m of matches) {
        if (!/[dsl]vh\]/.test(m)) offenders.push(`${path}: ${m}`) // allow dvh / svh / lvh
      }
      if (/\bmin-h-screen\b/.test(src)) offenders.push(`${path}: min-h-screen (use min-h-dvh)`)
    }
    expect(offenders, `raw vh usage found:\n${offenders.join('\n')}`).toEqual([])
  })

  it('playback state lives in the player store, not PlayerView local refs', () => {
    // Re-adding local playback refs would break MediaSession / native controls (one source of truth).
    for (const ref of ['playing', 'currentTime', 'duration', 'rate']) {
      expect(
        playerViewSrc,
        `PlayerView must not declare a local '${ref}' ref — it belongs to stores/player.ts`,
      ).not.toMatch(new RegExp(`const\\s+${ref}\\s*=\\s*ref\\(`))
    }
    expect(playerViewSrc).toMatch(/usePlayerStore\(\)/)
  })

  it('MediaSession is wired in the player store (lock-screen / headphone controls)', () => {
    expect(playerStoreSrc).toMatch(/navigator\.mediaSession/)
    expect(playerStoreSrc).toMatch(/setActionHandler/)
  })
})
