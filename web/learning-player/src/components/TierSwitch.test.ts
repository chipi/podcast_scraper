import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

/**
 * The header's backend badge (operator 2026-09-24).
 *
 * The rule it exists to serve: **PROD means the app is talking to prod; DEV means it is talking to
 * a local dev server.** It was derived from `getTier()` alone, which is only what the SWITCH is set
 * to — so every simulator and e2e build, which bakes `VITE_API_BASE_URL`, displayed PROD while
 * every request went to `127.0.0.1`. A badge whose whole job is "which backend am I looking at"
 * was answering the opposite.
 *
 * `tier.ts` reads `import.meta.env` and a `__DEV_API_BASE__` define at module scope, so each case
 * re-imports it under a fresh mock rather than trying to mutate the loaded module.
 */
async function mountWith(opts: { native: boolean; baked?: string; storedTier?: string }) {
  vi.resetModules()
  vi.doMock('@capacitor/core', () => ({
    Capacitor: { isNativePlatform: () => opts.native, getPlatform: () => (opts.native ? 'ios' : 'web') },
  }))
  vi.stubEnv('VITE_API_BASE_URL', opts.baked ?? '')
  // The switch only renders in an internal NATIVE build.
  vi.stubGlobal('__MOBILE_INTERNAL__', opts.native)
  vi.stubGlobal('__DEV_API_BASE__', 'https://dev.example/api/app')
  localStorage.setItem('lp_tier', opts.storedTier ?? 'prod')

  const TierSwitch = (await import('./TierSwitch.vue')).default
  return mount(TierSwitch)
}

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.doUnmock('@capacitor/core')
  localStorage.clear()
})

describe('TierSwitch label', () => {
  it('says DEV when a baked base points somewhere other than the live API', async () => {
    // The simulator/e2e case, and the bug: stored tier is 'prod', but the traffic is local.
    const w = await mountWith({ native: true, baked: 'http://127.0.0.1:4174/api/app' })
    expect(w.find('[data-testid="tier-switch"]').text()).toBe('DEV')
  })

  it('says PROD only when the target really is the live API', async () => {
    const w = await mountWith({ native: true })
    expect(w.find('[data-testid="tier-switch"]').text()).toBe('PROD')
  })

  it('says DEV when the switch itself is set to dev', async () => {
    const w = await mountWith({ native: true, storedTier: 'dev' })
    expect(w.find('[data-testid="tier-switch"]').text()).toBe('DEV')
  })

  it('names the resolved base in the tooltip, so the badge can be checked not just trusted', async () => {
    const w = await mountWith({ native: true, baked: 'http://127.0.0.1:4174/api/app' })
    expect(w.find('[data-testid="tier-switch"]').attributes('title')).toContain('127.0.0.1:4174')
  })
})
