import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  analyticsEnabled,
  EVENT_NAMES,
  SOURCES,
  toCountBucket,
  toDurationBucket,
  toRankBucket,
  track,
  userOptedOut,
} from './analytics'

/**
 * #2264 — the typed event registry.
 *
 * The registry is the contract for the whole analytics epic (#2263): the funnel, the dashboards
 * and every metric are defined in terms of these names and props. These tests hold three
 * properties the compiler cannot, or that would be easy to lose in a refactor:
 *
 *   1. no call site anywhere in `src/` tracks a name that is not in the registry (the guard the
 *      issue asks for by name);
 *   2. `track()` is genuinely inert when analytics is off or the user opted out;
 *   3. a throw inside the tracker never reaches the caller.
 */

const UMAMI_ENV = {
  VITE_UMAMI_SRC: 'https://analytics.example.test/script.js',
  VITE_UMAMI_WEBSITE_ID: 'test-website-id',
  VITE_ANALYTICS_OFF: '',
}

function enableAnalytics() {
  for (const [k, v] of Object.entries(UMAMI_ENV)) vi.stubEnv(k, v)
}

afterEach(() => {
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
  try {
    localStorage.clear()
  } catch {
    /* storage may be unavailable; nothing to clear */
  }
  delete (window as unknown as { umami?: unknown }).umami
})

// ── 1. The guard the issue asks for ──────────────────────────────────────────

describe('registry guard', () => {
  it('every tracked name in the codebase is in EVENT_NAMES', () => {
    // A `track('typo')` is already a type error, but `vue-tsc` runs in the build and type gates,
    // not in this suite — the spec asks specifically for a UNIT test that fails on an
    // unregistered name, so that a call site cannot land green here and red later.
    //
    // This is a TEXT scan, so it also matches `track('x')` written inside a comment. That is left
    // deliberately: the failure is loud and trivially fixed, and the effect is that documentation
    // examples have to name real events. Do not narrow the regex to dodge a comment — fix the
    // comment, or the event name it cites is wrong anyway.
    const sources = import.meta.glob('../**/*.{ts,vue}', {
      query: '?raw',
      import: 'default',
      eager: true,
    }) as Record<string, string>

    const registered = new Set<string>(EVENT_NAMES)
    const offenders: string[] = []

    for (const [path, src] of Object.entries(sources)) {
      // Skip this file and the registry itself: both legitimately contain names as data.
      if (path.includes('analytics.test.ts') || path.endsWith('services/analytics.ts')) continue
      // `track('name'` / `track("name"` — the only shape a literal call can take.
      for (const m of src.matchAll(/\btrack\(\s*['"]([^'"]+)['"]/g)) {
        const name = m[1] as string
        if (!registered.has(name)) offenders.push(`${path}: track('${name}')`)
      }
    }

    expect(
      offenders,
      'these call sites track a name that is not in EVENT_NAMES — add it to the registry with its ' +
        'props, or fix the name. Do not loosen this check.',
    ).toEqual([])
  })

  it('holds all 39 catalog events, with no duplicates', () => {
    // 39 rather than 40: `show_missing` is excluded because it needs an affordance that does not
    // exist. If this number changes, the catalog changed — update #2267, do not just bump it.
    expect(EVENT_NAMES).toHaveLength(39)
    expect(new Set(EVENT_NAMES).size).toBe(EVENT_NAMES.length)
  })

  it('does not contain show_missing, which has no affordance to fire from', () => {
    expect(EVENT_NAMES as readonly string[]).not.toContain('show_missing')
    // Its signal is carried here instead.
    expect(EVENT_NAMES as readonly string[]).toContain('empty_state_shown')
  })

  it('holds all 20 source values, with no duplicates', () => {
    expect(SOURCES).toHaveLength(20)
    expect(new Set(SOURCES).size).toBe(SOURCES.length)
  })
})

// ── 2. Enablement and the opt-out ────────────────────────────────────────────

describe('enablement', () => {
  it('is disabled when no script url or website id resolves', () => {
    vi.stubEnv('VITE_UMAMI_SRC', '')
    vi.stubEnv('VITE_UMAMI_SRC_DEV', '')
    vi.stubEnv('VITE_UMAMI_WEBSITE_ID', '')
    vi.stubEnv('VITE_ANALYTICS_OFF', '1')
    expect(analyticsEnabled()).toBe(false)
  })

  it('is disabled when the user opted out, even with env present', () => {
    enableAnalytics()
    localStorage.setItem('umami.disabled', '1')
    expect(userOptedOut()).toBe(true)
    expect(analyticsEnabled()).toBe(false)
  })

  it('is enabled with env present and no opt-out', () => {
    enableAnalytics()
    expect(analyticsEnabled()).toBe(true)
  })
})

describe('track', () => {
  let calls: Array<[string, unknown]>

  beforeEach(() => {
    calls = []
    ;(window as unknown as { umami?: unknown }).umami = {
      track: (name: string, props?: unknown) => calls.push([name, props]),
    }
  })

  it('forwards the name and props when enabled', () => {
    enableAnalytics()
    track('entity_open', { kind: 'topic', presentation: 'card', source: 'home_momentum' })
    expect(calls).toEqual([
      ['entity_open', { kind: 'topic', presentation: 'card', source: 'home_momentum' }],
    ])
  })

  it('forwards a no-prop event with no props', () => {
    enableAnalytics()
    track('landing_view')
    expect(calls).toEqual([['landing_view', undefined]])
  })

  it('is a no-op when the user opted out', () => {
    enableAnalytics()
    localStorage.setItem('umami.disabled', '1')
    track('landing_view')
    expect(calls).toEqual([])
  })

  it('is a no-op when analytics is off', () => {
    vi.stubEnv('VITE_UMAMI_SRC', '')
    vi.stubEnv('VITE_UMAMI_SRC_DEV', '')
    vi.stubEnv('VITE_UMAMI_WEBSITE_ID', '')
    vi.stubEnv('VITE_ANALYTICS_OFF', '1')
    track('landing_view')
    expect(calls).toEqual([])
  })

  it('swallows a throw from the tracker — telemetry never breaks the app', () => {
    enableAnalytics()
    ;(window as unknown as { umami?: unknown }).umami = {
      track: () => {
        throw new Error('umami exploded')
      },
    }
    expect(() => track('transcript_seek')).not.toThrow()
  })

  it('does not throw when the umami global is absent entirely', () => {
    enableAnalytics()
    delete (window as unknown as { umami?: unknown }).umami
    expect(() => track('offline_session')).not.toThrow()
  })
})

// ── 3. Buckets ───────────────────────────────────────────────────────────────

describe('buckets', () => {
  it('buckets counts at the documented boundaries', () => {
    expect(toCountBucket(0)).toBe('0')
    expect(toCountBucket(1)).toBe('1')
    expect(toCountBucket(2)).toBe('2-5')
    expect(toCountBucket(5)).toBe('2-5')
    expect(toCountBucket(6)).toBe('6-20')
    expect(toCountBucket(20)).toBe('6-20')
    expect(toCountBucket(21)).toBe('21+')
    expect(toCountBucket(9999)).toBe('21+')
  })

  it('buckets ranks at the documented boundaries', () => {
    expect(toRankBucket(1)).toBe('1')
    expect(toRankBucket(2)).toBe('2-3')
    expect(toRankBucket(3)).toBe('2-3')
    expect(toRankBucket(4)).toBe('4-10')
    expect(toRankBucket(10)).toBe('4-10')
    expect(toRankBucket(11)).toBe('11+')
  })

  it('buckets durations in SECONDS at the documented boundaries', () => {
    expect(toDurationBucket(0)).toBe('<1m')
    expect(toDurationBucket(59)).toBe('<1m')
    expect(toDurationBucket(60)).toBe('1-5m')
    expect(toDurationBucket(299)).toBe('1-5m')
    expect(toDurationBucket(300)).toBe('5-15m')
    expect(toDurationBucket(899)).toBe('5-15m')
    expect(toDurationBucket(900)).toBe('15-45m')
    expect(toDurationBucket(2699)).toBe('15-45m')
    expect(toDurationBucket(2700)).toBe('45m+')
  })

  it('never throws on hostile numbers — a bucketer must not break a render', () => {
    for (const n of [NaN, Infinity, -Infinity, -1]) {
      expect(() => toCountBucket(n)).not.toThrow()
      expect(() => toRankBucket(n)).not.toThrow()
      expect(() => toDurationBucket(n)).not.toThrow()
    }
    expect(toCountBucket(NaN)).toBe('0')
    expect(toCountBucket(-5)).toBe('0')
    expect(toRankBucket(NaN)).toBe('1')
    expect(toDurationBucket(NaN)).toBe('<1m')
  })
})
