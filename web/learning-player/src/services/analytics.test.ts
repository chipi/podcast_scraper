import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  analyticsEnabled,
  EVENT_NAMES,
  identify,
  installUmami,
  resetIdentity,
  resolveSession,
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


/**
 * Intercept `document.head.appendChild` and return what would have been appended.
 *
 * The tests must NOT actually append a `<script src=…>`: happy-dom then tries to fetch it and
 * raises an UNHANDLED async `DOMException [NotSupportedError]: JavaScript file loading is
 * disabled`, which fails every test in the file rather than the one that caused it. Intercepting
 * also makes the assertion sharper — the element we construct is what is under test, not the
 * browser's willingness to load it.
 */
function captureAppends(): HTMLElement[] {
  const appended: HTMLElement[] = []
  vi.spyOn(document.head, 'appendChild').mockImplementation(<T extends Node>(node: T): T => {
    appended.push(node as unknown as HTMLElement)
    return node
  })
  return appended
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
  // Injected tags are global state. Leaving one behind makes `installUmami` correctly decline to
  // add another, and the NEXT test then asserts against the previous test's tag — which is exactly
  // how this suite first went red.
  document.querySelectorAll('script[data-umami-installed]').forEach((el) => el.remove())
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

// ── 4. Identity (#2265) ──────────────────────────────────────────────────────

describe('identify', () => {
  let identified: Array<[string | Record<string, unknown>, unknown]>

  beforeEach(() => {
    identified = []
    ;(window as unknown as { umami?: unknown }).umami = {
      track: () => {},
      identify: (id: string | Record<string, unknown>, props?: unknown) =>
        identified.push([id, props]),
    }
  })

  it('attaches the id and the session properties', () => {
    enableAnalytics()
    identify('a1b2c3', { platform: 'ios', app_version: '1.2.3', channel: 'testflight' })
    expect(identified).toEqual([
      ['a1b2c3', { platform: 'ios', app_version: '1.2.3', channel: 'testflight' }],
    ])
  })

  it('ignores an empty id rather than identifying with a blank', () => {
    // An account whose backfill has not run yet has no analytics_id. Sending '' would file every
    // such account into one bucket that looks like a single very busy participant.
    enableAnalytics()
    identify('', { platform: 'web', app_version: '1.0.0', channel: 'web' })
    expect(identified).toEqual([])
  })

  it('does nothing when the user opted out', () => {
    enableAnalytics()
    localStorage.setItem('umami.disabled', '1')
    identify('a1', { platform: 'web', app_version: '1.0.0', channel: 'web' })
    expect(identified).toEqual([])
  })

  it('never throws, even if the tracker explodes', () => {
    enableAnalytics()
    ;(window as unknown as { umami?: unknown }).umami = {
      identify: () => {
        throw new Error('boom')
      },
    }
    expect(() =>
      identify('a1', { platform: 'web', app_version: '1.0.0', channel: 'web' }),
    ).not.toThrow()
  })
})

describe('resetIdentity', () => {
  it('removes the umami global so the previous id cannot survive sign-out', () => {
    // The tracker cannot be asked to forget: its `identify` assigns only when the derived id is
    // defined (`void 0 !== a && (V = a)`), so `identify({})` leaves the previous distinct id in
    // place. Replacing the global is the only reset, and sign-out in this app does not reload.
    enableAnalytics()
    captureAppends()
    ;(window as unknown as { umami?: unknown }).umami = { track: () => {}, identify: () => {} }
    resetIdentity()
    expect((window as unknown as { umami?: unknown }).umami).toBeUndefined()
  })

  it('removes the old tag and installs a fresh one, so anonymous traffic is still tracked', () => {
    // The spec wants logged-out landing views recorded as an ANONYMOUS session, not dropped.
    enableAnalytics()
    const stale = document.createElement('script')
    stale.setAttribute('data-umami-installed', '1')
    document.head.appendChild(stale)

    const appended = captureAppends()
    resetIdentity()

    expect(stale.parentNode, 'the stale tag must be removed').toBeNull()
    expect(appended, 'exactly one fresh tag').toHaveLength(1)
    expect(appended[0]?.getAttribute('data-exclude-search')).toBe('true')
  })

  it('never throws when there is nothing to reset', () => {
    captureAppends()
    expect(() => resetIdentity()).not.toThrow()
  })
})

describe('installUmami', () => {
  it('builds one tag with the query string excluded and the right website id', () => {
    enableAnalytics()
    const appended = captureAppends()
    installUmami()
    expect(appended).toHaveLength(1)
    const tag = appended[0]!
    expect(tag.getAttribute('data-exclude-search')).toBe('true')
    expect(tag.getAttribute('data-website-id')).toBe('test-website-id')
    expect(tag.getAttribute('src')).toBe(UMAMI_ENV.VITE_UMAMI_SRC)
    expect((tag as HTMLScriptElement).defer).toBe(true)
  })

  it('is idempotent — a second call does not double-count page views', () => {
    enableAnalytics()
    const present = document.createElement('script')
    present.setAttribute('data-umami-installed', '1')
    document.head.appendChild(present)

    const appended = captureAppends()
    installUmami()
    installUmami()
    expect(appended, 'a tag is already installed; none should be added').toHaveLength(0)
  })

  it('injects nothing when no website id resolves', () => {
    vi.stubEnv('VITE_UMAMI_SRC', '')
    vi.stubEnv('VITE_UMAMI_SRC_DEV', '')
    vi.stubEnv('VITE_UMAMI_WEBSITE_ID', '')
    vi.stubEnv('VITE_ANALYTICS_OFF', '1')
    const appended = captureAppends()
    installUmami()
    expect(appended).toHaveLength(0)
  })
})

describe('resolveSession', () => {
  it('reports a known platform and a real app version', () => {
    const s = resolveSession()
    expect(['ios', 'android', 'web']).toContain(s.platform)
    expect(typeof s.app_version).toBe('string')
    expect(typeof s.channel).toBe('string')
  })

  it('does not carry a cohort, locale or device model', () => {
    // The spec is explicit: cohorts are computed at read time from the operator's analytics_id
    // list, and anything Umami already derives must not be duplicated onto the session.
    expect(Object.keys(resolveSession()).sort()).toEqual(['app_version', 'channel', 'platform'])
  })
})

// ── 5. Completeness (#2267) ──────────────────────────────────────────────────

describe('every registered event is actually wired', () => {
  it('has a call site in src/ for all 39 names', () => {
    // The registry test at the top proves no call site invents a name. This proves the converse,
    // which is the failure that hides: an event sitting in the registry with nothing emitting it
    // looks exactly like a feature nobody used. A dashboard built on it reports zero, honestly and
    // misleadingly, and there is no error anywhere to notice.
    const sources = import.meta.glob('../**/*.{ts,vue}', {
      query: '?raw',
      import: 'default',
      eager: true,
    }) as Record<string, string>

    const wired = new Set<string>()
    for (const [path, src] of Object.entries(sources)) {
      if (path.endsWith('services/analytics.ts') || path.includes('.test.')) continue
      if (path.includes('__checks__')) continue
      // `track('x'` / `track("x"` and the ternary form `track(cond ? 'a' : 'b'`.
      for (const m of src.matchAll(/track\(\s*(?:[^,()]*\?\s*)?['"]([a-z_]+)['"]/g)) {
        wired.add(m[1] as string)
      }
      for (const m of src.matchAll(/\?\s*['"]([a-z_]+)['"]\s*:\s*['"]([a-z_]+)['"]/g)) {
        wired.add(m[1] as string)
        wired.add(m[2] as string)
      }
    }

    const unwired = EVENT_NAMES.filter((n) => !wired.has(n))
    expect(
      unwired,
      'these events are in the registry but nothing emits them — wire them, or remove them from ' +
        'the registry with the reason. An event that cannot fire makes a report say "zero" when ' +
        'the honest answer is "never measured".',
    ).toEqual([])
  })
})
