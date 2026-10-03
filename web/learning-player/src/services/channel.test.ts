import { afterEach, describe, expect, it, vi } from 'vitest'
import { Capacitor } from '@capacitor/core'
import { CHANNELS, resolveChannel } from './channel'

/**
 * #2265 — the distribution channel.
 *
 * The value exists to make one saved Umami view possible: `channel in (testflight, play_internal)`,
 * which separates the beta cohort from the operator's own usage and from web traffic. The failure
 * these tests guard against is not "no value" but "a confidently wrong value", because a wrong
 * channel silently puts the wrong sessions in or out of the cohort.
 */

function native(is: boolean) {
  vi.spyOn(Capacitor, 'isNativePlatform').mockReturnValue(is)
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('resolveChannel', () => {
  it('uses a stamped value when the release lane set one', () => {
    native(true)
    for (const stamped of ['testflight', 'play_internal', 'app_store', 'play_store'] as const) {
      vi.stubGlobal('__APP_CHANNEL__', stamped)
      expect(resolveChannel()).toBe(stamped)
    }
  })

  it('trims whitespace from a stamped value', () => {
    native(true)
    vi.stubGlobal('__APP_CHANNEL__', '  testflight\n')
    expect(resolveChannel()).toBe('testflight')
  })

  it('reports unknown on a native build when no lane stamped a channel', () => {
    // This is the state of every native build until #2189 / #2191 / #2192 wire their lanes, and the
    // state of a build the operator side-loads onto their own phone, forever.
    native(true)
    vi.stubGlobal('__APP_CHANNEL__', '')
    expect(resolveChannel()).toBe('unknown')
  })

  it('rejects an unrecognised stamped value rather than inventing a category', () => {
    // A typo in a release lane (`APP_CHANNEL=testfilght`) must not become a new channel in the
    // dashboards, where it would look like a real cohort and be excluded from the beta filter.
    native(true)
    vi.stubGlobal('__APP_CHANNEL__', 'testfilght')
    expect(resolveChannel()).toBe('unknown')
  })

  it('never claims web for a native build', () => {
    // The specific wrong answer worth a test of its own: `web` on a native build would make the
    // beta filter return nothing while appearing to work.
    native(true)
    for (const stamped of ['', 'nonsense', '   ']) {
      vi.stubGlobal('__APP_CHANNEL__', stamped)
      expect(resolveChannel()).not.toBe('web')
    }
  })

  it('never claims a beta channel for a build that did not declare one', () => {
    // The opposite wrong answer: guessing `testflight` would fold the operator's own device into
    // the cohort it is supposed to measure.
    native(true)
    vi.stubGlobal('__APP_CHANNEL__', '')
    expect(['testflight', 'play_internal']).not.toContain(resolveChannel())
  })

  it('resolves the web app without a stamp', () => {
    native(false)
    vi.stubGlobal('__APP_CHANNEL__', '')
    // `import.meta.env.DEV` is true under vitest, so this is the dev rung.
    expect(resolveChannel()).toBe('dev')
  })

  it('lists exactly the spec six plus unknown', () => {
    expect([...CHANNELS].sort()).toEqual(
      [
        'app_store',
        'dev',
        'play_internal',
        'play_store',
        'testflight',
        'unknown',
        'web',
      ].sort(),
    )
  })
})
