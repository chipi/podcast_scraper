import { afterEach, describe, expect, it, vi } from 'vitest'

const native = vi.hoisted(() => ({ value: false }))
vi.mock('../services/tier', () => ({ isNativeShell: () => native.value }))

import { PUBLIC_ORIGIN, shareUrl } from './shareLink'

afterEach(() => {
  native.value = false
})

/** One https link for everything shared (operator 2026-10-05). */
describe('shareUrl', () => {
  it("on the web, links to the page's own origin — right on every tier", () => {
    expect(shareUrl('episode', 'p06-721')).toBe(`${window.location.origin}/episode/p06-721`)
  })

  it("in the app, links to the public site, never the WebView's private origin", () => {
    // capacitor://localhost (iOS) / https://localhost (Android) opened nothing for the recipient.
    native.value = true
    expect(shareUrl('person', 'person:grady-booch')).toBe(
      `${PUBLIC_ORIGIN}/person/person%3Agrady-booch`,
    )
  })

  it('names a moment with ?t= in whole seconds, and drops a bad one', () => {
    native.value = true
    expect(shareUrl('episode', 'x', 65.9)).toBe(`${PUBLIC_ORIGIN}/episode/x?t=65`)
    expect(shareUrl('episode', 'x', -1)).toBe(`${PUBLIC_ORIGIN}/episode/x`)
    expect(shareUrl('episode', 'x', null)).toBe(`${PUBLIC_ORIGIN}/episode/x`)
  })
})
