// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { initAnalytics } from './analytics'

/**
 * DR drill 2026-09-07 regression guard.
 *
 * The Umami `<script>` was appended to `<head>` during app start-up. A head
 * script — even `defer` — keeps the window `load` event pending until its fetch
 * settles, so a client that cannot reach the analytics host stalls on "loading"
 * for the full network timeout. On the drill the Playwright browser ran from a
 * GitHub runner with no tailnet route to the homelab umami host; the request
 * never settled (`status=-1`) and `page.goto` blocked 92.5s of a 120s budget
 * while every other resource completed in under 2s. The graph canvas itself was
 * healthy — it mounted 3.7s after the assertion started, with no time left.
 *
 * The contract these tests lock: analytics NEVER gates page readiness.
 */

const INSTALLED = 'script[data-umami-installed]'

function setReadyState(value: DocumentReadyState): void {
  Object.defineProperty(document, 'readyState', { configurable: true, value })
}

function injectedScripts(): NodeListOf<Element> {
  return document.querySelectorAll(INSTALLED)
}

describe('initAnalytics', () => {
  beforeEach(() => {
    vi.stubEnv('VITE_UMAMI_SRC', 'https://analytics.example/script.js')
    vi.stubEnv('VITE_UMAMI_WEBSITE_ID', 'test-website-id')
    document.head.innerHTML = ''
    setReadyState('loading')
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    document.head.innerHTML = ''
    setReadyState('complete')
  })

  it('does NOT inject while the document is still loading', () => {
    initAnalytics()
    // The regression: injecting here is what held `load` open for ~90s.
    expect(injectedScripts()).toHaveLength(0)
  })

  it('injects once the window load event has fired', () => {
    initAnalytics()
    expect(injectedScripts()).toHaveLength(0)

    window.dispatchEvent(new Event('load'))

    const scripts = injectedScripts()
    expect(scripts).toHaveLength(1)
    const el = scripts[0] as HTMLScriptElement
    expect(el.src).toBe('https://analytics.example/script.js')
    expect(el.getAttribute('data-website-id')).toBe('test-website-id')
  })

  it('injects immediately when the document has already finished loading', () => {
    setReadyState('complete')
    initAnalytics()
    expect(injectedScripts()).toHaveLength(1)
  })

  it('stays idempotent across repeated calls before load', () => {
    initAnalytics()
    initAnalytics()
    window.dispatchEvent(new Event('load'))
    // Two queued listeners must still yield exactly one script tag.
    expect(injectedScripts()).toHaveLength(1)
  })

  it('injects nothing when analytics is not configured', () => {
    vi.stubEnv('VITE_UMAMI_SRC', '')
    vi.stubEnv('VITE_UMAMI_WEBSITE_ID', '')
    initAnalytics()
    window.dispatchEvent(new Event('load'))
    expect(injectedScripts()).toHaveLength(0)
  })
})
