import type { Page } from '@playwright/test'

/**
 * Requests a test fails on purpose (an aborted /api/health for offline mode, a 404 for a
 * failure-mode row). Chromium logs each failed request as a console error ("Failed to load
 * resource: …"); Firefox does not. That line is the browser reporting the test's own fault, so
 * console-error checks skip it — for the declared URLs only. Any other error, including a failed
 * request to an undeclared URL, still fails the test.
 */
const faults = new WeakMap<Page, RegExp[]>()

export function declareInjectedFault(page: Page, url: RegExp): void {
  faults.set(page, [...(faults.get(page) ?? []), url])
}

export function isInjectedFaultLog(page: Page, text: string, url: string): boolean {
  return (
    text.startsWith('Failed to load resource') &&
    (faults.get(page) ?? []).some((pattern) => pattern.test(url))
  )
}
