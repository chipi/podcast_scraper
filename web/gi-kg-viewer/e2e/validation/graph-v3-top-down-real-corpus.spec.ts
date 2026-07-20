/**
 * Tier-3 real-corpus validation for graph-v3 tier 8 (top-down mount +
 * expand-on-tap) and the tier 7-1a enricher schema that feeds it.
 *
 * Sibling to ``real-corpus.spec.ts`` (V1–V6) — same helpers, same run
 * command:
 *
 *   make serve                         # in another terminal
 *   make ci-ui-validation \
 *     CORPUS="$(pwd)/tests/fixtures/viewer-validation-corpus/$(cat tests/fixtures/FIXTURES_VERSION)"
 *
 * IMPORTANT: pass the **version dir** (`v3/`), NOT `v3/corpus/`. Raw
 * `feeds/<feed>/metadata/*.gi.json` bundles that `serve-api` reads live
 * under the version subdir; pointing the walk at ``v3/corpus/`` returns
 * an empty artifact list. See TESTING.md.
 *
 * The synthetic viewer-validation-corpus/v3 is a committed, deterministic
 * 36-episode / 9-show fixture. After running ``make enrich CORPUS=…
 * PROFILE=airgapped`` it carries a real ``topic_theme_clusters.json`` at
 * enricher v1.1.0 with real super_theme_id fields → the specs run
 * against the actual synthetic corpus, not a mock and not a
 * freshly-built stack corpus.
 */
import { expect, test } from '@playwright/test'
import {
  SHELL_HEADING_RE,
  dismissGraphGestureOverlayIfPresent,
  mainViewsNav,
  signInIsolated,
  statusBarCorpusPathInput,
} from '../helpers'

const CORPUS_PATH = process.env.CORPUS_PATH ?? ''
if (!CORPUS_PATH) {
  throw new Error(
    'Tier-3 graph-v3 validation requires CORPUS_PATH. Set ' +
      'CORPUS_PATH=tests/fixtures/viewer-validation-corpus/v3/corpus or run ' +
      'via `make ci-ui-validation CORPUS=…`.',
  )
}

const IGNORE_CONSOLE = [
  /^Failed to load resource: the server responded with a status of 404 \(Not Found\)$/,
]

test.describe('Tier-3 graph-v3 tier 8 — top-down mount + expand-on-tap (real corpus)', () => {
  test('V-G1 — top-down opt-in swaps mount to SuperTheme bubbles + tap expands children', async ({
    page,
    request,
  }, testInfo) => {
    // Sanity: the corpus must carry an enricher v1.1.0 theme_clusters
    // artifact. Fail LOUDLY here (not skip) — the operator's fixture is
    // committed and enriched deterministically, so a miss means the
    // corpus wasn't re-enriched before the walk.
    const themeRes = await request.get(
      `/api/corpus/theme-clusters?path=${encodeURIComponent(CORPUS_PATH)}`,
    )
    expect(
      themeRes.status(),
      `topic_theme_clusters.json must exist under ${CORPUS_PATH}. ` +
        'Run `make enrich CORPUS=tests/fixtures/viewer-validation-corpus/v3 PROFILE=airgapped CORPUS_ONLY=1` first.',
    ).toBe(200)
    type ThemeCluster = { super_theme_id?: string; super_theme_label?: string }
    type ThemeDoc = { clusters?: ThemeCluster[]; super_theme_count?: number }
    const themeBody = (await themeRes.json()) as { data?: ThemeDoc } | ThemeDoc
    const themeDoc: ThemeDoc =
      'data' in themeBody && themeBody.data ? themeBody.data : (themeBody as ThemeDoc)
    const clusters = themeDoc.clusters ?? []
    const supersInDoc = new Set(
      clusters
        .map((c) => c.super_theme_id ?? '')
        .filter((s) => s.length > 0),
    )
    expect(
      supersInDoc.size,
      `topic_theme_clusters must carry super_theme_id (enricher v1.1.0+). ` +
        'Re-run the enricher on this corpus.',
    ).toBeGreaterThan(0)

    // === Boot + sign in, THEN attach the console-error harness ===
    // ``signInIsolated`` blocks until the callback resolves + the
    // shell heading is up, so any console error captured AFTER this
    // returns is definitely inside the authenticated app — not a
    // pre-auth artefact.
    await signInIsolated(page, 'graph-v3-top-down', testInfo)

    const errors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() !== 'error') return
      const text = msg.text()
      if (IGNORE_CONSOLE.some((r) => r.test(text))) return
      errors.push(text)
    })
    page.on('pageerror', (err) => errors.push(err.message))

    const corpusInput = statusBarCorpusPathInput(page)
    await corpusInput.waitFor({ state: 'visible', timeout: 15_000 })
    await corpusInput.fill(CORPUS_PATH)
    await corpusInput.press('Enter').catch(() => {})
    await page.waitForTimeout(2_000)

    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible({ timeout: 15_000 })
    await expect(page.locator('.graph-canvas')).toBeVisible({ timeout: 60_000 })
    await page.waitForFunction(
      () =>
        (window as unknown as { __GIKG_CY_DEV__?: unknown }).__GIKG_CY_DEV__ !==
        undefined,
      undefined,
      { timeout: 30_000 },
    )
    await dismissGraphGestureOverlayIfPresent(page)

    // === Flip load-mode → topDown ===
    const chip = page.getByTestId('graph-load-mode-chip')
    await expect(chip).toBeVisible()
    const startLabel = await chip.textContent()
    if (startLabel && !/Everything/.test(startLabel)) {
      await chip.click()
      await expect(chip).toContainText(/Everything/)
    }
    await chip.click()
    await expect(chip).toContainText(/Top-down/)

    // === Assert the mount contract ===
    const mountSnapshot = await page.waitForFunction(
      (expectedMin) => {
        const dev = (
          window as unknown as {
            __GIKG_CY_DEV__?: {
              nodes: (sel: string) => {
                length: number
                map: (fn: (n: unknown) => unknown) => unknown[]
              }
            }
          }
        ).__GIKG_CY_DEV__
        if (!dev) return false
        const supers = dev.nodes('[type = "SuperTheme"]')
        if (supers.length < expectedMin) return false
        return {
          count: supers.length,
          ids: supers.map((n) => (n as { id: () => string }).id()),
        }
      },
      supersInDoc.size,
      { timeout: 20_000 },
    )
    const snapshot = (await mountSnapshot.jsonValue()) as {
      count: number
      ids: string[]
    }
    expect(snapshot.count).toBeGreaterThanOrEqual(supersInDoc.size)
    expect(snapshot.count).toBeLessThanOrEqual(8) // viewer clamp (gap-4)

    // === Tap → expand-on-tap injects projected children ===
    const targetSid = snapshot.ids[0]!
    await page.evaluate((sid) => {
      const dev = (
        window as unknown as {
          __GIKG_CY_DEV__?: {
            $id: (id: string) => { emit: (event: string) => void }
          }
        }
      ).__GIKG_CY_DEV__
      dev?.$id(sid).emit('tap')
    }, targetSid)
    await page.waitForFunction(
      (sid) => {
        const dev = (
          window as unknown as {
            __GIKG_CY_DEV__?: {
              nodes: (sel: string) => {
                length: number
                map: (fn: (n: unknown) => unknown) => unknown[]
              }
            }
          }
        ).__GIKG_CY_DEV__
        return dev ? dev.nodes(`[parent = "${sid}"]`).length > 0 : false
      },
      targetSid,
      { timeout: 20_000 },
    )

    // === Collapse → children disappear ===
    await page.evaluate((sid) => {
      const dev = (
        window as unknown as {
          __GIKG_CY_DEV__?: {
            $id: (id: string) => { emit: (event: string) => void }
          }
        }
      ).__GIKG_CY_DEV__
      dev?.$id(sid).emit('tap')
    }, targetSid)
    await page.waitForFunction(
      (sid) => {
        const dev = (
          window as unknown as {
            __GIKG_CY_DEV__?: { nodes: (sel: string) => { length: number } }
          }
        ).__GIKG_CY_DEV__
        return dev ? dev.nodes(`[parent = "${sid}"]`).length === 0 : false
      },
      targetSid,
      { timeout: 20_000 },
    )

    expect(errors, 'no console errors during graph-v3 top-down walk').toEqual([])
  })
})
