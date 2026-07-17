/**
 * graph-v3 tier 8 stack-test — top-down mount + expand-on-tap over the
 * seeded corpus.
 *
 * Gates:
 *   1. The corpus at ``/app/output`` must carry a
 *      ``enrichments/topic_theme_clusters.json`` with at least one
 *      cluster whose ``super_theme_id`` is set. If the enricher wasn't
 *      re-run at v1.1.0 or the corpus has no clusters, the test is
 *      SKIPPED — this is intended as a v1.1.0 acceptance gate, not a
 *      regression trap for older stack-test corpora.
 *   2. The viewer must expose ``__GIKG_CY_DEV__`` at runtime (DEV-gated
 *      Cytoscape handle used across other stack-tests to inspect
 *      canvas state without brittle DOM traversal).
 *
 * What we assert:
 *   1. Flipping load-mode to ``topDown`` swaps the canvas to a
 *      SuperTheme-only slice with N nodes where N ≤ 8 (viewer clamp)
 *      and ≥ the number of super-themes in the doc.
 *   2. Tapping a SuperTheme injects projected children (Topics /
 *      Insights / Persons) whose ``parent`` attr points at the tapped
 *      super-theme.
 *   3. Tapping again collapses back to the preview.
 *
 * Sibling to ``stack-graph-v3-search-reveals.spec.ts`` which covers
 * tier 8-3 (Digest / search hits into a collapsed super-theme).
 */
import { expect, test } from '@playwright/test'

const STACK_TEST_CORPUS_PATH = '/app/output'

type ThemeCluster = {
  graph_compound_parent_id?: string
  super_theme_id?: string
  super_theme_label?: string
  member_count?: number
}
type ThemeDoc = {
  clusters?: ThemeCluster[]
  super_theme_count?: number
}

test.describe('Stack — graph-v3 tier 8 top-down mount + expand', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript((corpus) => {
      try {
        window.localStorage.setItem('ps_corpus_path', corpus)
        window.localStorage.setItem('ps_graph_hints_seen', '1')
      } catch {
        /* ignore quota / private mode */
      }
    }, STACK_TEST_CORPUS_PATH)
  })

  test('top-down mode mounts super-theme bubbles + expand-on-tap injects children', async ({
    page,
    request,
  }) => {
    /* Gate on the enricher output shape — skip cleanly on corpora that
     * pre-date v1.1.0 so this spec is safe to add to the default stack. */
    const themeRes = await request.get(
      `/api/artifacts/${encodeURIComponent(
        'enrichments/topic_theme_clusters.json',
      )}?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    test.skip(
      themeRes.status() !== 200,
      'topic_theme_clusters.json missing on this stack corpus — skipping graph-v3 tier 8 spec',
    )
    const themeBody = (await themeRes.json()) as { data?: ThemeDoc } | ThemeDoc
    const themeDoc: ThemeDoc =
      'data' in themeBody && themeBody.data ? themeBody.data : (themeBody as ThemeDoc)
    const clusters = themeDoc.clusters ?? []
    const supersInDoc = new Set(
      clusters
        .map((c) => c.super_theme_id ?? '')
        .filter((s) => s.length > 0),
    )
    test.skip(
      supersInDoc.size === 0,
      'topic_theme_clusters carries no super_theme_id — enricher v1.0.x fixture, skipping graph-v3 tier 8 spec',
    )

    const consoleErrors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text())
    })
    page.on('pageerror', (err) => consoleErrors.push(err.message))

    await page.goto('/')
    await page
      .getByRole('navigation', { name: 'Main views' })
      .getByRole('button', { name: 'Graph' })
      .click()
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible({ timeout: 10_000 })
    await expect(page.locator('.graph-canvas')).toBeVisible({ timeout: 60_000 })
    // Give fcose a beat to settle before poking the store.
    await page.waitForFunction(
      () =>
        (window as unknown as { __GIKG_CY_DEV__?: { nodes: () => { length: number } } })
          .__GIKG_CY_DEV__ !== undefined,
      undefined,
      { timeout: 20_000 },
    )

    // === Flip load-mode → topDown and assert SuperTheme mount ===
    const chip = page.getByTestId('graph-load-mode-chip')
    await expect(chip).toBeVisible()
    // Ensure we START at 'everything' so the toggle lands us in topDown.
    // The chip text is `Load-mode: <mode> ▾`.
    const startLabel = await chip.textContent()
    if (startLabel && !/Everything/.test(startLabel)) {
      // Already in topDown → toggle out first so we can capture the
      // canonical enter-transition path.
      await chip.click()
      await expect(chip).toContainText(/Everything/)
    }
    await chip.click()
    await expect(chip).toContainText(/Top-down/)

    // After the mount swap, the canvas holds SuperTheme nodes only.
    // Wait for the redraw to settle then read the cy state via the dev hook.
    const superSnapshot = await page.waitForFunction(
      (expectedMin) => {
        const dev = (
          window as unknown as {
            __GIKG_CY_DEV__?: {
              nodes: (sel: string) => { length: number; map: (fn: (n: unknown) => unknown) => unknown[] }
            }
          }
        ).__GIKG_CY_DEV__
        if (!dev) return false
        const superNodes = dev.nodes('[type = "SuperTheme"]')
        if (superNodes.length < expectedMin) return false
        return {
          count: superNodes.length,
          ids: superNodes.map((n) => (n as { id: () => string }).id()),
        }
      },
      supersInDoc.size,
      { timeout: 15_000 },
    )
    const snapshot = (await superSnapshot.jsonValue()) as { count: number; ids: string[] }
    expect(snapshot.count).toBeGreaterThanOrEqual(supersInDoc.size)
    // Viewer clamp bound (gap-4).
    expect(snapshot.count).toBeLessThanOrEqual(8)

    // === Tap one super-theme → assert projected children appear ===
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

    const projected = await page.waitForFunction(
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
        if (!dev) return false
        const kids = dev.nodes(`[parent = "${sid}"]`)
        if (kids.length === 0) return false
        return {
          childCount: kids.length,
          childTypes: Array.from(
            new Set(kids.map((n) => (n as { data: (k: string) => string }).data('type'))),
          ),
        }
      },
      targetSid,
      { timeout: 15_000 },
    )
    const proj = (await projected.jsonValue()) as { childCount: number; childTypes: string[] }
    expect(proj.childCount).toBeGreaterThan(0)
    // Projected children must include at least one Topic or Insight
    // (the tier T propagation targets). Not a strict shape — different
    // corpora will emit different mixes.
    const hasMeaningfulKid = proj.childTypes.some((t) =>
      ['Topic', 'Insight', 'Person', 'Podcast', 'Entity_person', 'Entity_organization'].includes(t),
    )
    expect(
      hasMeaningfulKid,
      `projected children should include a tier-T-propagated type; got ${proj.childTypes.join(', ')}`,
    ).toBe(true)

    // === Tap again → collapse back to preview ===
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
      { timeout: 15_000 },
    )

    expect(
      consoleErrors.filter((e) => !/DevTools|Extension/.test(e)),
      'no console errors during top-down mount / expand / collapse',
    ).toEqual([])
  })
})
