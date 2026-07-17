/**
 * graph-v3 tier 8-3 stack-test — search reveals hidden.
 *
 * When the graph is in top-down mode and a Digest / Search / Explore hit
 * targets a node that lives INSIDE a collapsed super-theme, the viewer
 * must auto-expand that super-theme so the hit surfaces (instead of
 * silently dropping the focus). The mechanism is
 * ``maybeExpandTopDownForPendingFocus`` in GraphCanvas — this spec
 * drives the same code path through ``nav.requestFocusNode`` (the store
 * entry point Digest / Search wire into) so we exercise the actual
 * interaction, not a component-level mock.
 *
 * Gates (skip cleanly on mismatched corpora):
 *   1. topic_theme_clusters.json exists at v1.1.0 with at least one
 *      cluster whose ``super_theme_id`` is populated.
 *   2. There is at least one Topic in the KG whose ``themeClusterId``
 *      matches a cluster with a super_theme_id → the target we can
 *      reveal.
 */
import { expect, test } from '@playwright/test'

const STACK_TEST_CORPUS_PATH = '/app/output'

type ThemeCluster = {
  graph_compound_parent_id?: string
  super_theme_id?: string
  super_theme_label?: string
}
type ThemeDoc = { clusters?: ThemeCluster[] }

test.describe('Stack — graph-v3 tier 8-3 search reveals hidden', () => {
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

  test('a focus request targeting a collapsed super-theme child auto-expands + selects the node', async ({
    page,
    request,
  }) => {
    // Gate on enricher v1.1.0 shape.
    const themeRes = await request.get(
      `/api/artifacts/${encodeURIComponent(
        'enrichments/topic_theme_clusters.json',
      )}?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    test.skip(
      themeRes.status() !== 200,
      'topic_theme_clusters.json missing — skipping tier 8-3 spec',
    )
    const themeBody = (await themeRes.json()) as { data?: ThemeDoc } | ThemeDoc
    const themeDoc: ThemeDoc =
      'data' in themeBody && themeBody.data ? themeBody.data : (themeBody as ThemeDoc)
    const clusters = themeDoc.clusters ?? []
    const clustersWithSuper = clusters.filter(
      (c) => typeof c.super_theme_id === 'string' && c.super_theme_id.trim().length > 0,
    )
    test.skip(
      clustersWithSuper.length === 0,
      'no cluster carries super_theme_id — enricher pre-v1.1.0, skipping tier 8-3 spec',
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
    await page.waitForFunction(
      () =>
        (window as unknown as { __GIKG_CY_DEV__?: unknown }).__GIKG_CY_DEV__ !==
        undefined,
      undefined,
      { timeout: 20_000 },
    )

    // Find a Topic node in the FULL artifact whose themeClusterId is one
    // of the clusters with a super_theme_id — that's our reveal target.
    const target = await page.evaluate(() => {
      const dev = (
        window as unknown as {
          __GIKG_CY_DEV__?: {
            nodes: (sel: string) => {
              length: number
              first: () => {
                id: () => string
                data: (k: string) => unknown
              }
            }
          }
        }
      ).__GIKG_CY_DEV__
      if (!dev) return null
      /* Any Topic with a tagged themeClusterId works — tier T propagation
       * populates the same field on Insights / Persons too, but Topics
       * are the anchor. */
      const nodes = dev.nodes('[type = "Topic"][themeClusterId]')
      if (nodes.length === 0) return null
      const first = nodes.first()
      return { id: first.id(), themeClusterId: first.data('themeClusterId') as string }
    })
    test.skip(
      target === null,
      'no Topic with a themeClusterId in the loaded artifact — skipping tier 8-3 spec',
    )

    // Flip to top-down BEFORE requesting focus so the mount is the
    // synthetic slice and our target is guaranteed hidden.
    const chip = page.getByTestId('graph-load-mode-chip')
    await expect(chip).toBeVisible()
    const startLabel = await chip.textContent()
    if (startLabel && !/Top-down/.test(startLabel)) {
      await chip.click()
      await expect(chip).toContainText(/Top-down/)
    }

    // The target Topic must NOT be present on the canvas yet
    // (super-theme collapsed → child projection not injected).
    const preRevealPresent = await page.evaluate((tid) => {
      const dev = (
        window as unknown as {
          __GIKG_CY_DEV__?: {
            $id: (id: string) => { length: number }
          }
        }
      ).__GIKG_CY_DEV__
      return dev ? dev.$id(tid).length > 0 : false
    }, target!.id)
    expect(preRevealPresent, 'target Topic should be hidden pre-reveal').toBe(false)

    // Drive nav.requestFocusNode via the store — same entry point Digest
    // / Search use. maybeExpandTopDownForPendingFocus should fire on
    // the next tryApplyPendingFocus pass.
    await page.evaluate((tid) => {
      const win = window as unknown as {
        __PINIA__?: {
          state: { value: Record<string, unknown> }
          _s?: Map<string, { requestFocusNode?: (id: string) => void }>
        }
      }
      /* Use the exported Pinia store map (safer than reaching into the
       * app instance). Test harness stack-tests already assume DEV
       * mode where these hooks are exposed. */
      const stores = win.__PINIA__?._s
      const nav = stores?.get('graphNavigation') as
        | { requestFocusNode?: (id: string) => void }
        | undefined
      if (!nav || typeof nav.requestFocusNode !== 'function') {
        throw new Error('graphNavigation store or requestFocusNode missing')
      }
      nav.requestFocusNode(tid)
    }, target!.id)

    // Wait for the reveal → the target must land on the canvas + be
    // selected.
    await page.waitForFunction(
      (tid) => {
        const dev = (
          window as unknown as {
            __GIKG_CY_DEV__?: {
              $id: (id: string) => { length: number; selected: () => boolean }
            }
          }
        ).__GIKG_CY_DEV__
        if (!dev) return false
        const n = dev.$id(tid)
        return n.length > 0 && n.selected()
      },
      target!.id,
      { timeout: 20_000 },
    )

    expect(
      consoleErrors.filter((e) => !/DevTools|Extension/.test(e)),
      'no console errors during search-reveals-hidden path',
    ).toEqual([])
  })
})
