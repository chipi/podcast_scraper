import { expect, test } from '@playwright/test'
import { loadGraphViaFilePicker } from './helpers'

/**
 * Locks Cytoscape label placement: main-label bbox must not intersect the node disc in model space.
 * Complements DevTools manual checks (`window.__GIKG_CY_DEV__` in Vite dev).
 */
test.describe('Graph labels clear node disc', () => {
  test('label box does not overlap circular node body (offline fixture)', async ({ page }) => {
    await loadGraphViaFilePicker(page)

    const res = await page.evaluate(() => {
      const el = document.querySelector('.graph-canvas') as HTMLElement & {
        _cyreg?: { cy: import('cytoscape').Core }
      }
      const cy = el?._cyreg?.cy
      if (!cy) {
        return { ok: false as const, reason: 'no cytoscape instance' }
      }
      const pad = 0.75
      const overlaps: { id: string; d: number; r: number }[] = []
      const distPointToRect = (
        cx: number,
        cy0: number,
        x1: number,
        y1: number,
        x2: number,
        y2: number,
      ): number => {
        const px = Math.max(x1, Math.min(cx, x2))
        const py = Math.max(y1, Math.min(cy0, y2))
        return Math.hypot(cx - px, cy0 - py)
      }
      cy.nodes().forEach((n) => {
        const lab = n.data('label') as string | undefined
        if (!lab) return
        const body = n.boundingBox({ includeNodes: true, includeLabels: false, includeEdges: false })
        const label = n.boundingBox({ includeNodes: false, includeLabels: true, includeEdges: false })
        if (!label || label.w <= 0 || label.h <= 0) return
        const r = Math.min(body.w, body.h) / 2
        const cx = (body.x1 + body.x2) / 2
        const cy0 = (body.y1 + body.y2) / 2
        const d = distPointToRect(cx, cy0, label.x1, label.y1, label.x2, label.y2)
        if (d < r - pad) overlaps.push({ id: n.id(), d, r })
      })
      return { ok: true as const, overlapCount: overlaps.length, overlaps }
    })

    expect(res.ok, res.ok ? '' : (res as { reason: string }).reason).toBe(true)
    if (res.ok) {
      expect(res.overlapCount, JSON.stringify(res.overlaps)).toBe(0)
    }
  })
})
