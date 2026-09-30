import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * The phone header is symmetric (operator 2026-09-30: "fully symmetric").
 *
 * Measured on the visible INK — each icon's drawn shapes, stroke included — not on the button boxes.
 * The boxes lied: the queue and bell buttons are padded around a 20px glyph while the avatar is a
 * solid circle, so equal box gaps read as a 20px gap then a 10px one. Fixed by narrowing the two
 * icon buttons on phones and giving the avatar a matching left margin; this pins the result:
 *
 * - queue→bell and bell→avatar gaps are EQUAL (within half a pixel);
 * - the logo and the avatar sit the same distance from their screen edges (the 16px page gutter);
 * - "Close Listening" clears the first icon even at 320px.
 */
for (const width of [320, 360, 390]) {
  test(`the phone header is symmetric at ${width}px`, async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'mobile-chrome', 'phone header')
    await page.setViewportSize({ width, height: 700 })
    await signInIsolated(page, `header-symmetry-${width}`, testInfo)
    await page.goto('/library')
    await expect(page.getByTestId('header-profile')).toBeVisible()

    const m = await page.evaluate(() => {
      const ink = (sel: string): [number, number] => {
        const svg = document.querySelector(`${sel} svg`) as SVGSVGElement
        const scale = svg.getBoundingClientRect().width / svg.viewBox.baseVal.width
        const half = (parseFloat(getComputedStyle(svg).strokeWidth) || 2) * scale * 0.5
        let l = Infinity
        let r = -Infinity
        svg.querySelectorAll('path, circle, rect, line, polyline').forEach((s) => {
          const b = s.getBoundingClientRect()
          l = Math.min(l, b.left - half)
          r = Math.max(r, b.right + half)
        })
        return [l, r]
      }
      const queue = ink('[data-testid="masthead-queue"]')
      const bell = ink('[data-testid="notifications-bell"]')
      const avatar = document.querySelector('[data-testid="header-profile"]')!.getBoundingClientRect()
      const logo = document.querySelector('header a svg')!.getBoundingClientRect()
      const titles = document.querySelectorAll('header > div > a:first-child span')
      const range = document.createRange()
      range.selectNodeContents(titles[titles.length - 1])
      return {
        queueToBell: bell[0] - queue[1],
        bellToAvatar: avatar.left - bell[1],
        logoFromLeft: logo.left,
        avatarFromRight: window.innerWidth - avatar.right,
        wordmarkToQueue: queue[0] - range.getBoundingClientRect().right,
      }
    })

    expect(
      Math.abs(m.queueToBell - m.bellToAvatar),
      `icon gaps differ: queue→bell ${m.queueToBell.toFixed(1)}px, bell→avatar ${m.bellToAvatar.toFixed(1)}px`,
    ).toBeLessThanOrEqual(0.5)
    expect(
      Math.abs(m.logoFromLeft - m.avatarFromRight),
      `edges differ: logo ${m.logoFromLeft}px from the left, avatar ${m.avatarFromRight}px from the right`,
    ).toBeLessThanOrEqual(0.5)
    expect(m.wordmarkToQueue, 'the wordmark runs into the first icon').toBeGreaterThanOrEqual(8)
  })
}
