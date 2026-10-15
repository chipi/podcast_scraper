#!/usr/bin/env node
/**
 * Render a release-notes page (public/releases/<version>/) to a PDF that can be sent to testers
 * directly, e.g. over WhatsApp (operator 2026-10-15), and/or to review screenshots.
 *
 *   BASE=http://127.0.0.1:5181 node scripts/release-pdf.mjs pdf    <page-path> <out.pdf>
 *   BASE=http://127.0.0.1:5181 node scripts/release-pdf.mjs shot   <page-path> <out.png> [width]
 *
 * <page-path> is served by any server that serves public/, e.g. `/releases/1.0.4/index.html` on the
 * vite dev server. Links in the PDF point at PUBLIC_BASE (default https://closelistening.app). The
 * PDF keeps the dark page (printBackground) at a phone-friendly width, one
 * long page, so it reads like the web page rather than a document cut into A4 sheets.
 */
import { chromium } from '@playwright/test'

const BASE = process.env.BASE ?? 'http://127.0.0.1:5181'
const [mode, path, out, widthArg] = process.argv.slice(2)
if (!mode || !path || !out) {
  console.error('usage: release-pdf.mjs pdf|shot <page-path> <out> [width]')
  process.exit(2)
}

const browser = await chromium.launch()
const width = Number(widthArg ?? (mode === 'pdf' ? 430 : 1280))
const page = await browser.newPage({ viewport: { width, height: 900 }, deviceScaleFactor: 2 })
await page.goto(BASE + path, { waitUntil: 'networkidle' })
// Lazy images load only near the viewport: scroll through once so every one is in.
await page.evaluate(async () => {
  for (const img of document.images) img.loading = 'eager'
  await Promise.all([...document.images].map((i) => (i.complete ? null : new Promise((r) => { i.onload = i.onerror = r }))))
})

if (mode === 'shot') {
  await page.screenshot({ path: out, fullPage: true })
} else {
  // One continuous page: a chat app shows a PDF as a scroll, and a page break through a
  // screenshot reads as a bug. The sticky masthead becomes static so it prints once.
  // Chromium's PDF output flattens large blurred shadows into solid blocks behind the phones, so
  // the PDF trades them for a hairline.
  await page.addStyleTag({
    content: '.mast{position:static!important}.phone,.crop,.release .peek img{box-shadow:0 0 0 1px #2a3449!important}',
  })
  // Links in a sent file must point at the live site, never at the local server that rendered it;
  // "Back to the app" means nothing outside the app, so it goes.
  const publicBase = process.env.PUBLIC_BASE ?? 'https://closelistening.app'
  await page.evaluate(({ publicBase, path }) => {
    document.querySelectorAll('.mast .back').forEach((el) => el.remove())
    const here = new URL(path, publicBase)
    // Same-page jumps (`#learn`) stay as they are, so they move within the PDF.
    document.querySelectorAll('a[href]:not([href^="#"])').forEach((a) => {
      a.href = new URL(a.getAttribute('href'), here).href.replace(/index\.html$/, '')
    })
  }, { publicBase, path })
  await page.emulateMedia({ media: 'screen' })
  const height = await page.evaluate(() => document.documentElement.scrollHeight)
  await page.pdf({ path: out, width: `${width}px`, height: `${height + 2}px`, printBackground: true, pageRanges: '1' })
}
await browser.close()
console.log(`wrote ${out}`)
