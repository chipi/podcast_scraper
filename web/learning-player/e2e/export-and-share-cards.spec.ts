import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/** A fresh account per RUN: these tests follow, save and delete, and an account reused from the
 *  last run starts in the state that run left (already Following) — a flip test then flips it off. */
const RUN = Date.now().toString(36)

/**
 * The export viewer and the share cards (operator 2026-10-05, UXS-011 §Brief export, UXS-014
 * §Share, UXS-017).
 *
 * - The Brief has ONE "Download brief" link; it opens the notes in the in-app viewer, whose
 *   top right carries Markdown and Print or share. (Library highlights: capture.spec.ts.)
 * - "Share card" shares the SERVER's card — `/og/{kind}/{id}.png` for every kind, and a highlight's
 *   quote card from `/api/app/highlights/{id}/card.png` — never one the app draws itself.
 */
const EPISODE = 'Index Investing Without the Myths'
const PNG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

/** PNG width × height from the IHDR chunk. */
function pngSize(buf: Buffer): [number, number] {
  return [buf.readUInt32BE(16), buf.readUInt32BE(20)]
}

async function openEpisode(page: Page): Promise<void> {
  await page.goto('/podcast/p05')
  await page.getByText(EPISODE).first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await expect(page.getByRole('heading', { name: new RegExp(EPISODE) }).first()).toBeVisible()
}

test.beforeEach(async ({ page }, testInfo) => {
  await signInIsolated(page, `export-share-${testInfo.title.slice(0, 18)}-${RUN}`, testInfo)
})

test('Brief: ONE "Download brief" link opens the viewer with Markdown and Print or share', async ({
  page,
}) => {
  await openEpisode(page)
  await page.getByTestId('player-open-insights').click()
  const panel = page.getByTestId('knowledge-panel')
  const open = panel.getByTestId('episode-notes-export')
  await expect(open).toHaveText('Download brief')
  // No second chip per format — the old Markdown + PDF pair is gone.
  await expect(panel.getByTestId('episode-notes-pdf')).toHaveCount(0)
  await expect(panel.getByRole('button', { name: /^PDF$/ })).toHaveCount(0)

  await open.click()
  const viewer = page.getByTestId('export-viewer')
  // Visible ABOVE the panel — on mobile the panel is a modal in the top layer, and a viewer
  // teleported anywhere else rendered behind it ("nothing happens when I click PDF").
  await expect(viewer).toBeVisible()
  await expect(viewer.getByTestId('export-viewer-close')).toBeInViewport()
  const frame = page.frameLocator('[data-testid="export-viewer-frame"]')
  await expect(frame.getByText(EPISODE).first()).toBeVisible()
  await expect(viewer.getByTestId('export-viewer-share')).toHaveText('Print or share')

  const md = viewer.getByTestId('export-viewer-md')
  const href = await md.getAttribute('href')
  expect(href).toMatch(/\/api\/app\/episodes\/[^/]+\/notes\.md$/)
  const body = await (await page.request.get(new URL(href!, page.url()).toString())).text()
  expect(body).toContain(EPISODE)

  await viewer.getByTestId('export-viewer-close').click()
  await expect(viewer).toHaveCount(0)
  await expect(panel).toBeVisible()
})

test.describe('Share card — the server card, through the real button', () => {
  // A desktop browser has no file share, so the card arrives as a DOWNLOAD — exactly the image a
  // phone hands its share sheet. Mobile Chrome may take the file-share path instead, so this runs
  // where the outcome is a file the test can read.
  test.skip(({ isMobile }) => isMobile, 'desktop: the card downloads; a phone shares it')

  const PAGES: Array<[string, string, string]> = [
    ['show', '/podcast/p05', '/og/show/p05.png'],
    ['topic', '/topic/topic:risk-management', '/og/topic/topic%3Arisk-management.png'],
    ['person', '/person/person:nora', '/og/person/person%3Anora.png'],
    ['storyline', '/storyline/topic:risk-management', '/og/storyline/topic%3Arisk-management.png'],
    ['theme', '/theme/tc:broadcast-format', '/og/theme/tc%3Abroadcast-format.png'],
  ]

  for (const [kind, path, card] of PAGES) {
    test(`${kind}: Share card downloads the server's ${kind} card`, async ({ page }) => {
      await page.goto(path)
      await shareAndCheck(page, card)
    })
  }

  test('episode: Share card downloads the server\'s episode card', async ({ page }) => {
    await openEpisode(page)
    const slug = new URL(page.url()).pathname.split('/').pop()!
    await shareAndCheck(page, `/og/episode/${slug}.png`)
  })

  test("a highlight shares its quote card; the card is its owner's alone", async ({ page, browser }) => {
    const eps = await (await page.request.get('/api/app/podcasts/p05/episodes')).json()
    const slug = (eps as { items: { slug: string; title: string }[] }).items.find((e) =>
      e.title.startsWith('Index Investing'),
    )!.slug
    const created = await page.request.post('/api/app/highlights', {
      data: { episode_slug: slug, kind: 'span', start_ms: 65_000, quote_text: 'a line worth sharing' },
    })
    const hid = ((await created.json()) as { id: string }).id
    await page.goto('/library?tab=saved')
    const req = page.waitForRequest((r) => r.url().includes(`/api/app/highlights/${hid}/card.png`))
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button', { name: 'Share as card' }).first().click(),
    ])
    await req
    const buf = await readDownload(download)
    expect([...buf.subarray(0, 8)]).toEqual(PNG)
    expect(pngSize(buf)).toEqual([1080, 1440])

    // Signed out, the same URL is refused — a highlight is private, never an unfurl.
    const anon = await browser.newContext()
    try {
      const r = await anon.request.get(new URL(`/api/app/highlights/${hid}/card.png`, page.url()).toString())
      expect(r.status()).toBe(401)
    } finally {
      await anon.close()
    }
  })
})

async function readDownload(download: import('@playwright/test').Download): Promise<Buffer> {
  const stream = await download.createReadStream()
  const chunks: Buffer[] = []
  for await (const c of stream) chunks.push(c as Buffer)
  return Buffer.concat(chunks)
}

/** Open the page's Share menu, take "Share card", and check the PNG is the server's card. */
async function shareAndCheck(page: Page, cardPath: string): Promise<void> {
  const trigger = page.getByTestId('share-menu').first()
  await expect(trigger).toBeVisible()
  await trigger.click()
  const req = page.waitForRequest((r) => new URL(r.url()).pathname === cardPath)
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByTestId('share-card').first().click(),
  ])
  await req // the image came from the SERVER card route, not a canvas in the page
  const buf = await readDownload(download)
  expect([...buf.subarray(0, 8)], 'not a PNG').toEqual(PNG)
  expect(pngSize(buf)).toEqual([1080, 1440])
}
