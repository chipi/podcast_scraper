#!/usr/bin/env node
/**
 * Screenshots for the release notes (operator 2026-10-15): one phone-sized PNG per change, taken
 * from the running app against the synthetic fixture corpus, so no real episode or person appears.
 *
 *   BASE=http://127.0.0.1:5181 [SET=1.0.4] node scripts/release-shots.mjs <out-dir> [shot-name ...]
 *
 * BASE is a `vite` dev server whose /api and /audio proxy to the fixture API and the mock audio
 * host (VITE_API_TARGET / VITE_MEDIA_TARGET). Every shot runs on its own; a failure is reported and
 * the rest still run, so one moved control never costs the whole set.
 */
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { chromium } from '@playwright/test'

const BASE = process.env.BASE ?? 'http://127.0.0.1:5181'
const OUT = process.argv[2]
const ONLY = new Set(process.argv.slice(3))
if (!OUT) {
  console.error('usage: node scripts/release-shots.mjs <out-dir> [shot ...]')
  process.exit(2)
}

// A fictional voice: some fixture shows borrow real people's names, and a release note must not
// put invented quotes in a real person's mouth.
const EPISODE = 'p02-8fdd7b7f79'
const SPANISH = 'p10-d9d02f5622'
const REEL = '/listen/topic/topic%3Asystems-thinking?minutes=15'
const STORYLINE = '/storyline/thc%3Amanaging-risk'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/** Element crops: the fixed bottom navigation would sit over the crop's lower edge. */
async function hideChrome(page) {
  await page.addStyleTag({ content: '[data-testid="bottom-nav"]{display:none!important}' })
}

async function settle(page, ms = 900) {
  await page.waitForLoadState('networkidle').catch(() => {})
  await sleep(ms)
}

/** Seed one account the way a settled listener would leave it: a name, three interests, two
 * episodes part-heard. */
async function seed(page) {
  await page.request.post('/api/app/profile/name', { data: { name: 'Alex' } })
  await page.request.put('/api/app/interests', {
    data: { items: ['tc:show-themes', 'tc:lifelong-learning', 'thc:managing-risk'] },
  })
  const tz = -new Date().getTimezoneOffset()
  for (const [slug, pos] of [
    ['p07-a5eb82076a', [20, 140]],
    ['p09-db2d39ef2e', [15, 95]],
  ]) {
    for (const position_seconds of pos) {
      await page.request.put(`/api/app/playback/${slug}`, { data: { position_seconds, tz_offset_minutes: tz } })
    }
  }
}

const SHOTS = {
  async 'short-listen'(page) {
    await page.goto(REEL)
    await page.getByTestId('topic-reel-current').waitFor()
    await settle(page, 2500)
    await page.getByTestId('topic-reel-up-next').nth(2).click()
    await settle(page, 2500)
    await page.getByTestId('topic-reel-pause').click().catch(() => {})
    await page.evaluate(() => window.scrollTo(0, 0))
    await settle(page)
    return page.screenshot()
  },
  async 'subject-top'(page) {
    await page.goto(STORYLINE)
    await page.getByTestId('learn-door').waitFor()
    await settle(page)
    return page.screenshot()
  },
  async 'learn-week'(page) {
    await page.goto(STORYLINE)
    await page.getByTestId('learn-door').click()
    await page.getByTestId('learn-door-yes').click({ timeout: 3000 }).catch(() => {})
    await settle(page)
    await page.goto('/learn')
    await page.getByTestId('learn-view').waitFor()
    await settle(page, 1500)
    return page.screenshot()
  },
  async home(page) {
    await page.goto('/')
    await page.getByTestId('home-in-progress').waitFor({ timeout: 15000 })
    await settle(page, 1500)
    return page.screenshot()
  },
  async 'home-in-progress'(page) {
    await page.goto('/')
    await hideChrome(page)
    const el = page.getByTestId('home-in-progress')
    await el.waitFor({ timeout: 15000 })
    await el.scrollIntoViewIfNeeded()
    // The row scrolls sideways; start it at its first card.
    await el.evaluate((root) => root.querySelectorAll('*').forEach((n) => { n.scrollLeft = 0 }))
    await settle(page, 1500)
    return el.screenshot()
  },
  async discover(page) {
    await page.goto('/browse')
    await page.getByTestId('discover-scope').waitFor()
    await settle(page, 2000)
    return page.screenshot()
  },
  async explore(page) {
    await page.goto('/explore')
    await page.getByTestId('explore-view').waitFor()
    await settle(page, 1500)
    return page.screenshot()
  },
  async player(page) {
    await page.goto(`/episode/${EPISODE}`)
    await page.getByTestId('player-obi').waitFor()
    await settle(page, 1500)
    return page.screenshot()
  },
  async moments(page) {
    await page.goto(`/episode/${EPISODE}`)
    await page.getByTestId('player-open-moments').click()
    await page.getByTestId('moments-card').waitFor()
    await settle(page, 3000)
    return page.screenshot()
  },
  async brief(page) {
    await page.goto(`/episode/${EPISODE}`)
    await page.getByTestId('player-open-insights').click()
    await page.getByTestId('knowledge-panel').waitFor()
    await settle(page, 1500)
    return page.screenshot()
  },
  async 'brief-search'(page) {
    await page.goto(`/episode/${EPISODE}`)
    await page.getByTestId('player-open-insights').click()
    const panel = page.getByTestId('knowledge-panel')
    await panel.waitFor()
    const box = panel.getByRole('searchbox').or(panel.locator('input[type="search"]')).first()
    await box.fill('reliability')
    await box.press('Enter')
    await settle(page, 2000)
    await panel.getByText(/Search results/).first().evaluate((el) => el.scrollIntoView({ block: 'start' }))
    await settle(page, 600)
    return page.screenshot()
  },
  async search(page) {
    await page.goto('/search?q=risk&scope=all')
    await page.getByTestId('search-episode-count').waitFor({ timeout: 15000 })
    await settle(page, 1500)
    return page.screenshot()
  },
  async 'search-episode'(page) {
    await page.goto('/search?q=systems%20thinking&scope=all')
    await page.getByTestId('search-episode-count').waitFor({ timeout: 15000 })
    await hideChrome(page)
    const card = page.getByTestId('episode-group').first()
    await card.scrollIntoViewIfNeeded()
    await settle(page, 1500)
    return card.screenshot()
  },
  async collections(page) {
    const made = await page.request.post('/api/app/collections', { data: { name: 'Risk, from every angle' } })
    const board = await made.json()
    for (const [kind, ref] of [
      ['episode', 'p07-a5eb82076a'],
      ['episode', 'p09-db2d39ef2e'],
      ['episode', 'p02-8fdd7b7f79'],
      ['episode', 'p03-5b8ca67393'],
      ['storyline', 'thc:managing-risk'],
    ]) {
      await page.request.post(`/api/app/collections/${board.id}/items`, { data: { kind, ref } })
    }
    await page.goto('/library')
    await page.getByTestId('library-tab-collections').click()
    await page.getByTestId('collection-open').first().waitFor()
    await settle(page, 1500)
    return page.screenshot()
  },
  async 'board-brief'(page) {
    await page.goto('/library')
    await page.getByTestId('library-tab-collections').click()
    await page.getByTestId('collection-open').first().click()
    await page.getByTestId('collection-brief').click()
    await page.getByTestId('board-brief').waitFor()
    await settle(page, 2500)
    return page.screenshot()
  },
  async 'theme-members'(page) {
    await page.goto('/theme/tc%3Arisk-management')
    const el = page.getByTestId('theme-topics')
    await el.waitFor()
    await hideChrome(page)
    await el.scrollIntoViewIfNeeded()
    await settle(page)
    return el.screenshot()
  },
  async 'show-licence'(page) {
    await page.goto('/podcast/p05')
    await page.getByTestId('podcast-licence-link').waitFor({ timeout: 15000 })
    await settle(page, 1200)
    return page.screenshot()
  },
  async translated(page) {
    await page.goto(`/episode/${SPANISH}`)
    await page.getByTestId('player-open-insights').click()
    const label = page.getByTestId('translated-from').first()
    await label.waitFor({ timeout: 15000 })
    await label.scrollIntoViewIfNeeded()
    await settle(page, 1200)
    return page.screenshot()
  },
  async stats(page) {
    await page.goto('/profile')
    await page.getByRole('tab', { name: /stats/i }).click()
    const card = page.getByTestId('stats-listening')
    await card.waitFor()
    await card.scrollIntoViewIfNeeded()
    await settle(page, 2000)
    return card.screenshot()
  },
}

const PHONE = {
  baseURL: BASE,
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 3,
  isMobile: true,
  hasTouch: true,
  colorScheme: 'dark',
}
const browser = await chromium.launch()

/** A page signed in as a fresh account; `seeded` makes it a settled listener first. */
async function account(seeded = true) {
  const context = await browser.newContext(PHONE)
  const p = await context.newPage()
  await p.goto(`/api/app/auth/login?as=${encodeURIComponent(`release-notes-${Date.now().toString(36)}`)}`)
  await p.waitForURL((u) => !u.pathname.startsWith('/api/'), { timeout: 15000 }).catch(() => {})
  if (seeded) await seed(p)
  else await p.request.post('/api/app/profile/name', { data: { name: 'Alex' } })
  return p
}

/*
 * 1.0.3 (SET=1.0.3), shot against that version's own app, server and fixture data (a worktree at
 * 722d11907): the screens its notes describe, with the test ids that version carried.
 */
const SHOTS_103 = {
  async 'guided-start'() {
    const p = await account(false)
    await p.goto('/')
    await p.getByTestId('guided-skip').or(p.getByTestId('guided-explain')).first().waitFor({ timeout: 15000 })
    await settle(p, 1500)
    return p.screenshot()
  },
  async notes(page) {
    await page.goto(`/episode/${EPISODE}`)
    await page.getByTestId('player-open-insights').click()
    await page.getByTestId('knowledge-panel').waitFor()
    await settle(page, 1500)
    return page.screenshot()
  },
  async 'save-to-board'(page) {
    await page.request.post('/api/app/collections', { data: { name: 'Risk, from every angle' } })
    await page.request.post('/api/app/collections', { data: { name: 'Systems' } })
    await page.goto(`/episode/${EPISODE}`)
    await page.getByTestId('add-to-collection').first().click()
    await page.getByTestId('add-to-collection-menu').waitFor()
    await settle(page, 1200)
    return page.screenshot()
  },
  async trends(page) {
    await page.goto('/browse')
    await settle(page, 2500)
    return page.screenshot()
  },
  async login() {
    const context = await browser.newContext(PHONE)
    const p = await context.newPage()
    await p.goto('/login')
    await p.getByTestId('login-view').waitFor()
    await settle(p, 1200)
    return p.screenshot()
  },
}

/**
 * A plain screen for the older releases, whose test ids differ: go to `path`, optionally tap the
 * first control whose name matches `tap`, and photograph the screen (signed out with `signedOut`).
 */
function screen(path, { tap, signedOut = false, wait = 2000 } = {}) {
  return async (page) => {
    const p = signedOut ? await (await browser.newContext(PHONE)).newPage() : page
    await p.goto(path)
    await settle(p, wait)
    if (tap) {
      await p.getByRole('button', { name: tap }).or(p.getByRole('tab', { name: tap })).or(p.getByText(tap, { exact: true })).first().click()
      await settle(p, wait)
    }
    return p.screenshot()
  }
}

// 1.0.1 (SET=1.0.1) at 7b8a8b4d6, and the first beta, 1.0.0 (SET=1.0.0) at 78fbeb6ad.
const SHOTS_101 = {
  interests: screen('/profile', { tap: /^Interests$/ }),
  'delete-account': screen('/account/delete'),
  storyline: screen('/storyline/thc%3Amanaging-risk'),
  'translated-transcript': screen(`/episode/${SPANISH}`, { tap: /transcript/i }),
}
const SHOTS_100 = {
  welcome: screen('/welcome', { signedOut: true }),
  home: screen('/'),
  player: screen(`/episode/${EPISODE}`, { wait: 3500 }),
  notes: screen(`/episode/${EPISODE}`, { tap: /episode notes/i }),
  library: screen('/library'),
  storyline: screen('/storyline/thc%3Amanaging-risk'),
}

const SETS = { '1.0.4': SHOTS, '1.0.3': SHOTS_103, '1.0.1': SHOTS_101, '1.0.0': SHOTS_100 }
const SET = process.env.SET ?? '1.0.4'
if (!SETS[SET]) {
  console.error(`no shot set ${SET}; have ${Object.keys(SETS).join(', ')}`)
  process.exit(2)
}
const page = await account(true)
await mkdir(OUT, { recursive: true })

let failed = 0
let ran = 0
for (const [name, shoot] of Object.entries(SETS[SET])) {
  if (ONLY.size && !ONLY.has(name)) continue
  ran += 1
  try {
    const png = await shoot(page)
    const file = join(OUT, `${name}.png`)
    const { writeFile } = await import('node:fs/promises')
    await writeFile(file, png)
    console.log(`ok    ${name}`)
  } catch (err) {
    failed += 1
    console.log(`FAIL  ${name}: ${String(err?.message ?? err).split('\n')[0]}`)
  }
}
await browser.close()
console.log(`${ran - failed} ok, ${failed} failed`)
process.exit(failed ? 1 : 0)
