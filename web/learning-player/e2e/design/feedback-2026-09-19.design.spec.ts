import { expect, test, type Locator, type Page } from '@playwright/test'
import { expectSignedIn } from '../helpers'

/**
 * ONE focused shot per item of the operator's 2026-09-19 device round.
 *
 * Distinct from `surfaces.design.spec.ts`, which shoots whole surfaces for a composition review.
 * The question here is narrower — "did the thing I asked for actually change, and does it look
 * right" — so each shot is CROPPED to the element that changed. A full-page screenshot answers
 * that badly: the reader has to hunt for a 40px row inside a 9000px page, and a shot that happens
 * to miss the element looks the same as one where the element is fine.
 *
 * Same rule as the sibling spec: assert enough that a blank or still-loading capture FAILS rather
 * than quietly producing a PNG of a spinner. A screenshot nobody can trust is worse than none,
 * because it gets reviewed anyway.
 *
 *   npm run design:shots -- feedback-2026-09-19
 *
 * Output: `design-results/<variant>/<viewport>/fb-<item>.png`
 */
const VARIANT = process.env.DESIGN_VARIANT || 'baseline'
const dir = (name: string) => `design-results/${VARIANT}/${test.info().project.name}/fb-${name}.png`

const IDENTITY = 'design-surfaces'

async function signIn(page: Page): Promise<void> {
  await page.goto(`/api/app/auth/login?as=${IDENTITY}`)
  await expectSignedIn(page)
}

async function settle(page: Page): Promise<void> {
  await page.waitForLoadState('networkidle')
  await page.evaluate(() => {
    const pending = Array.from(document.images).filter(
      (i) => !i.complete && i.getAttribute('loading') !== 'lazy' && !!i.currentSrc,
    )
    const settled = Promise.all(
      pending.map((i) => new Promise((res) => { i.onload = i.onerror = res })),
    )
    return Promise.race([settled, new Promise((res) => setTimeout(res, 3000))])
  })
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => r(null))))
}

/**
 * Shoot ONE element, with breathing room around it.
 *
 * A bare `locator.screenshot()` crops flush to the element's box, which strips the context that
 * makes a layout judgement possible — a filter strip with no page around it gives no evidence it
 * stopped wrapping. `scrollIntoViewIfNeeded` then a clipped page shot keeps the neighbours.
 */
async function shootNear(page: Page, target: Locator, name: string, pad = 24): Promise<void> {
  await expect(target).toBeVisible({ timeout: 30_000 })
  await target.scrollIntoViewIfNeeded()
  await settle(page)
  const box = await target.boundingBox()
  if (!box) throw new Error(`${name}: element has no box — it is present but not laid out`)
  const vp = page.viewportSize()!
  await page.screenshot({
    path: dir(name),
    clip: {
      x: Math.max(0, box.x - pad),
      y: Math.max(0, box.y - pad),
      width: Math.min(vp.width - Math.max(0, box.x - pad), box.width + pad * 2),
      height: Math.min(vp.height - Math.max(0, box.y - pad), box.height + pad * 2),
    },
  })
}

/**
 * Seed follows and saves.
 *
 * The design identity is a constant account that starts empty, so Following, Saved, the Profile
 * pills and the trends expand control all correctly render nothing — and a screenshot run that
 * skipped four items looked, from the outside, exactly like one where those items were broken.
 * Seeded through the API rather than by clicking, because the UI path costs a minute of navigation
 * per item and proves nothing this spec is asking about.
 */
async function seedFollowsAndSaves(page: Page): Promise<void> {
  await page.evaluate(async () => {
    const j = (u: string) => fetch(u, { credentials: 'include' }).then((r) => r.json())
    const post = (u: string, body: unknown, method = 'POST') =>
      fetch(u, {
        method,
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

    // Follow two shows, so Following has rows and its type chips have something to filter.
    const shows = (await j('/api/app/podcasts')).items ?? []
    for (const s of shows.slice(0, 3)) {
      await post('/api/app/library', { feed_id: s.feed_id, title: s.title })
    }

    // Save a show and a couple of episodes, so Saved renders its sections and its filter bar.
    const eps = (await j('/api/app/episodes?page_size=8')).items ?? []
    for (const e of eps.slice(0, 7)) {
      await post('/api/app/favorites', { kind: 'episode', ref: e.slug, label: e.title }, 'PUT')
    }
    if (shows[0]) {
      await post(
        '/api/app/favorites',
        { kind: 'show', ref: shows[0].feed_id, label: shows[0].title },
        'PUT',
      )
    }

    // One of EACH interest kind — the whole point of the profile-pill shot is the contrast.
    const topics = (await j('/api/app/trending?kind=topic&limit=2')).items ?? []
    const people = (await j('/api/app/trending?kind=person&limit=1')).items ?? []
    const stories = (await j('/api/app/trending?kind=storyline&limit=1')).items ?? []
    const clusters = (await j('/api/app/clusters?limit=1')).items ?? []

    // Two boards and a few notes — without them the Boards headings render no count at all
    // (correctly: the count is gated on there being something to count), so the shot of "does the
    // heading carry a tally" would be a shot of an empty tab.
    for (const name of ['Weekend reading', 'Risk & rates']) {
      await post('/api/app/collections', { name })
    }
    const noteTargets: Array<[string, string, string]> = [
      ['episode', eps[0]?.slug ?? '', 'The framing here is the useful part.'],
      ['episode', eps[1]?.slug ?? '', 'Come back to the second half.'],
      ...topics.slice(0, 1).map((t: { entity_id: string }): [string, string, string] => [
        'topic',
        t.entity_id,
        'Worth tracking how this develops.',
      ]),
    ]
    for (const [target, target_id, text] of noteTargets) {
      if (target_id) await post('/api/app/notes', { target, target_id, text })
    }

    await post(
      '/api/app/interests',
      {
        items: [
          ...topics.map((t: { entity_id: string }) => t.entity_id),
          ...people.map((p: { entity_id: string }) => p.entity_id),
          ...stories.map((s: { entity_id: string }) => s.entity_id),
          ...clusters.map((c: { id: string }) => c.id),
        ],
      },
      'PUT',
    )
  })
}

/**
 * Shoot the band spanning TWO elements — for an ask that is about ORDER rather than appearance.
 * Cropping to one of them proves nothing: "the arc moved up beside the sparkline" is only visible
 * when both are in frame with nothing between them.
 */
async function shootSpan(page: Page, top: Locator, bottom: Locator, name: string): Promise<void> {
  await expect(top).toBeVisible({ timeout: 30_000 })
  await expect(bottom).toBeVisible({ timeout: 30_000 })
  await top.scrollIntoViewIfNeeded()
  await settle(page)
  const a = await top.boundingBox()
  const b = await bottom.boundingBox()
  if (!a || !b) throw new Error(`${name}: one of the two elements has no box`)
  const vp = page.viewportSize()!
  const y = Math.max(0, a.y - 12)
  await page.screenshot({
    path: dir(name),
    clip: {
      x: 0,
      y,
      width: vp.width,
      height: Math.min(vp.height - y, b.y + b.height + 12 - y),
    },
  })
}

/** Open the first topic card reachable from Discover's trends. */
async function openTopicCard(page: Page): Promise<void> {
  await page.goto('/browse')
  await page.locator('[data-testid="discovery-tab-topic"]').click()
  await page.locator('[data-testid="discovery-row"] button').first().click()
  await expect(page.locator('[data-testid="ec-top-voices"], [data-testid="ec-similar-topic"]').first())
    .toBeVisible({ timeout: 30_000 })
}

// --- Home ----------------------------------------------------------------------------------- //

/**
 * Seed an in-progress listen. Both the resume hero and Recommended are gated on one — Recommended
 * is keyed off the most recent play — so on a fresh account neither section exists and a shot of
 * "Home" would be a shot of their absence.
 */
async function seedInProgress(page: Page): Promise<void> {
  await page.goto('/')
  await page.locator('a[href*="/episode/"]').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await page.evaluate(() => {
    const a = document.querySelector('audio')
    if (a) {
      a.currentTime = 120
      a.dispatchEvent(new Event('timeupdate'))
    }
  })
  await page.waitForTimeout(1200) // let the position write land
}

test('home: recommended caps at four with a show-more', async ({ page }) => {
  await signIn(page)
  await seedInProgress(page)
  await page.goto('/')
  const heading = page.getByText('Recommended for you')
  if (!(await heading.count())) {
    test.skip(true, 'no recommendations for this listen — the section correctly does not render')
  }
  await shootNear(page, heading.locator('xpath=ancestor::section[1]'), 'recommended-four')
})

test('home: the queue is reachable beside Resume', async ({ page }) => {
  await signIn(page)
  // Play something so the resume hero exists at all — it is gated on in-progress listening.
  await page.goto('/')
  await page.locator('a[href*="/episode/"]').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await page.evaluate(() => {
    const a = document.querySelector('audio')
    if (a) {
      a.currentTime = 120
      a.dispatchEvent(new Event('timeupdate'))
    }
  })
  await page.goto('/')
  const resume = page.locator('[data-testid="home-resume"]')
  if (await resume.count()) {
    await shootNear(page, resume.locator('xpath=..'), 'queue-beside-resume')
  } else {
    test.skip(true, 'no in-progress listen in this corpus — the resume hero does not render')
  }
})

// --- Discover ------------------------------------------------------------------------------- //

test('discover: "all" expands the trends list in place', async ({ page }) => {
  await signIn(page)
  await page.goto('/browse')
  const seeAll = page.locator('[data-testid="discovery-see-all"]')
  if (!(await seeAll.count())) {
    test.skip(true, 'fewer trend rows than the cap — the expand control correctly does not render')
  }
  await shootNear(page, page.locator('[data-testid="discovery-explorer"]'), 'trends-all-collapsed')
  await seeAll.click()
  await shootNear(page, page.locator('[data-testid="discovery-explorer"]'), 'trends-all-expanded')
})

test('discover: the episodes load-more matches the shows one', async ({ page }) => {
  await signIn(page)
  await page.goto('/browse?tab=episodes')
  const more = page.locator('[data-testid="catalog-load-more"]')
  if (!(await more.count())) test.skip(true, 'corpus fits on one page — no load-more to shoot')
  await shootNear(page, more, 'episodes-load-more')
})

// --- Topic card ----------------------------------------------------------------------------- //

test('topic: the conversation arc sits under the activity sparkline', async ({ page }) => {
  await signIn(page)
  await openTopicCard(page)
  const spark = page.locator('[data-testid="ec-topic-activity"]')
  if (!(await spark.count())) test.skip(true, 'this topic has too few dated episodes for a sparkline')
  // Span the sparkline through the arc: the ask was about ORDER, so both must be in one frame.
  const arc = page.locator('[data-testid="topic-conversation-arc"]')
  if (!(await arc.count())) {
    await shootNear(page, spark.locator('xpath=..'), 'topic-charts-stacked', 8)
    test.skip(true, 'no conversation arc for this topic — shot the sparkline alone')
  }
  await shootSpan(page, spark, arc, 'topic-charts-stacked')
})

test('topic: similar topics excludes the topic you are on', async ({ page }) => {
  await signIn(page)
  await openTopicCard(page)
  const chips = page.locator('[data-testid="ec-similar-topic"]')
  if (!(await chips.count())) test.skip(true, 'this topic has no siblings')
  await shootNear(page, chips.first().locator('xpath=../..'), 'similar-topics-no-self')
})

test('topic: strongest shows carry artwork', async ({ page }) => {
  await signIn(page)
  await openTopicCard(page)
  const shows = page.locator('[data-testid="ec-top-shows"]')
  if (!(await shows.count())) test.skip(true, 'this topic spans a single show')
  await shootNear(page, shows, 'top-shows-artwork')
})

test('topic: the episode list caps at ten', async ({ page }) => {
  await signIn(page)
  await openTopicCard(page)
  const more = page.locator('[data-testid="entity-episodes-more"]')
  if (!(await more.count())) test.skip(true, 'fewer than ten episodes on this topic')
  await shootNear(page, more, 'entity-episodes-cap')
})

// --- Library -------------------------------------------------------------------------------- //

test('library: the Following filters sit on one row', async ({ page }) => {
  await signIn(page)
  // Following is the `shows` tab, and it renders the SAME `SavedFilterBar` as Saved (minus the
  // colour strip), so it carries Saved's static testids rather than ones of its own.
  await page.goto('/')
  await seedFollowsAndSaves(page)
  await page.goto('/library?tab=shows')
  const bar = page.locator('[data-testid="saved-filter-bar"]')
  if (!(await bar.count())) test.skip(true, 'this account follows nothing — the bar does not render')
  await shootNear(page, bar, 'following-one-row')
})

test('library: Saved type filters, colours and mute all sit on one row', async ({ page }) => {
  await signIn(page)
  await page.goto('/')
  await seedFollowsAndSaves(page)
  await page.goto('/library?tab=saved')
  const bar = page.locator('[data-testid="saved-filter-bar"]')
  if (!(await bar.count())) test.skip(true, 'nothing saved on this account — the bar does not render')
  await shootNear(page, bar, 'saved-filters-one-row')
})

test('library: Boards headings carry their counts', async ({ page }) => {
  await signIn(page)
  await page.goto('/')
  await seedFollowsAndSaves(page)
  await page.goto('/library?tab=collections')
  await shootNear(page, page.getByRole('heading', { name: /collections/i }).first(), 'boards-counts')
})

test('library: the Notes filters sit on one row and carry counts', async ({ page }) => {
  await signIn(page)
  await page.goto('/')
  await seedFollowsAndSaves(page)
  await page.goto('/library?tab=collections')
  const filter = page.locator('[data-testid="notes-type-filter"]')
  if (!(await filter.count())) {
    test.skip(true, 'fewer than two note kinds — the strip correctly does not render')
  }
  await shootNear(page, filter, 'notes-filter-counts')
})

// --- Profile -------------------------------------------------------------------------------- //

test('profile: topic, theme, storyline and person pills are told apart', async ({ page }) => {
  await signIn(page)
  await page.goto('/')
  await seedFollowsAndSaves(page)
  await page.goto('/profile?tab=topics')
  const pills = page.locator('[data-testid^="profile-interest-"]')
  if (!(await pills.count())) test.skip(true, 'this account follows nothing yet')
  await shootNear(page, pills.first().locator('xpath=..'), 'profile-interest-kinds')
})

// --- Player --------------------------------------------------------------------------------- //

test('player: the transcript line reads time, separator, speaker — flush left', async ({ page }) => {
  await signIn(page)
  await page.goto('/')
  await page.locator('a[href*="/episode/"]').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  // Opt-in per episode: `transcriptOpen` starts false and the panel is `hidden` until the toggle
  // is pressed, so the element exists from first paint and is not visible. Waiting on visibility
  // without pressing it is a guaranteed 30s timeout.
  await page.locator('[data-testid="transcript-toggle"]').click()
  const transcript = page.locator('[data-testid="transcript"]')
  await expect(transcript).toBeVisible({ timeout: 30_000 })
  // Proof the capture is of real transcript content, not an empty scroller.
  await expect(transcript.locator('[data-testid="seg"]').first()).toBeVisible({ timeout: 30_000 })
  await shootNear(page, transcript, 'transcript-time-then-speaker', 8)
})
