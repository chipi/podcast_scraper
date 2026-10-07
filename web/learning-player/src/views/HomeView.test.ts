import { flushPromises, mount } from '@vue/test-utils'
import { kindPill } from '../utils/interests'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary, Me, Podcast } from '../services/types'
import { resetStaleness } from '../composables/useSectionState'
import { useDownloadsStore } from '../stores/downloads'
import { useInterestsStore } from '../stores/interests'
import { useLibraryStore } from '../stores/library'
import HomeView from './HomeView.vue'

// Defaults to "nothing cached", so every test above keeps the behaviour it was written for.
const readCached = vi.fn(async (_k: string): Promise<unknown> => null)
// The device-side sources `localContinue` reads. Default: nothing recorded, nothing downloaded —
// so every test above keeps the behaviour it was written for.
const allPositions = vi.fn(
  (): Array<{ slug: string; seconds: number; finished: boolean; updatedAt: number }> => [],
)
const localKnowledgeFor = vi.fn(async (_s: string): Promise<unknown> => null)
const localArtworkFor = vi.fn((_s: string): string | null => null)
vi.mock('../services/playbackPositions', async (orig) => ({
  ...(await orig<typeof import('../services/playbackPositions')>()),
  allPositions: () => allPositions(),
}))
vi.mock('../services/downloads', async (orig) => ({
  ...(await orig<typeof import('../services/downloads')>()),
  localKnowledgeFor: (s: string) => localKnowledgeFor(s),
  localArtworkFor: (s: string) => localArtworkFor(s),
}))

vi.mock('../services/contentCache', () => ({
  isArrayCache: (v: unknown) => Array.isArray(v),
  hasArrayFields:
    (...f: string[]) =>
    (v: unknown) =>
      typeof v === 'object' &&
      v !== null &&
      !Array.isArray(v) &&
      f.every((k) => Array.isArray((v as Record<string, unknown>)[k])),
  readCached: (k: string) => readCached(k),
  writeCached: async () => {},
}))
import homeViewSource from './HomeView.vue?raw'
import episodeTileSource from '../components/EpisodeTile.vue?raw'
import { useAuthStore } from '../stores/auth'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    { path: '/browse', name: 'browse', component: { template: '<div/>' } },
    { path: '/catalog', name: 'catalog', component: { template: '<div/>' } },
    { path: '/search', name: 'search', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    // Home's resume hero links here now (operator 2026-09-19): the queue's only other entrances
    // are the full player and the mini-player, so with nothing playing it was unreachable.
    { path: '/queue', name: 'queue', component: { template: '<div/>' } },
    { path: '/browse/topics', name: 'browse-topics', component: { template: '<div/>' } },
    { path: '/browse/people', name: 'browse-people', component: { template: '<div/>' } },
  ],
})

function ep(slug: string, title: string): EpisodeSummary {
  return {
    slug, title, feed_id: 'f', podcast_title: 'Show', publish_date: '2024-01-01',
    duration_seconds: 1800, episode_image_url: null, feed_image_url: null, artwork_url: null,
    status: 'ready', summary_preview: 'r', topics: [], has_transcript: true, has_summary: true,
    has_gi: false, has_kg: false, has_bridge: false,
    // Required by EpisodeSummary. Absent here for a long time — test files are excluded
    // from tsconfig.app.json, so nothing type-checks fixtures against the real shape.
    summary_text: null, summary_bullets: [],
  }
}

beforeEach(() => {
  setActivePinia(createPinia())
  // The embedded TrendingTopics + Storylines fetch trending topics / theme clusters; keep these
  // tests off the network (their own coverage lives in TrendingTopics.test.ts / Storylines.test.ts).
  vi.spyOn(api, 'getTrendingTopics').mockResolvedValue({
    has_velocity_data: false,
    window_months: [],
    topics: [],
    storylines: [],
  })
  vi.spyOn(api, 'getStorylines').mockResolvedValue([])
  vi.spyOn(api, 'getTrending').mockResolvedValue([])
  // Home loads the user's interests (to gate the "choose interests" card); default to none so the
  // card shows for signed-in users as before. Individual tests override for the "already chosen" case.
  vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
  try {
    localStorage.removeItem('lp.interests.dismissed')
  } catch {
    /* happy-dom storage edge — ignore */
  }
})
afterEach(() => vi.restoreAllMocks())

/**
 * Home inside a `<KeepAlive>`, which is how `App.vue` renders it — and how EVERY test in this file
 * mounts it (#2024).
 *
 * `onActivated` only fires for a component under KeepAlive. Home loads continue-listening there,
 * and refreshes it on every return, so a plain `mount(HomeView)` left that section in `loading`
 * forever: the rail was untestable, and every assertion in this file was written against a page
 * where it had never resolved. `mount(HomeView)` and the running app took different code paths,
 * which is the sharp edge — a green test here did not mean the page worked.
 */
/**
 * Home under KeepAlive with a switch, so a test can leave and come back.
 *
 * Returning to Home is `onActivated` firing a SECOND time, which is the whole reason
 * `refreshContinueQuietly` exists — and nothing exercised it (#2024).
 */
function mountReturnable() {
  const wrapper = mount(
    {
      components: { HomeView },
      data: () => ({ here: true }),
      template: '<KeepAlive><HomeView v-if="here" /></KeepAlive>',
    },
    { global: { plugins: [i18n, router] } },
  )
  return {
    wrapper,
    leave: async () => {
      await wrapper.setData({ here: false })
      await flushPromises()
    },
    comeBack: async () => {
      await wrapper.setData({ here: true })
      await flushPromises()
    },
  }
}

function mountKeptAlive() {
  return mount(
    { components: { HomeView }, template: '<KeepAlive><HomeView /></KeepAlive>' },
    { global: { plugins: [i18n, router] } },
  )
}

function signIn(): void {
  useAuthStore().user = { user_id: 'u', email: 'e@x.com', name: 'N' } as unknown as Me
}

describe('HomeView (discover state, signed out)', () => {
  it('renders the ask hero and What\'s new, but no shows section', async () => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({
      items: [ep('a-1', 'First Ep'), ep('a-2', 'Second Ep')], scope: 'all',
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      { feed_id: 'showa', title: 'Show A', artwork_url: null, image_url: null, episode_count: 2 } as Podcast,
    ])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([]) // no history → discover state

    const w = mountKeptAlive()
    await flushPromises()
    expect(w.text()).toContain("Find what's worth hearing") // discover hero
    expect(w.text()).toContain("What's new")
    expect(w.text()).toContain('First Ep')
    // "Your shows" is per-user (#1585): signed out there are no follows, so no section — and
    // crucially it must NOT fall back to showing the whole catalogue, which is what it used to do.
    expect(w.text()).not.toContain('Your shows')
    expect(w.text()).not.toContain('Show A')
  })

  it('shows artwork on What\'s-new rows 02+ (#15 variant A)', async () => {
    const row = ep('a-2', 'Second Ep')
    row.artwork_url = 'https://x/row.png'
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({
      items: [ep('a-1', 'First Ep'), row], scope: 'all',
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])

    const w = mountKeptAlive()
    await flushPromises()
    // The ranked row (index 1) now carries a square thumbnail from its artwork.
    expect(w.find('img[src="https://x/row.png"]').exists()).toBe(true)
  })

  it("What's new: #01 carries ♡ queue ⋯ in a row; 02+ stack queue ⋯ with the ♡ in the ⋯ (operator 2026-10-07)", async () => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({
      items: [ep('a-1', 'First Ep'), ep('a-2', 'Second Ep'), ep('a-3', 'Third Ep')], scope: 'all',
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])

    const w = mountKeptAlive()
    await flushPromises()
    const rows = w.findAll('[data-testid="episode-actions"]')
    expect(rows).toHaveLength(3) // #01 + two ranked rows
    // Each control by its test id, or its accessible name where it has none (the queue toggle).
    const controls = (r: (typeof rows)[number]) =>
      r.findAll('button').map((b) => b.attributes('data-testid') ?? b.attributes('aria-label'))
    const featured = controls(rows[0])
    expect(featured).toHaveLength(3)
    expect(featured[0]).toBe('favorite-button')
    expect(featured[1]).toMatch(/queue/i)
    expect(featured[2]).toBe('overflow-trigger')
    // 02+: the heart lives in the ⋯, so each stacked row is two targets tall, not three.
    for (const r of rows.slice(1)) {
      const got = controls(r)
      expect(got).toHaveLength(2)
      expect(got[0]).toMatch(/queue/i)
      expect(got[1]).toBe('overflow-trigger')
    }
    expect(rows[0].classes()).not.toContain('flex-col') // the #01 card: one row, top right
    expect(rows[1].classes()).toContain('flex-col') // 02+: one column, top to bottom
    expect(rows[2].classes()).toContain('flex-col')
  })

  it("What's new: the card and rows OPEN the episode — only Resume plays (operator 2026-10-05)", async () => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({
      items: [ep('a-1', 'First Ep'), ep('a-2', 'Second Ep')], scope: 'all',
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])

    const w = mountKeptAlive()
    await flushPromises()
    const hrefs = w.findAll('a').map((a) => a.attributes('href') ?? '')
    expect(hrefs).toContain('/episode/a-1') // the #01 card
    expect(hrefs).toContain('/episode/a-2') // a ranked row
    expect(hrefs.some((h) => h.includes('play=1'))).toBe(false)
    // No separate ▶, inside the link or out.
    expect(w.findAll('a[href^="/episode/"] [aria-hidden="true"]').some((s) => s.text() === '▶')).toBe(false)
  })

  it('has no Trends — they live on Discover only (operator 2026-10-07)', async () => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({
      items: [ep('a-1', 'First Ep')], scope: 'all',
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.find('[data-testid="home-discovery"]').exists()).toBe(false)
    expect(w.find('[data-testid="discovery-tab-topic"]').exists()).toBe(false)
    // The way to them stays: the Discover strip deep-links into Discover's Trends.
    expect(w.find('[data-testid="home-browse-nav"]').exists()).toBe(true)
  })

  it('the Discover strip links all four Trends kinds, Themes included (operator 2026-10-07)', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    const w = mountKeptAlive()
    await flushPromises()
    const nav = w.get('[data-testid="home-browse-nav"]')
    expect(nav.findAll('a').map((a) => a.attributes('href'))).toEqual([
      '/browse?trends=topic',
      '/browse?trends=person',
      '/browse?trends=theme',
      '/browse?trends=storyline',
    ])
  })

  it('each Discover chip wears its kind colour, the one Interests uses (operator 2026-10-08)', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    const w = mountKeptAlive()
    await flushPromises()
    for (const kind of ['topic', 'person', 'theme', 'storyline'] as const) {
      const chip = w.get(`[data-testid="home-discover-${kind === 'topic' ? 'topics' : kind === 'person' ? 'people' : `${kind}s`}"]`)
      for (const cls of kindPill(kind).split(' ')) expect(chip.classes()).toContain(cls)
    }
    // One row: the lead-in is a title line above the chips, not a chip-row sibling.
    expect(w.get('[data-testid="home-browse-nav"] > div').classes()).toContain('flex-nowrap')
  })

  it("Recommended is absent until there is a basis for it (operator 2026-10-07)", async () => {
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({ items: [ep('n-1', 'New One')], scope: 'all' })
    vi.spyOn(api, 'getRecommended').mockResolvedValue({ items: [], basis: 'none' })
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.find('[data-testid="home-recommended"]').exists()).toBe(false)
  })

  it("with no listen yet, Recommended comes from what you follow, and never repeats What's new", async () => {
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({ items: [ep('dup', 'Already New')], scope: 'yours' })
    vi.spyOn(api, 'getRecommended').mockResolvedValue({
      items: [ep('dup', 'Already New'), ep('pick-1', 'A Pick')],
      basis: 'interests',
    })
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    const rec = w.get('[data-testid="home-recommended"]')
    expect(rec.text()).toContain('Picked from what you follow')
    expect(rec.text()).toContain('A Pick')
    expect(rec.text()).not.toContain('Already New')
  })

  it("puts Recommended above What's new (operator 2026-10-07)", () => {
    const tpl = homeViewSource.slice(homeViewSource.indexOf('<template>'))
    const rec = tpl.indexOf(":title=\"t('home.recommended')\"")
    const wn = tpl.indexOf(":title=\"t('home.whatsNew')\"")
    expect(rec).toBeGreaterThan(0)
    expect(wn).toBeGreaterThan(0)
    expect(rec).toBeLessThan(wn)
  })

  it('submitting the search navigates to /search', async () => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({ items: [], scope: 'all', })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    const push = vi.spyOn(router, 'push')
    const w = mountKeptAlive()
    await flushPromises()
    await w.find('input#home-search').setValue('memory')
    await w.find('form').trigger('submit')
    expect(push).toHaveBeenCalledWith({ name: 'search', query: { q: 'memory' } })
  })
})

describe('HomeView distinguishes empty from broken (#1591)', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
  })

  it('renders an error with a retry when the fetch fails', async () => {
    // The defect: every section did `.catch(() => [])` and then hid itself when empty, so a total
    // API outage rendered the same page as a brand-new account — a hero, a search box, two chips.
    vi.spyOn(api, 'getWhatsNew').mockRejectedValue(new Error('boom'))
    const w = mountKeptAlive()
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(true)
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
    // The section header still renders — it is what tells you this content exists at all.
    expect(w.text()).toContain("What's new")
  })

  it('retry re-fetches and recovers', async () => {
    // No error state anywhere in the app previously offered a retry: the only move was a reload.
    const spy = vi.spyOn(api, 'getWhatsNew').mockRejectedValueOnce(new Error('boom'))
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.find('[data-testid="section-error"]').exists()).toBe(true)

    spy.mockResolvedValueOnce({ items: [ep('a-1', 'Recovered Ep')], scope: 'all' })
    await w.get('[data-testid="section-retry"]').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
    expect(w.text()).toContain('Recovered Ep')
  })

  it('a successful-but-empty load still hides the section', async () => {
    // Hide when the SYSTEM is empty — there is no action the user can take, so an empty shell is
    // noise. Contrast "Your shows", which is empty because of a user action not yet taken.
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({
      items: [], scope: 'all',
    })
    const w = mountKeptAlive()
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
    expect(w.text()).not.toContain("What's new")
  })
})

// "Your shows" was removed from Home (operator 2026-09-14) — the shows you follow already live in
// Library › Following, so the four tests that asserted the follows-grid on Home were removed with it.

describe('HomeView interests card (3.5)', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({ items: [], scope: 'all', })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
  })

  it('is hidden when signed out', async () => {
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.text()).not.toContain('Personalize your Home')
  })

  it('shows to signed-in users and opens the cluster picker', async () => {
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([{ id: 'tc:ai', label: 'AI', size: 3 }])
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router], stubs: { teleport: true } } })
    await flushPromises()
    expect(w.text()).toContain('Personalize your Home')
    await w.findAll('button').find((b) => b.text() === 'Choose interests')!.trigger('click')
    await flushPromises()
    expect(w.find('[role="dialog"]').exists()).toBe(true)
    // A theme suggestion in the picker, behind its section's + Add.
    await w.get('[data-testid="interest-add-theme"]').trigger('click')
    await flushPromises()
    expect(w.findAll('[data-testid="interest-suggestion"]').map((b) => b.text())).toContain('+ AI')
  })

  it('welcomes the person by first name, and offers two real buttons (operator 2026-10-07)', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    useAuthStore().user = { user_id: 'u', email: 'm@x.com', name: 'Marko Dragoljevic' } as unknown as Me
    const w = mountKeptAlive()
    await flushPromises()
    const card = w.get('[data-testid="interests-welcome"]')
    expect(card.text()).toContain('Welcome, Marko')
    expect(card.text()).toContain('shape your Home and recommendations')
    // Buttons, with the labels the device journeys find them by.
    expect(card.get('[data-testid="interests-choose"]').text()).toBe('Choose interests')
    expect(card.get('[data-testid="interests-not-now"]').text()).toBe('Not now')
    // Step 1 of the guided start (operator 2026-10-07): shows are step 2, reached by Skip or by
    // choosing three interests.
    expect(card.attributes('data-step')).toBe('1')
    expect(card.get('[data-testid="guided-skip"]').text()).toBe('Skip step')
  })

  it('does not greet an email-link account by its address', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    useAuthStore().user = { user_id: 'u', email: 'm@x.com', name: 'm@x.com' } as unknown as Me
    const w = mountKeptAlive()
    await flushPromises()
    const card = w.get('[data-testid="interests-welcome"]')
    expect(card.text()).toContain('Welcome to Close Listening')
    expect(card.text()).not.toContain('m@x.com')
  })

  it('an empty Your Week stays hidden during and after the guided start (operator 2026-10-08)', async () => {
    // A week in review for someone who has done nothing yet is empty by construction, and an
    // empty one is skipped, not explained.
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'getYourWeek').mockResolvedValue({ sections: [] } as never)
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.find('[data-testid="interests-welcome"]').exists()).toBe(true)
    expect(w.find('[data-testid="your-week"]').exists()).toBe(false)
    // Three interests and a followed show reach the last step; finishing closes the flow.
    useInterestsStore().ids = ['tc:ai', 'tc:science', 'topic:risk']
    useLibraryStore().items = [{ feed_id: 'f1' } as never]
    await flushPromises()
    expect(w.get('[data-testid="interests-welcome"]').attributes('data-step')).toBe('3')
    await w.get('[data-testid="guided-finish"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="interests-welcome"]').exists()).toBe(false)
    expect(w.find('[data-testid="your-week"]').exists()).toBe(false)
  })

  it('"Not now" closes the whole guide; "Skip step" only moves past the step', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    await w.get('[data-testid="guided-skip"]').trigger('click')
    expect(w.get('[data-testid="interests-welcome"]').attributes('data-step')).toBe('2')
    await w.get('[data-testid="interests-not-now"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="interests-welcome"]').exists()).toBe(false)
  })

  it('Your Week with content shows even beside the welcome card (followed a show, no interests)', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'getYourWeek').mockResolvedValue({
      sections: [{ kind: 'follows', items: [{ episode_slug: 'e1', episode_title: 'Followed Ep', deep_link: '/e/e1' }] }],
    } as never)
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.find('[data-testid="interests-welcome"]').exists()).toBe(true)
    expect(w.find('[data-testid="your-week"]').exists()).toBe(true)
  })

  it('a listener who already has interests sees Your Week straight away', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['tc:ai'])
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.find('[data-testid="your-week"]').exists()).toBe(true)
  })

  it('dismissing hides the card', async () => {
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    await w.findAll('button').find((b) => b.text() === 'Not now')!.trigger('click')
    expect(w.text()).not.toContain('Personalize your Home')
  })

  it('is hidden when the user already has interests (regression)', async () => {
    // The bug: the card showed even to users with a full interest set. It must only offer to pick
    // interests when there are none.
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['tc:ai', 'tc:science'])
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.text()).not.toContain('Personalize your Home')
  })

  it('reflects interests SAVED IN SESSION, not only those present at load', async () => {
    // iOS-F1, caught by `PersonalisationTests.test10` on device and by nothing here.
    //
    // The test above loads an account that ALREADY has interests, which the store gets right for
    // free. The broken case is choosing them WHILE the app is open: `InterestsPicker.save()` PUT
    // the list and told nobody, so this card — gated on `interests.ids.length === 0` — went on
    // asking. Home only appeared correct because its own handler set the DISMISSED flag, which
    // hid the card for the wrong reason and only on the path that starts from Home. Save from
    // Profile, come back to Home, and it still prompted. Present since #1111 (2026-06-28).
    //
    // Asserting through the STORE, not the flag: that is what `showInterestsCard` reads, and it is
    // the thing every other surface shares.
    // Driven through the PICKER, not by poking the store: the seam is exactly where the bug lived,
    // and a test that calls `replaceAll` itself would have passed against the broken code.
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([{ id: 'tc:ai', label: 'AI', size: 5 }])
    vi.spyOn(api, 'putUserInterests').mockResolvedValue(['tc:ai'])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router], stubs: { teleport: true } } })
    await flushPromises()
    expect(w.text()).toContain('Personalize your Home') // precondition: it IS asking

    await w.findAll('button').find((b) => b.text() === 'Choose interests')!.trigger('click')
    await flushPromises()
    // A suggestion reads "+ AI", behind its section's + Add, since the sheet became the four
    // interest sections (2026-10-04).
    await w.get('[data-testid="interest-add-theme"]').trigger('click')
    await flushPromises()
    await w.findAll('button').find((b) => b.text() === '+ AI')!.trigger('click')
    await w.findAll('button').find((b) => b.text() === 'Save')!.trigger('click')
    await flushPromises()

    expect(useInterestsStore().ids).toEqual(['tc:ai'])
    // The card reads the STORE: one saved interest shows as progress toward the three the guided
    // start asks for (operator 2026-10-07), rather than a card still asking from zero.
    expect(w.get('[data-testid="guided-interests-to-go"]').text()).toBe('2 more to go')
    // And NOT because we marked the offer declined — that would suppress it forever, so a user who
    // later cleared their interests would never be offered it again.
    expect(localStorage.getItem('lp.interests.dismissed')).toBeNull()
  })

  // Compact Discover strip (renamed from Browse topics/people, operator 2026-09-14): chips
  // deep-linking into Browse's Trends section on the matching kind. The standalone /trends page
  // they used to open was a thinner copy of that section and is deleted (operator 2026-09-18).
  it('renders the compact "Discover" strip as Browse trends deep links', async () => {
    const w = mountKeptAlive()
    await flushPromises()
    const nav = w.get('[data-testid="home-browse-nav"]')
    const links = nav.findAll('a')
    const hrefs = links.map((a) => a.attributes('href'))
    expect(hrefs).toContain('/browse?trends=topic')
    expect(hrefs).toContain('/browse?trends=storyline')
    expect(hrefs).toContain('/browse?trends=person')
    expect(nav.text()).toContain('Explore what people are talking about')
  })

  it('carries no Trending shows section — it lives on Discover (operator 2026-10-05)', async () => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({
      items: [], scope: 'all',
    })
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    const trending = vi.spyOn(api, 'getTrending')
    const catalogue = vi.spyOn(api, 'getPodcasts')
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.find('[data-testid="trending-shows-rail"]').exists()).toBe(false)
    // The catalogue fetch existed only to give that rail its artwork.
    expect(catalogue).not.toHaveBeenCalled()
    expect(trending.mock.calls.some((c) => c[0] === 'show')).toBe(false)
  })

  // --- an outage must not look like a new account (#1591, S7) ---
  //
  // Continue was one of the last sections on `.catch(() => [])` (#1591). Its twin, the follows
  // catalogue, left Home with Trending shows (2026-10-05).

  it('a playback outage does not silently swap the resume hero for the discover hero', async () => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({
      items: [], scope: 'all',
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockRejectedValue(new Error('502'))
    signIn()

    const w = mountKeptAlive()
    await flushPromises()

    // A playback outage shows the honest error skeleton at the top, not a fabricated hero. (The
    // ask/search title moved to its own section lower down (H.3), so its presence no longer signals
    // a top "discover hero" — the swap this guarded against can't happen: the top is resume-or-error.)
    expect(w.find('[data-testid="section-error"]').exists()).toBe(true)
  })
})

describe('the primary controls share one height (#2004 item 2)', () => {
  // The search row renders in BOTH hero states, so it needs no special setup. Resume only exists in
  // the resume state, which needs auth + playback history — see `resumeState` in HomeView.
  it('the search input and Search button declare one shared height', async () => {
    // They were sized by padding plus inherited font-size, so the height was emergent: Resume ~40px,
    // input ~46px, button ~48px. Nobody chose those numbers. Asserting the class rather than a
    // measured height because jsdom does not lay out — the point is that ONE value is stated.
    const w = mountKeptAlive()
    await flushPromises()
    for (const id of ['home-search-input', 'home-search-submit']) {
      const el = w.find(`[data-testid="${id}"]`)
      expect(el.exists(), `${id} should render`).toBe(true)
      expect(el.classes(), `${id} should declare the shared height`).toContain('h-11')
    }
  })

  it('sizes them by height, not by vertical padding', async () => {
    // The regression to prevent: someone re-adds `py-*` and the controls drift apart again.
    const w = mountKeptAlive()
    await flushPromises()
    for (const id of ['home-search-input', 'home-search-submit']) {
      const cls = w.find(`[data-testid="${id}"]`).classes()
      expect(cls.filter((c) => /^py-\d/.test(c)), `${id} should not set vertical padding`).toEqual([])
    }
  })

  it('Resume uses the same height as the search row', () => {
    // Source-level: the resume hero needs auth + playback history to render, and the value under
    // test is a static class. Pinned so the three cannot drift apart again.
    expect(homeViewSource).toMatch(/data-testid="home-resume"[\s\S]{0,200}?\bh-11\b/)
  })

  it('Resume RESUMES — it carries play=1, not just the right height', () => {
    // The only assertion on this control was its height class, so the fix that made it start
    // playing instead of opening paused could be reverted silently (review 2026-09-18). Source-level
    // for the same reason as above: the hero needs auth + playback history to render.
    expect(homeViewSource).toMatch(
      /data-testid="home-resume"[\s\S]{0,400}?play:\s*'1'|play:\s*'1'[\s\S]{0,400}?data-testid="home-resume"/,
    )
  })

  it('the resume hero is Resume ALONE — the queue link moved to the masthead', () => {
    // It carried a queue link from 2026-09-19 until 2026-09-27, added because nothing else could
    // open the queue with no episode in progress. The masthead now carries one at every width and
    // on every screen, which covers that case strictly better — a hero that only renders WHILE
    // something is in progress is the worst possible home for "I finished everything, show me what
    // I queued". Removed as duplication, not as a capability.
    //
    // The invariant it was really guarding — the queue stays reachable without playing something
    // first — did not go away with it. It moved to `queue is reachable at every width` in
    // __checks__/touch-affordances.test.ts, against the masthead control.
    expect(homeViewSource).not.toContain('data-testid="home-open-queue"')
  })
})

describe('cards align by the tile, not by cutting text (#2004 items 3/3b)', () => {
  it('the Recommended grid clips neither the title nor the show name', async () => {
    // The clamped title actually overflowed INTO the show name here — an ellipsis at line two AND a
    // visible third line, because the clamp computed but the overflow still painted. The specific
    // defect was the RESERVED HEIGHT (`min-h-[2.5rem]`), not clamping as such: the clamp computed
    // against one height while the box painted at another.
    const w = mountKeptAlive()
    await flushPromises()
    expect(homeViewSource).not.toMatch(/line-clamp-2 min-h-\[2\.5rem\]/)
    expect(homeViewSource).not.toMatch(/lp-kicker mt-0\.5 truncate/)
    expect(w.exists()).toBe(true)
  })

  it('renders the SHARED grid tile rather than its own copy of one', () => {
    // Recommended hand-rolled EpisodeTile's shape — square artwork, overlaid actions, show name and
    // title — and the two drifted: title-above-show here against show-above-title there, unclamped
    // here against clamped there. One component now owns the shape (operator 2026-09-17), so this
    // asserts the delegation rather than re-pinning a second copy's classes.
    expect(homeViewSource).toMatch(/<EpisodeTile\s+:episode="ep"/)
    // Scoped to the Recommended section, which is a GRID; the rails are asserted below.
    // Anchored to the HEADING, not to the bare key — `cacheKey: "home.recommended"` sits up in the
    // script block, so starting there swept in every rail between it and the template.
    // Either quote style: the heading moved into `<SectionHeading :title="t('home.recommended')" />`,
    // where the attribute's own double quotes force single quotes inside (operator 2026-09-18).
    const recommendedAnchor = Math.max(
      homeViewSource.indexOf('t("home.recommended")'),
      homeViewSource.indexOf("t('home.recommended')"),
    )
    const recommendedSection = homeViewSource.slice(
      recommendedAnchor,
      homeViewSource.indexOf('<InterestsPicker'),
    )
    expect(recommendedAnchor, 'the Recommended heading anchor vanished').toBeGreaterThan(-1)
    expect(recommendedSection.length, 'could not isolate the Recommended section').toBeGreaterThan(0)
    expect(recommendedSection, 'the grid rebuilt its own tile again').not.toMatch(/aspect-square/)
  })

  it('Jump back in is the standard rail of the standard tile (operator 2026-10-05)', () => {
    // It hand-rolled its own tile — 2-line title, show name UNDER the title, no actions — and was
    // the one episode rail that looked different from every other.
    const jump = homeViewSource.slice(
      homeViewSource.indexOf("t('home.jumpBackIn')"),
      homeViewSource.indexOf('</section>', homeViewSource.indexOf("t('home.jumpBackIn')")),
    )
    expect(jump).toMatch(/<CardRail>/)
    expect(jump).toMatch(/class="lp-rail-item"/)
    expect(jump).toMatch(/<EpisodeTile[\s\S]*:progress=/)
    expect(jump, 'Jump back in rebuilt its own tile').not.toMatch(/aspect-square/)
  })

  it('keeps the grid even — the cell-filling requirement moved to the tile, it did not lapse', () => {
    // #1584's requirement still holds; EpisodeTile is what pays for it now. Asserted at the source
    // of truth so deleting it there fails here too.
    expect(episodeTileSource).toMatch(/<article class="relative flex h-full flex-col/)
  })
})

/**
 * #1909's requirement, in the operator's words: "everything should look the same as last time
 * online, just stale — I should never see 'I cannot load stuff'."
 */
describe('Home with no network shows what it had, not a wall of errors (#1909)', () => {
  beforeEach(() => {
    readCached.mockReset().mockResolvedValue(null)
    resetStaleness()
  })

  it('keeps the cached rail and says it is stale, instead of an error card', async () => {
    readCached.mockImplementation(async (k: string) =>
      k === 'home.whatsnew.v2' ? { items: [ep('a-1', 'From Last Time')], scope: 'all' } : null,
    )
    vi.spyOn(api, 'getWhatsNew').mockRejectedValue(new Error('offline'))
    const w = mountKeptAlive()
    await flushPromises()

    expect(w.text(), 'the cached episode is gone').toContain('From Last Time')
    // The requirement, literally: never "I cannot load stuff" when we hold the content.
    expect(
      w.find('[data-testid="section-error"]').exists(),
      'an error card rendered over content we had cached',
    ).toBe(false)
    expect(
      w.find('[data-testid="stale-notice"]').exists(),
      'nothing said the content was out of date',
    ).toBe(true)
  })

  it('the notice carries the retry that the stale rail no longer has', async () => {
    // A stale section renders no error card, so it offers no `section-retry`. Without this the
    // page would be quieter and have no way to refresh at all.
    readCached.mockImplementation(async (k: string) =>
      k === 'home.whatsnew.v2' ? { items: [ep('a-1', 'From Last Time')], scope: 'all' } : null,
    )
    const spy = vi.spyOn(api, 'getWhatsNew').mockRejectedValue(new Error('offline'))
    const w = mountKeptAlive()
    await flushPromises()
    const calls = spy.mock.calls.length

    spy.mockResolvedValue({
      items: [ep('a-2', 'Fresh Again')],
      scope: 'all',
    })
    await w.get('[data-testid="stale-retry"]').trigger('click')
    await flushPromises()

    expect(spy.mock.calls.length, 'the retry did not re-fetch').toBeGreaterThan(calls)
    expect(w.text()).toContain('Fresh Again')
    expect(
      w.find('[data-testid="stale-notice"]').exists(),
      'the notice stayed up after a successful refresh',
    ).toBe(false)
  })

  it("a What's-new row is one player link and reports no ranking click (operator 2026-10-07)", async () => {
    // What's new is no longer the ranked /discover feed, so a click on it is not a ranking
    // impression to score — reporting one would credit a feed the listener never saw.
    readCached.mockImplementation(async (k: string) =>
      k === 'home.whatsnew.v2' ? { items: [ep('feat-0', 'Featured'), ep('row-one', 'Row One')], scope: 'all' } : null,
    )
    vi.spyOn(api, 'getWhatsNew').mockRejectedValue(new Error('offline'))
    const rec = vi.spyOn(api, 'recordDiscoverClick').mockReturnValue(undefined)
    const w = mountKeptAlive()
    await flushPromises()

    const rowUl = w.find('ul.max-w-3xl') // the What's-new rows list (featured sits in a separate div)
    const rowLinks = rowUl.findAll('a')
    const epLink = rowLinks.find((a) => a.attributes('href')?.includes('row-one'))
    expect(epLink, 'the row should carry an episode link').toBeTruthy()
    expect(rowLinks.length, 'the row is a single link (no away-navigating show link)').toBe(1)
    await epLink!.trigger('click')
    expect(rec).not.toHaveBeenCalled()
  })

  it("labels What's new by whose it is, and Recommended by why (operator 2026-10-07)", async () => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({ items: [ep('a-1', 'Mine')], scope: 'yours' })
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.text()).toContain('New in your shows and topics')
    expect(w.text()).not.toContain('New across all shows')
  })

  it("says so when What's new falls back to every show", async () => {
    vi.spyOn(api, 'getWhatsNew').mockResolvedValue({ items: [ep('a-1', 'Anyone')], scope: 'all' })
    signIn()
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.text()).toContain('New across all shows')
  })

  it('no notice when everything is fresh', async () => {
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.find('[data-testid="stale-notice"]').exists()).toBe(false)
  })

})

/**
 * "Continue listening" was built from `GET /playback` — the SERVER's list — so with no network it
 * vanished, on a device that had written every one of those positions itself and, for a downloaded
 * episode, holds the title and show too.
 */
describe('Continue listening falls back to the device (#1909)', () => {
  const POS = { slug: 'dl-1', seconds: 120, finished: false, updatedAt: 2 }

  function markDownloaded(over: Record<string, unknown> = {}) {
    useDownloadsStore().entries['dl-1'] = {
      slug: 'dl-1',
      state: 'downloaded',
      updatedAt: 1,
      title: 'Half Finished',
      showTitle: 'The Drift',
      feedId: 'p06',
      durationSeconds: 600,
      ...over,
    } as never
  }

  beforeEach(() => {
    allPositions.mockReset().mockReturnValue([])
    localKnowledgeFor.mockReset().mockResolvedValue(null)
    localArtworkFor.mockReset().mockReturnValue(null)
    readCached.mockReset().mockResolvedValue(null)
  })

  it('rebuilds the rail from device positions when the server cannot answer', async () => {
    allPositions.mockReturnValue([POS])
    markDownloaded()
    signIn()
    vi.spyOn(api, 'getPlaybackList').mockRejectedValue(new Error('offline'))
    const w = mountKeptAlive()
    await flushPromises()
    await flushPromises()
    await flushPromises()
    await flushPromises()
    expect(w.text(), 'the rail did not rebuild from the device').toContain('Half Finished')
  })

  it('prefers the stored server detail over the registry stub', async () => {
    // The sidecar holds the summary and the real publish date; the registry holds three fields.
    allPositions.mockReturnValue([POS])
    markDownloaded()
    localKnowledgeFor.mockResolvedValue({
      detail: { slug: 'dl-1', title: 'Title From The Sidecar', podcast_title: 'The Drift' },
      insights: [],
      topics: [],
      persons: [],
    })
    signIn()
    vi.spyOn(api, 'getPlaybackList').mockRejectedValue(new Error('offline'))
    const w = mountKeptAlive()
    await flushPromises()
    await flushPromises()
    await flushPromises()
    await flushPromises()
    expect(w.text()).toContain('Title From The Sidecar')
  })

  it('lists only episodes it can DESCRIBE — a bare slug is worse than no row', async () => {
    // A MIXED list, deliberately: asserting only the absence of the undescribable one passes just
    // as well when the whole rail has failed, which is how this test first passed for the wrong
    // reason. The describable one must be present in the same breath.
    allPositions.mockReturnValue([
      { slug: 'never-downloaded', seconds: 90, finished: false, updatedAt: 3 },
      POS,
    ])
    markDownloaded()
    signIn()
    vi.spyOn(api, 'getPlaybackList').mockRejectedValue(new Error('offline'))
    const w = mountKeptAlive()
    await flushPromises()
    await flushPromises()
    await flushPromises()
    await flushPromises()
    expect(w.text(), 'the describable episode is missing too — the rail just failed').toContain(
      'Half Finished',
    )
    expect(w.text()).not.toContain('never-downloaded')
  })

  it('skips finished episodes and ones barely started', async () => {
    allPositions.mockReturnValue([
      { slug: 'dl-1', seconds: 500, finished: true, updatedAt: 3 },
      { slug: 'dl-1', seconds: 0.5, finished: false, updatedAt: 2 },
    ])
    markDownloaded()
    signIn()
    vi.spyOn(api, 'getPlaybackList').mockRejectedValue(new Error('offline'))
    const w = mountKeptAlive()
    await flushPromises()
    await flushPromises()
    await flushPromises()
    await flushPromises()
    expect(w.text()).not.toContain('Half Finished')
  })

  it('does not touch the device when the server answers', async () => {
    allPositions.mockReturnValue([POS])
    markDownloaded()
    signIn()
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    const w = mountKeptAlive()
    await flushPromises()
    expect(allPositions, 'read the device for a request that succeeded').not.toHaveBeenCalled()
    expect(w.text()).not.toContain('Half Finished')
  })
})

/**
 * What `onActivated` owns and nothing covered (#2024). These are only reachable because the file
 * mounts under KeepAlive now.
 */
describe('returning to Home (#2024)', () => {
  const POS = [{ slug: 'ep-1', position_seconds: 120, finished: false, updated_at: 2 }]

  beforeEach(() => {
    allPositions.mockReset().mockReturnValue([])
    localKnowledgeFor.mockReset().mockResolvedValue(null)
    readCached.mockReset().mockResolvedValue(null)
  })

  it('refreshes continue-listening on the way back in', async () => {
    signIn()
    const spy = vi.spyOn(api, 'getPlaybackList').mockResolvedValue(POS as never)
    const { leave, comeBack } = mountReturnable()
    await flushPromises()
    const first = spy.mock.calls.length
    expect(first, 'the rail never loaded on the first activation').toBeGreaterThan(0)

    await leave()
    await comeBack()
    expect(spy.mock.calls.length, 'coming back did not refresh the resume hero').toBeGreaterThan(
      first,
    )
  })

  it('a failed refresh keeps the resume hero rather than blanking it', async () => {
    // The reason the quiet refresh exists: returning to Home must not drop you back to the
    // discover hero because one round-trip fell over.
    signIn()
    const spy = vi.spyOn(api, 'getPlaybackList').mockResolvedValue(POS as never)
    vi.spyOn(api, 'getEpisode').mockResolvedValue(ep('ep-1', 'Half Finished') as never)
    const { wrapper, leave, comeBack } = mountReturnable()
    await flushPromises()
    expect(wrapper.text()).toContain('Half Finished')

    spy.mockRejectedValue(new Error('offline'))
    await leave()
    await comeBack()
    expect(
      wrapper.text(),
      'a dropped refresh blanked/swapped the resume hero',
    ).toContain('Half Finished')
  })

  /**
   * NOT distinguishable from a full `loadContinue()` any more, and that is worth knowing.
   * `refreshContinueQuietly` exists to avoid the skeleton flash a full reload used to cause — but
   * `useSectionState` now revalidates in place and only drops to `loading` when it has nothing, so
   * both paths keep the hero. Forcing the full load does not turn this red. The quiet path still
   * earns its keep (it skips the cache read), just not by this property.
   */
  it('a failed refresh now MARKS the rail stale, so the notice and retry appear', async () => {
    // The quiet refresh used to write `data` directly, bypassing the section — so a dropped
    // refresh left the rail looking current: no stale flag, no notice, no retry. Folding it into
    // `loadContinue` is what makes the failure visible.
    signIn()
    const spy = vi.spyOn(api, 'getPlaybackList').mockResolvedValue(POS as never)
    vi.spyOn(api, 'getEpisode').mockResolvedValue(ep('ep-1', 'Half Finished') as never)
    const { wrapper, leave, comeBack } = mountReturnable()
    await flushPromises()
    expect(wrapper.find('[data-testid="stale-notice"]').exists()).toBe(false)

    spy.mockRejectedValue(new Error('offline'))
    await leave()
    await comeBack()

    expect(wrapper.text(), 'the hero was blanked').toContain('Half Finished')
    expect(
      wrapper.find('[data-testid="stale-notice"]').exists(),
      'a failed refresh left the rail looking current',
    ).toBe(true)
  })

  it('does not flash a skeleton over a hero it already has', async () => {
    // Observed DURING the refresh, with the request held open — after it settles a skeleton would
    // be gone either way, so asserting at the end proves nothing. This is the operator's original
    // complaint: the reload glitch on returning to Home.
    signIn()
    const spy = vi.spyOn(api, 'getPlaybackList').mockResolvedValue(POS as never)
    vi.spyOn(api, 'getEpisode').mockResolvedValue(ep('ep-1', 'Half Finished') as never)
    const { wrapper, leave, comeBack } = mountReturnable()
    await flushPromises()

    let release!: (v: unknown) => void
    spy.mockReturnValue(new Promise((r) => (release = r as (v: unknown) => void)) as never)
    await leave()
    const back = comeBack()
    await flushPromises()

    // Asserting the HERO still holds its content, not the absence of any skeleton anywhere: other
    // sections legitimately reload on activation, and a blanket query catches them too — which is
    // how the first version of this failed for the wrong reason.
    expect(
      wrapper.text(),
      'returning to Home blanked the resume hero while refreshing it',
    ).toContain('Half Finished')
    release(POS)
    await back
  })
})
