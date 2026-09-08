import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary, Me, Podcast } from '../services/types'
import { resetStaleness } from '../composables/useSectionState'
import { useDownloadsStore } from '../stores/downloads'
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
  readCached: (k: string) => readCached(k),
  writeCached: async () => {},
}))
import homeViewSource from './HomeView.vue?raw'
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
    theme_clusters: [],
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
 * Home inside a `<KeepAlive>`, which is how `App.vue` renders it.
 *
 * The continue-listening rail loads in `onActivated`, and Vue only fires that for a component
 * under KeepAlive — so a plain `mount(HomeView)` leaves that section in `loading` forever and the
 * rail cannot be tested at all. Every test above mounts plainly, which is why none of them
 * exercises this path.
 */
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
    vi.spyOn(api, 'getDiscover').mockResolvedValue({
      items: [ep('a-1', 'First Ep'), ep('a-2', 'Second Ep')], page: 1, page_size: 8, total: 2, has_more: false,
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      { feed_id: 'showa', title: 'Show A', artwork_url: null, image_url: null, episode_count: 2 } as Podcast,
    ])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([]) // no history → discover state

    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain("Find any moment you've heard.") // discover hero
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
    vi.spyOn(api, 'getDiscover').mockResolvedValue({
      items: [ep('a-1', 'First Ep'), row], page: 1, page_size: 8, total: 2, has_more: false,
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])

    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    // The ranked row (index 1) now carries a square thumbnail from its artwork.
    expect(w.find('img[src="https://x/row.png"]').exists()).toBe(true)
  })

  it('folds Rising/Trending/Storylines into one tabbed area, Rising default (#4)', async () => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({
      items: [ep('a-1', 'First Ep')], page: 1, page_size: 8, total: 1, has_more: false,
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])

    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const tabs = w.find('[data-testid="home-discovery"]')
    expect(tabs.exists()).toBe(true)
    for (const key of ['rising', 'trending', 'storylines']) {
      expect(w.find(`[data-testid="discovery-tab-${key}"]`).exists()).toBe(true)
    }
    // Rising is selected by default; switching updates aria-selected.
    expect(w.get('[data-testid="discovery-tab-rising"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="discovery-tab-storylines"]').attributes('aria-selected')).toBe('false')
    await w.get('[data-testid="discovery-tab-storylines"]').trigger('click')
    expect(w.get('[data-testid="discovery-tab-storylines"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="discovery-tab-rising"]').attributes('aria-selected')).toBe('false')
  })

  it('submitting the search navigates to /search', async () => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({ items: [], page: 1, page_size: 8, total: 0, has_more: false })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    const push = vi.spyOn(router, 'push')
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
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
    vi.spyOn(api, 'getDiscover').mockRejectedValue(new Error('boom'))
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(true)
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
    // The section header still renders — it is what tells you this content exists at all.
    expect(w.text()).toContain("What's new")
  })

  it('retry re-fetches and recovers', async () => {
    // No error state anywhere in the app previously offered a retry: the only move was a reload.
    const spy = vi.spyOn(api, 'getDiscover').mockRejectedValueOnce(new Error('boom'))
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.find('[data-testid="section-error"]').exists()).toBe(true)

    spy.mockResolvedValueOnce({
      items: [ep('a-1', 'Recovered Ep')], page: 1, page_size: 8, total: 1, has_more: false,
    })
    await w.get('[data-testid="section-retry"]').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
    expect(w.text()).toContain('Recovered Ep')
  })

  it('a successful-but-empty load still hides the section', async () => {
    // Hide when the SYSTEM is empty — there is no action the user can take, so an empty shell is
    // noise. Contrast "Your shows", which is empty because of a user action not yet taken.
    vi.spyOn(api, 'getDiscover').mockResolvedValue({
      items: [], page: 1, page_size: 8, total: 0, has_more: false,
    })
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
    expect(w.text()).not.toContain("What's new")
  })
})

describe('HomeView "Your shows" is your follows, not the catalogue (#1585)', () => {
  const catalogue = [
    { feed_id: 'showa', title: 'Show A', artwork_url: null, image_url: null, episode_count: 2 } as Podcast,
    { feed_id: 'showb', title: 'Show B', artwork_url: null, image_url: null, episode_count: 5 } as Podcast,
  ]

  beforeEach(() => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({ items: [], page: 1, page_size: 8, total: 0, has_more: false })
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    vi.spyOn(api, 'getPodcasts').mockResolvedValue(catalogue)
  })

  it('renders only the followed shows, joined to catalogue artwork', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([
      { feed_id: 'showb', feed_url: null, title: 'Show B', added_at: 1 },
    ])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Your shows')
    expect(w.text()).toContain('Show B')
    // Show A is in the corpus but NOT followed. Before #1585 this section rendered the whole
    // catalogue while calling itself "Your shows".
    expect(w.text()).not.toContain('Show A')
  })

  it('offers the action, not just a description of it, when you follow nothing', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    // A section that silently self-hides can't tell a new user the feature exists — but an empty
    // state that only *describes* following is barely better, since it sends you off to a show page
    // to find the control. Suggested shows carry the follow control itself.
    expect(w.text()).toContain('Your shows')
    expect(w.text()).toContain('Follow a show')
    expect(w.findAll('[aria-pressed]').length).toBeGreaterThan(0)
  })

  it('following from the empty state moves the show into the grid, in place', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
    vi.spyOn(api, 'followShow').mockResolvedValue([
      { feed_id: 'showa', feed_url: null, title: 'Show A', added_at: 1 },
    ])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Follow a show') // empty state

    await w.get('[aria-pressed]').trigger('click')
    await flushPromises()

    // The whole point of putting the control here: no navigation, no reload.
    expect(w.text()).not.toContain('Follow a show')
    expect(w.text()).toContain('Show A')
  })

  it('still renders a followed feed that is absent from the catalogue', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([
      { feed_id: 'gone', feed_url: null, title: 'Departed Show', added_at: 1 },
    ])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Departed Show')
  })
})

describe('HomeView interests card (3.5)', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({ items: [], page: 1, page_size: 8, total: 0, has_more: false })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
  })

  it('is hidden when signed out', async () => {
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
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
    expect(w.text()).toContain('AI') // a cluster chip in the picker
  })

  it('dismissing hides the card', async () => {
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    await w.findAll('button').find((b) => b.text() === 'Not now')!.trigger('click')
    expect(w.text()).not.toContain('Personalize your Home')
  })

  it('is hidden when the user already has interests (regression)', async () => {
    // The bug: the card showed even to users with a full interest set. It must only offer to pick
    // interests when there are none.
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['tc:ai', 'tc:science'])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).not.toContain('Personalize your Home')
  })

  // Browse-nav strip opens the Browse hub on the matching tab (not the standalone pages) so the
  // hub's tab bar reflects where you are.
  it('renders "Browse topics" and "Browse people" links into the Browse hub tabs', async () => {
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const nav = w.get('[data-testid="home-browse-nav"]')
    const links = nav.findAll('a')
    const hrefs = links.map((a) => a.attributes('href'))
    expect(hrefs).toContain('/browse?tab=topics')
    expect(hrefs).toContain('/browse?tab=people')
    expect(nav.text()).toContain('Browse topics')
    expect(nav.text()).toContain('Browse people')
  })

  it('resolves trending-show artwork from the catalogue, not from your follows (#1585 regression)', async () => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({
      items: [], page: 1, page_size: 8, total: 0, has_more: false,
    })
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    // #1585 repurposed `shows` from "the whole catalogue" to "shows you follow" and left the
    // trending rail reading it. Trending shows are mostly ones you DON'T follow, so their artwork
    // silently fell back to a generated gradient — a valid render, so no test noticed.
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      { feed_id: 'p01', title: 'Acquired', artwork_url: 'https://x/art.png', image_url: null, description: null, episode_count: 3 },
    ])
    // The rail joins artwork by entity_id → feed_id against the catalogue it is handed.
    vi.spyOn(api, 'getTrending').mockResolvedValue([
      {
        entity_id: 'p01',
        kind: 'show',
        label: 'Acquired',
        velocity: 2,
        volume: 5,
        heating_up: true,
        total: 5,
        series: [1, 2, 3],
      },
    ])

    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()

    // Signed out, with zero follows: the art must still resolve.
    expect(w.html()).toContain('https://x/art.png')
  })

  // --- an outage must not look like a new account (#1591, S7) ---
  //
  // These were the last two sections on `.catch(() => [])`, and the two most personal on the page.
  // #1591 fixed the sections around them and missed these.

  it('a library outage says so instead of claiming you follow nothing', async () => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({
      items: [], page: 1, page_size: 8, total: 0, has_more: false,
    })
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    // The library itself succeeds and is EMPTY — so the only thing standing between the user and
    // the "follow something" prompt is the catalogue fetch. Without this the test passes on an
    // unmocked getLibrary rejecting, which is not the failure being described.
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
    vi.spyOn(api, 'getPodcasts').mockRejectedValue(new Error('502'))
    signIn()

    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()

    // The "follow something to get started" prompt would be a lie to someone with 30 follows.
    expect(w.text()).not.toContain('Follow a show')
    expect(w.find('[data-testid="section-error"]').exists()).toBe(true)
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
  })

  it('a playback outage does not silently swap the resume hero for the discover hero', async () => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({
      items: [], page: 1, page_size: 8, total: 0, has_more: false,
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockRejectedValue(new Error('502'))
    signIn()

    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()

    // Telling a user mid-episode to go explore is how their place looks lost.
    expect(w.text()).not.toContain("Find any moment you've heard.")
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
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    for (const id of ['home-search-input', 'home-search-submit']) {
      const el = w.find(`[data-testid="${id}"]`)
      expect(el.exists(), `${id} should render`).toBe(true)
      expect(el.classes(), `${id} should declare the shared height`).toContain('h-11')
    }
  })

  it('sizes them by height, not by vertical padding', async () => {
    // The regression to prevent: someone re-adds `py-*` and the controls drift apart again.
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
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
})

describe('cards align by the tile, not by cutting text (#2004 items 3/3b)', () => {
  it('the Recommended grid clips neither the title nor the show name', async () => {
    // The clamped title actually overflowed INTO the show name here — an ellipsis at line two AND a
    // visible third line, because the clamp computed but the overflow still painted.
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(homeViewSource).not.toMatch(/line-clamp-2 min-h-\[2\.5rem\]/)
    expect(homeViewSource).not.toMatch(/lp-kicker mt-0\.5 truncate/)
    expect(w.exists()).toBe(true)
  })

  it('keeps the grid even by filling the cell instead', () => {
    // #1584's requirement still holds — it is now paid for by the layout. Removing either class
    // reopens ragged rows, so both are pinned.
    expect(homeViewSource).toMatch(/name: 'player'[\s\S]{0,120}?flex h-full flex-col/)
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
      k === 'home.whatsnew' ? [ep('a-1', 'From Last Time')] : null,
    )
    vi.spyOn(api, 'getDiscover').mockRejectedValue(new Error('offline'))
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
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
      k === 'home.whatsnew' ? [ep('a-1', 'From Last Time')] : null,
    )
    const spy = vi.spyOn(api, 'getDiscover').mockRejectedValue(new Error('offline'))
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const calls = spy.mock.calls.length

    spy.mockResolvedValue({
      items: [ep('a-2', 'Fresh Again')],
      page: 1,
      page_size: 8,
      total: 1,
      has_more: false,
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

  it('no notice when everything is fresh', async () => {
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
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
