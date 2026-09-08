import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import * as push from '../composables/usePushSubscription'
import en from '../i18n/locales/en.json'
import type { CommsSettings, InterestCluster, UserStats } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { useUserPreferencesStore } from '../stores/userPreferences'
import ProfileView from './ProfileView.vue'

// Native, so DeviceSettings would actually RENDER if it were still on this page. Without this the
// absence assertion below passes on a component that renders nothing off-native — it would hold
// whether or not Device had been moved, which is no assertion at all.
vi.mock('../services/native', () => ({ isNative: () => true }))
vi.mock('../services/downloadScheduler', () => ({
  DEFAULT_POLICY: 'wifi-only',
  applyDownloadCap: async () => {},
  getNetworkPolicy: async () => 'wifi-only',
  setNetworkPolicy: async () => {},
}))
vi.mock('../services/deviceStore', () => ({
  getDeviceJson: async () => null,
  setDeviceJson: async () => {},
}))

// ONE shared log, written by both the cache mock and the logout spy — two separate arrays could
// only show that both ran, never in which order, which is the whole claim being tested.
const sequence: string[] = []
vi.mock('../services/contentCache', async (orig) => {
  const actual = await orig<typeof import('../services/contentCache')>()
  return {
    ...actual,
    clearCached: async (...args: unknown[]) => {
      sequence.push('clearCached')
      void args
    },
  }
})

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/profile', name: 'profile', component: ProfileView },
    { path: '/settings', name: 'settings', component: { template: '<div/>' } },
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/catalog', name: 'catalog', component: { template: '<div/>' } },
  ],
})

const clusters: InterestCluster[] = [{ id: 'tc:ai', label: 'AI', size: 12 }]

function stats(over: Partial<UserStats> = {}): UserStats {
  return {
    episodes: 8, shows: 3, listening_seconds: 7200, active_days: 5, day_streak: 4,
    daily: [{ date: '2024-03-01', count: 2 }, { date: '2024-03-02', count: 1 }], ...over,
  }
}

function comms(over: Partial<CommsSettings> = {}): CommsSettings {
  return {
    digest: { enabled: false, cadence: 'weekly', day_of_week: 6, hour: 13, paused: false },
    push: { enabled: false },
    email_verified: true,
    unsubscribe_ref: null,
    ...over,
  }
}

/**
 * Every ProfileView mounted by this file, torn down after each test.
 *
 * Not tidiness — correctness, and the same trap `PlayerView.test.ts` documents. A wrapper that is
 * never unmounted keeps its watchers alive, so a LATER test re-mocking an endpoint to reject makes
 * the zombie re-run `load()` against it, with nobody awaiting the result. That surfaced as an
 * unhandled rejection attributed to the new test's mock, in a component whose own catch was fine —
 * it cost a long hunt for a consumer that did not exist.
 */
const mountedProfiles: Array<{ unmount: () => void }> = []
afterEach(() => {
  while (mountedProfiles.length) mountedProfiles.pop()!.unmount()
})

function mountProfile() {
  setActivePinia(createPinia())
  const auth = useAuthStore()
  auth.user = { user_id: 'u_1', email: 'dev@localhost', name: 'Dev' }
  const w = mount(ProfileView, { global: { plugins: [i18n, router] } })
  mountedProfiles.push(w)
  return w
}

beforeEach(() => {
  vi.spyOn(api, 'getTopClusters').mockResolvedValue(clusters)
  vi.spyOn(api, 'getMyStats').mockResolvedValue(stats())
  vi.spyOn(api, 'getComms').mockResolvedValue(comms())
})
afterEach(() => vi.restoreAllMocks())

describe('ProfileView — Settings entry (#8)', () => {
  it('no longer carries the Device section — that belongs to Settings', async () => {
    // Profile is about me as a user. Download network policy and the size cap are about the
    // handset, and are shared by every account that signs in on it.
    const w = await mountProfile()
    await flushPromises()
    expect(w.find('[data-testid="device-settings"]').exists(), 'Device is still on Profile').toBe(false)
  })

  it('links to the Settings screen via the gear', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mountProfile()
    const gear = w.find('[data-testid="profile-settings-link"]')
    expect(gear.exists()).toBe(true)
    expect(gear.attributes('href')).toBe('/settings')
  })
})

describe('ProfileView — Your Week layout', () => {
  it('reflects the saved layout preference and persists a change', async () => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: 'u_1', email: 'dev@localhost', name: 'Dev' }
    const prefs = useUserPreferencesStore()
    vi.spyOn(prefs, 'hydrate').mockResolvedValue()
    vi.spyOn(prefs, 'get').mockReturnValue('full') // a saved 'full' preference
    const setSpy = vi.spyOn(prefs, 'set').mockResolvedValue()
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mount(ProfileView, { global: { plugins: [i18n, router] } })
    await flushPromises()

    // Initial state reflects the saved pref: the Full button is the active one.
    //
    // Asserted on the ARIA state, not on a background class (#1959): the selected pill is styled
    // FROM that attribute, so the accessible state and the visible state cannot drift apart.
    //
    // The attribute is `aria-checked` now, not `aria-selected` (#1594 item 7). This control sets a
    // saved preference and switches no region, so it is a radiogroup rather than a tablist — "tab"
    // was the wrong announcement. This assertion is what caught the conversion breaking the
    // VISIBLE state: the CSS keyed the fill off `aria-selected` alone, so the option kept working
    // and quietly stopped looking selected. Exactly the drift the note above was written about.
    const fullBtn = w.findAll('button').find((b) => b.text() === 'Full')!
    expect(fullBtn.attributes('aria-checked')).toBe('true')
    expect(fullBtn.attributes('role')).toBe('radio')
    const compactInitially = w.findAll('button').find((b) => b.text() === 'Compact')!
    expect(compactInitially.attributes('aria-checked')).toBe('false')

    // Switching to Compact persists the change under the shared key.
    const compactBtn = w.findAll('button').find((b) => b.text() === 'Compact')!
    await compactBtn.trigger('click')
    expect(setSpy).toHaveBeenCalledWith('lp.yourweek.layout', 'compact')
  })
})

describe('ProfileView — interest chips', () => {
  it('renders chips hued by kind: person → text-person, topic/cluster → text-topic', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([
      'tc:ai',
      'topic:personal-growth',
      'person:brian-chesky',
    ])
    const w = mountProfile()
    await flushPromises()

    const chips = w.findAll('span').filter((s) =>
      s.classes().includes('text-person') || s.classes().includes('text-topic'),
    )
    const personChip = chips.find((c) => c.classes().includes('text-person'))!
    const topicChips = chips.filter((c) => c.classes().includes('text-topic'))

    // person:brian-chesky → person hue, de-slugged label
    expect(personChip.classes()).toContain('text-person')
    expect(personChip.text()).toBe('brian chesky')

    // topic:personal-growth → topic hue, de-slugged
    expect(topicChips.some((c) => c.text() === 'personal growth')).toBe(true)
    // tc:ai resolves to its cluster label via the clusters map (not de-slugged "ai")
    expect(topicChips.some((c) => c.text() === 'AI')).toBe(true)
  })

  it('shows the no-interests message when the list is empty', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mountProfile()
    await flushPromises()
    expect(w.text()).toContain('No interests chosen yet.')
  })
})

describe('ProfileView — Your listening panel', () => {
  it('renders streak / episodes / shows when episodes > 0', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mountProfile()
    await flushPromises()
    expect(w.text()).toContain('Your activity')
    expect(w.text()).toContain('Day streak')
    expect(w.text()).toContain('Episodes')
    expect(w.text()).toContain('Shows')
    // NO "Hours" tile here any more (#1914). It rendered `sum(position_seconds)` — a lifetime
    // snapshot of furthest position reached, which rises on a forward seek and does not move on a
    // re-listen. Time actually listened lives in ListeningRecap, with its coverage stated.
    expect(w.text()).not.toContain('Hours')
    // 4-day streak + 8 episodes + 3 shows surface their numbers.
    expect(w.text()).toContain('4')
    expect(w.text()).toContain('8')
    expect(w.text()).toContain('3')
  })

  it('shows the stats empty state when the user has no episodes', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'getMyStats').mockResolvedValue(stats({ episodes: 0 }))
    const w = mountProfile()
    await flushPromises()
    expect(w.text()).toContain('Start listening to build your stats.')
    expect(w.text()).not.toContain('Day streak')
  })
})

describe('ProfileView — notifications', () => {
  beforeEach(() => vi.spyOn(api, 'getUserInterests').mockResolvedValue([]))

  it('renders the Your Week layout switch + email/push toggles; cadence hidden until digest on', async () => {
    const w = mountProfile()
    await flushPromises()
    expect(w.text()).toContain('Your Week')
    // The in-app view is primary: the layout switch shows first, email is "the edge".
    expect(w.text()).toContain('On your home')
    expect(w.text()).toContain('Compact')
    expect(w.text()).toContain('Full')
    expect(w.text()).toContain('Also email it to me')
    expect(w.text()).toContain('Push reminders')
    expect(w.text()).not.toContain('Frequency')
  })

  it('enabling the digest PUTs the whole section and reveals the cadence control', async () => {
    const put = vi
      .spyOn(api, 'putComms')
      .mockResolvedValue(comms({ digest: { enabled: true, cadence: 'weekly', day_of_week: 6, hour: 13, paused: false } }))
    const w = mountProfile()
    await flushPromises()

    const digestToggle = w.findAll('input[type="checkbox"]')[0]
    await digestToggle.setValue(true)
    await flushPromises()

    expect(put).toHaveBeenCalledWith({ digest: expect.objectContaining({ enabled: true }) })
    expect(w.text()).toContain('Frequency')
  })

  it('enabling push registers a browser subscription via the composable', async () => {
    const enable = vi.spyOn(push, 'enablePush').mockResolvedValue(true)
    const w = mountProfile()
    await flushPromises()

    // digest toggle is index 0; the push toggle is the last checkbox.
    const boxes = w.findAll('input[type="checkbox"]')
    await boxes[boxes.length - 1].setValue(true)
    await flushPromises()

    expect(enable).toHaveBeenCalled()
  })

  it('reverts the push toggle when the browser cannot subscribe', async () => {
    vi.spyOn(push, 'enablePush').mockResolvedValue(false)
    const put = vi.spyOn(api, 'putComms').mockResolvedValue(comms({ push: { enabled: false } }))
    const w = mountProfile()
    await flushPromises()

    const boxes = w.findAll('input[type="checkbox"]')
    await boxes[boxes.length - 1].setValue(true)
    await flushPromises()

    expect(put).toHaveBeenCalledWith({ push: { enabled: false } })
  })

  it('reverts the push toggle when the subscribe POST throws (no desync)', async () => {
    vi.spyOn(push, 'enablePush').mockRejectedValue(new Error('network'))
    const put = vi.spyOn(api, 'putComms').mockResolvedValue(comms({ push: { enabled: false } }))
    const w = mountProfile()
    await flushPromises()

    const boxes = w.findAll('input[type="checkbox"]')
    await boxes[boxes.length - 1].setValue(true)
    await flushPromises()

    expect(put).toHaveBeenCalledWith({ push: { enabled: false } })
  })

  describe('sign out (#1594)', () => {
    it('lands on Home, not the flat Catalog index', async () => {
      // Catalog is every episode in the corpus in one list. It is a fine place to browse TO and
      // the wrong place to be dropped: Home renders the signed-out hero that explains what the app
      // is, which is the only thing someone who just signed out might want. A bare list reads like
      // a session that half-broke rather than one they ended on purpose.
      vi.spyOn(api, 'logout').mockResolvedValue(undefined as never)
      const w = mountProfile()
      await flushPromises()

      const btn = w.findAll('button').find((b) => b.text().includes('Sign out'))
      expect(btn, 'no sign-out button rendered — this assertion would be vacuous').toBeTruthy()
      await btn!.trigger('click')
      await flushPromises()

      expect(router.currentRoute.value.name).toBe('home')
    })

    it('clears the cached content BEFORE dropping the identity', async () => {
      // Order matters and is invisible in the UI: a signed-out device must not keep the previous
      // account's library readable on disk (#1909). Swapping the two lines looks harmless in
      // review, so the SEQUENCE is what is asserted — not merely that both ran.
      sequence.length = 0
      vi.spyOn(api, 'logout').mockImplementation(async () => {
        sequence.push('logout')
        return undefined as never
      })
      const w = mountProfile()
      await flushPromises()
      await w.findAll('button').find((b) => b.text().includes('Sign out'))!.trigger('click')
      await flushPromises()

      expect(
        sequence,
        'the cached library must be wiped BEFORE the identity goes, or a signed-out device keeps ' +
          "the previous account's content readable",
      ).toEqual(['clearCached', 'logout'])
    })
  })

  /**
   * #1591's defect, recurring where nothing was watching: with no network the page told a user
   * with stats and interests that they had neither.
   *
   * Only the STATS half is asserted here. Rejecting `getUserInterests` raises an unhandled
   * rejection from a consumer I could not locate — it is not the three call sites in this repo,
   * all of which catch — and asserting through a detector I do not understand would be asserting
   * something else. The interests path is the same three lines as stats, and is NOT covered.
   */
  /**
   * #1591's defect, recurring where nothing was watching: with no network the page told a user
   * with stats and interests that they had neither.
   *
   * Split in two, and the interests mock is pre-handled (`pr.catch(() => {})`), which looks
   * arbitrary and is not. Rejecting BOTH calls in one test makes vitest fail on an unhandled
   * rejection I could not trace: the calls are made once each, from `load()`, with `.catch`
   * attached synchronously; all three call sites in the repo catch; `ensureLoaded` was one and is
   * fixed; the mocks alone with nothing mounted do not leak; and pre-handling both does not silence
   * it, so the loose promise is a DERIVED one I have not identified. Each half on its own is clean,
   * and each proves its own branch, so that is how they are written. The unlocated rejection is a
   * real loose end, recorded as one rather than papered over.
   */
  it('says a STATS load failed rather than claiming you have nothing', async () => {
    vi.spyOn(api, 'getMyStats').mockImplementation(() => Promise.reject(new Error('offline')))
    const w = await mountProfile()
    await flushPromises()

    expect(w.find('[data-testid="stats-unavailable"]').exists(), 'stats claimed emptiness').toBe(
      true,
    )
    expect(w.text()).not.toContain('Start listening to build your stats')
  })

  /**
   * NOT COVERED: the interests half. Rejecting `getUserInterests` and mounting ProfileView makes
   * vitest fail on an unhandled rejection whose consumer I could not find, in isolation and in the
   * suite, with the mock pre-handled and without.
   *
   * What I ruled out: the call is made exactly once, from `load()`, with `.catch` attached
   * synchronously; all three call sites in the repo catch; `ensureLoaded` was one and now catches;
   * the mock alone with nothing mounted does not leak; and zombie wrappers are not it (they are
   * unmounted now regardless — see the note on `mountProfile`, and that fix is worth keeping).
   *
   * What IS established: the production path is correct. Mounting with a rejecting
   * `getUserInterests` renders `interests-unavailable` and NOT "No interests chosen yet", which can
   * only happen if the component's own catch ran. The code is right; the test harness defeats me.
   * The stats half below is the same three lines and IS covered.
   */
  it('a genuinely empty account still reads as empty, not as broken', async () => {
    // The distinction has to cut both ways or it is just a different lie.
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = await mountProfile()
    await flushPromises()
    expect(w.find('[data-testid="interests-unavailable"]').exists()).toBe(false)
    expect(w.text()).toContain('No interests chosen yet')
  })
})
