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

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/profile', name: 'profile', component: ProfileView },
    { path: '/settings', name: 'settings', component: { template: '<div/>' } },
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

function mountProfile() {
  setActivePinia(createPinia())
  const auth = useAuthStore()
  auth.user = { user_id: 'u_1', email: 'dev@localhost', name: 'Dev' }
  return mount(ProfileView, { global: { plugins: [i18n, router] } })
}

beforeEach(() => {
  vi.spyOn(api, 'getTopClusters').mockResolvedValue(clusters)
  vi.spyOn(api, 'getMyStats').mockResolvedValue(stats())
  vi.spyOn(api, 'getComms').mockResolvedValue(comms())
})
afterEach(() => vi.restoreAllMocks())

describe('ProfileView — Settings entry (#8)', () => {
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
    const fullBtn = w.findAll('button').find((b) => b.text() === 'Full')!
    expect(fullBtn.classes()).toContain('bg-accent')

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
  it('renders streak / episodes / shows / hours when episodes > 0', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mountProfile()
    await flushPromises()
    expect(w.text()).toContain('Your listening')
    expect(w.text()).toContain('Day streak')
    expect(w.text()).toContain('Episodes')
    expect(w.text()).toContain('Shows')
    expect(w.text()).toContain('Hours')
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
})
