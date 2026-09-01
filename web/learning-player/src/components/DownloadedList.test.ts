import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import * as deviceStore from '../services/deviceStore'
import { useDownloadsStore } from '../stores/downloads'

const isNative = vi.fn(() => true)
vi.mock('../services/native', () => ({ isNative: () => isNative() }))
vi.mock('../services/downloadScheduler', () => ({ markForOffline: vi.fn() }))
vi.mock('../services/downloads', () => ({
  deleteEpisode: vi.fn(),
  localArtworkFor: () => null,
}))

const DownloadedList = (await import('./DownloadedList.vue')).default
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountList = () =>
  mount(DownloadedList, {
    global: {
      plugins: [i18n],
      // `RouterLink: true` swallows the slot, so every row rendered empty.
      stubs: { RouterLink: { template: '<a><slot /></a>' } },
    },
  })

beforeEach(() => {
  setActivePinia(createPinia())
  isNative.mockReturnValue(true)
  vi.spyOn(deviceStore, 'getDeviceJson').mockResolvedValue(null)
  vi.spyOn(deviceStore, 'setDeviceJson').mockResolvedValue()
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('DownloadedList', () => {
  it('does not exist on the web', () => {
    isNative.mockReturnValue(false)
    expect(mountList().find('[data-testid="downloaded-section"]').exists()).toBe(false)
  })

  it('does not render at all until something is downloaded', () => {
    // An empty block on every visit to Saved is clutter for the majority who have none; the
    // control that creates the first download lives on the episode cards.
    const w = mountList()
    expect(w.find('[data-testid="downloaded-section"]').exists()).toBe(false)
  })

  it('renders from the registry alone, with no API call', async () => {
    // This is the one list that must be right with no network, so it must not need one.
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('ep-1', 'file:///a.mp3', 5 * 1024 * 1024)
    store.setMetadata('ep-1', {
      title: 'Index Investing Without the Myths',
      showTitle: 'Long Horizon Notes',
      durationSeconds: 416,
    })
    const w = mountList()
    const text = w.find('[data-testid="downloaded-item"]').text()
    expect(text).toContain('Index Investing Without the Myths')
    expect(text).toContain('Long Horizon Notes')
    expect(text).toContain('7 min')
  })

  it('falls back to the slug when metadata was never captured', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('ep-no-meta', 'file:///a.mp3', 1)
    expect(mountList().find('[data-testid="downloaded-item"]').text()).toContain('ep-no-meta')
  })

  it('labels a queued episode so it is not mistaken for a playable one', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.mark('ep-q')
    expect(mountList().find('[data-testid="downloaded-item"]').text()).toContain(
      en.downloads.waitingWifi,
    )
  })

  it('says an episode is gone rather than offering a doomed retry', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setFailed('ep-gone', 'not found', 'permanent')
    expect(mountList().find('[data-testid="downloaded-item"]').text()).toContain(
      en.downloads.unavailable,
    )
  })

  it('orders newest first', async () => {
    const store = useDownloadsStore()
    store.loaded = true
    store.entries = {
      old: { slug: 'old', state: 'downloaded', updatedAt: 1, title: 'Older' },
      recent: { slug: 'recent', state: 'downloaded', updatedAt: 9, title: 'Newer' },
    }
    const items = mountList().findAll('[data-testid="downloaded-item"]')
    expect(items[0].text()).toContain('Newer')
    expect(items[1].text()).toContain('Older')
  })

  it('reports storage used, counting only completed downloads', async () => {
    const store = useDownloadsStore()
    store.loaded = true
    store.entries = {
      a: { slug: 'a', state: 'downloaded', updatedAt: 1, bytes: 10 * 1024 * 1024 },
      b: { slug: 'b', state: 'queued', updatedAt: 2, bytes: 999 },
    }
    const text = mountList().find('[data-testid="downloaded-storage"]').text()
    expect(text).toContain('10 MB')
    expect(text).toContain('1 episodes')
  })
})
