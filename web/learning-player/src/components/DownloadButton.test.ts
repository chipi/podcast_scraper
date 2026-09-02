import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import * as deviceStore from '../services/deviceStore'
import { useAuthStore } from '../stores/auth'
import { useDownloadsStore } from '../stores/downloads'

const isNative = vi.fn(() => true)
const markForOffline = vi.fn()
const deleteEpisode = vi.fn()
const getNetworkPolicy = vi.fn(async () => 'wifi-only')

vi.mock('../services/native', () => ({ isNative: () => isNative() }))
vi.mock('../services/downloadScheduler', () => ({
  markForOffline: (s: string) => markForOffline(s),
  getNetworkPolicy: () => getNetworkPolicy(),
}))
vi.mock('../services/downloads', () => ({ deleteEpisode: (s: string) => deleteEpisode(s) }))

const DownloadButton = (await import('./DownloadButton.vue')).default

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountBtn = () =>
  mount(DownloadButton, { props: { slug: 'ep-1' }, global: { plugins: [i18n] } })

beforeEach(() => {
  setActivePinia(createPinia())
  // Downloading requires a session (#1912); these assert the signed-IN behaviour.
  useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
  isNative.mockReturnValue(true)
  markForOffline.mockResolvedValue(true)
  getNetworkPolicy.mockResolvedValue('wifi-only')
  deleteEpisode.mockResolvedValue(undefined)
  vi.spyOn(deviceStore, 'getDeviceJson').mockResolvedValue(null)
  vi.spyOn(deviceStore, 'setDeviceJson').mockResolvedValue()
})
afterEach(() => {
  vi.restoreAllMocks()
  // restoreAllMocks only undoes spyOn; the module-level vi.fn()s keep their call history,
  // which silently leaked calls between tests.
  vi.clearAllMocks()
})

describe('DownloadButton', () => {
  it('is hidden on the web, where there is no offline audio story', () => {
    isNative.mockReturnValue(false)
    expect(mountBtn().find('button').exists()).toBe(false)
  })

  it('offers a download when the episode is not on the device', () => {
    const btn = mountBtn().find('button')
    expect(btn.attributes('data-state')).toBe('none')
    expect(btn.attributes('aria-label')).toBe(en.downloads.download)
  })

  it('flags the episode when tapped', async () => {
    await mountBtn().find('button').trigger('click')
    await flushPromises()
    expect(markForOffline).toHaveBeenCalledWith('ep-1')
  })

  it('shows "Waiting for Wi-Fi" as its own state, not as an idle button', async () => {
    // Under L1 a flagged episode legitimately sits idle; without a distinct affordance that is
    // indistinguishable from a broken control.
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.mark('ep-1')
    const btn = mountBtn().find('button')
    expect(btn.attributes('data-state')).toBe('queued')
    expect(btn.attributes('aria-label')).toBe(en.downloads.waitingWifi)
  })

  it('reports progress, and stays indeterminate when the host sends no length', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    store.setDownloading('ep-1')
    store.setProgress('ep-1', 0.42)
    let btn = mountBtn().find('button')
    expect(btn.attributes('aria-label')).toBe('Downloading 42%')
    expect(btn.attributes('aria-busy')).toBe('true')

    // contentLength 0 leaves progress at 0 forever — "0%" would read as stuck.
    store.setProgress('ep-1', 0)
    btn = mountBtn().find('button')
    expect(btn.attributes('aria-label')).toBe(en.downloads.downloading)
  })

  it('removes the download when tapped in the downloaded state', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('ep-1', 'file:///a.mp3', 1)
    const btn = mountBtn().find('button')
    expect(btn.attributes('data-state')).toBe('downloaded')
    await btn.trigger('click')
    await flushPromises()
    expect(deleteEpisode).toHaveBeenCalledWith('ep-1')
    expect(markForOffline).not.toHaveBeenCalled()
  })

  it('cancels an in-flight transfer when tapped', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    store.setDownloading('ep-1')
    await mountBtn().find('button').trigger('click')
    await flushPromises()
    expect(deleteEpisode).toHaveBeenCalledWith('ep-1')
  })

  it('offers a retry after a transient failure', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setFailed('ep-1', 'socket reset', 'retryable')
    const btn = mountBtn().find('button')
    expect(btn.attributes('aria-label')).toBe(en.downloads.retry)
    await btn.trigger('click')
    await flushPromises()
    expect(markForOffline).toHaveBeenCalledWith('ep-1')
  })

  it('offers removal, not a retry, for an episode that is permanently gone', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setFailed('ep-1', 'not found', 'permanent')
    const btn = mountBtn().find('button')
    expect(btn.attributes('aria-label')).toBe(en.downloads.unavailable)
    await btn.trigger('click')
    await flushPromises()
    // A retry that can only fail again is not an affordance, it is a lie — but the row still has
    // to be dismissable, or it sits in the Downloaded list forever with no way out.
    expect(markForOffline).not.toHaveBeenCalled()
    expect(deleteEpisode).toHaveBeenCalledWith('ep-1')
  })

  it('says "waiting for a connection" when the user has allowed cellular', async () => {
    // "Waiting for Wi-Fi" is a lie for someone who opted into cellular — they are just offline.
    getNetworkPolicy.mockResolvedValue('any')
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.mark('ep-1')
    const w = mountBtn()
    await flushPromises()
    expect(w.find('button').attributes('aria-label')).toBe(en.downloads.waitingConnection)
  })

  it('signed out, it is a sign-in teaser rather than a silent failure (#1912)', async () => {
    setActivePinia(createPinia())
    const w = mountBtn()
    await flushPromises()
    const btn = w.find('button')
    // Rendered, not hidden — the capability stays visible, like the queue control.
    expect(btn.exists()).toBe(true)
    expect(btn.attributes('aria-label')).toBe(en.auth.signInToDownload)

    await btn.trigger('click')
    await flushPromises()
    // The episode routes become auth-gated in #1063/#1066; a download attempted signed out would
    // 401 and leave a `failed` row the user cannot explain. And an `anon` download vanishes on
    // sign-in, which reads as data loss.
    expect(markForOffline).not.toHaveBeenCalled()
  })
})
