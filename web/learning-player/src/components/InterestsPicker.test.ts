import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as analytics from '../services/analytics'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useInterestsStore } from '../stores/interests'
import InterestsPicker from './InterestsPicker.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
// Stub <Teleport> so the modal renders inline in the wrapper (it teleports to <body> in the app).
const mountPicker = () =>
  mount(InterestsPicker, {
    props: { trigger: 'home_prompt' as const },
    global: { plugins: [i18n], stubs: { teleport: true } },
  })

// Pinia: the picker now writes the saved set into the interests store itself (iOS-F1). Before that
// it PUT the list and told nobody, leaving each parent to update its own copy — which is how Home
// kept prompting "Personalize your Home" after interests were chosen from Profile.
beforeEach(() => setActivePinia(createPinia()))
// The sections fetch their own suggestions; give them one of each kind the tests tap.
beforeEach(() => {
  vi.spyOn(api, 'getTrending').mockImplementation(async (kind: string) =>
    kind === 'topic'
      ? [{ entity_id: 'topic:sleep', kind, label: 'Sleep', velocity: 1, volume: 1, heating_up: false, total: 1, series: [] }]
      : []
  )
  vi.spyOn(api, 'getTopClusters').mockResolvedValue([{ id: 'tc:ai', label: 'AI', size: 5 }])
  vi.spyOn(api, 'getStorylines').mockResolvedValue([])
})
afterEach(() => vi.restoreAllMocks())

const button = (w: ReturnType<typeof mountPicker>, text: string) =>
  w.findAll('button').find((b) => b.text() === text)!

describe('InterestsPicker', () => {
  it('is the same four sections as the Profile Interests tab', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mountPicker()
    await flushPromises()
    for (const kind of ['topic', 'person', 'theme', 'storyline']) {
      expect(w.find(`[data-testid="interests-section-${kind}"]`).exists(), kind).toBe(true)
    }
  })

  it('starts from what is followed, and Save writes the whole new set at once', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['person:jane'])
    const put = vi.spyOn(api, 'putUserInterests').mockResolvedValue(['person:jane', 'topic:sleep'])
    const w = mountPicker()
    await flushPromises()
    expect(w.findAll('[data-testid="interest-following-person"]')).toHaveLength(1)

    await w.get('[data-testid="interest-add-topic"]').trigger('click')
    await flushPromises()
    await button(w, '+ Sleep').trigger('click')
    // Nothing written until Save — Cancel has to mean nothing changed.
    expect(put).not.toHaveBeenCalled()
    await button(w, 'Save').trigger('click')
    await flushPromises()
    expect(put).toHaveBeenCalledWith(['person:jane', 'topic:sleep'])
    expect(w.emitted('saved')?.[0]).toEqual([['person:jane', 'topic:sleep']])
    expect(w.emitted('close')).toBeTruthy()
  })

  it('removing a follow in the sheet removes it from the saved set', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['person:jane', 'tc:ai'])
    const put = vi.spyOn(api, 'putUserInterests').mockResolvedValue(['person:jane'])
    const w = mountPicker()
    await flushPromises()
    await w.get('[data-testid="interest-following-theme"] [data-testid="interest-remove"]').trigger('click')
    await button(w, 'Save').trigger('click')
    await flushPromises()
    expect(put).toHaveBeenCalledWith(['person:jane'])
  })

  it('cannot save when the current interests failed to load — Save would wipe them', async () => {
    vi.spyOn(api, 'getUserInterests').mockRejectedValue(new Error('offline'))
    const put = vi.spyOn(api, 'putUserInterests').mockResolvedValue([])
    const w = mountPicker()
    await flushPromises()
    expect(w.find('[data-testid="interests-load-failed"]').exists()).toBe(true)
    const save = w.get('[data-testid="interests-save"]')
    expect(save.attributes('disabled')).toBeDefined()
    await save.trigger('click')
    expect(put).not.toHaveBeenCalled()
  })

  it('closes on the dimmed backdrop', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mountPicker()
    await flushPromises()
    await w.find('[role="dialog"]').trigger('click')
    expect(w.emitted('close')).toBeTruthy()
  })

  it('writes the saved set into the STORE, not just the server (iOS-F1)', async () => {
    // The picker PUTs an absolute list and used to tell nobody. `ProfileView` then assigned a local
    // ref and `HomeView` set its DISMISSED flag, so Home's "Personalize your Home" card — gated on
    // `interests.ids.length === 0` — kept asking after interests were chosen. Pinned here, at the
    // write, because that is the one place every opener goes through.
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'putUserInterests').mockResolvedValue(['tc:ai'])

    const store = useInterestsStore()
    expect(store.ids).toEqual([])

    const w = mountPicker()
    await flushPromises()
    await w.get('[data-testid="interest-add-theme"]').trigger('click')
    await flushPromises()
    await button(w, '+ AI').trigger('click')
    await button(w, 'Save').trigger('click')
    await flushPromises()

    // The SERVER's response is authoritative, not the local selection.
    expect(store.ids).toEqual(['tc:ai'])
    expect(store.loaded).toBe(true)
  })
})

describe('analytics (#2267)', () => {
  it('reports it was shown, with the trigger the opener passed', async () => {
    const spy = vi.spyOn(analytics, 'track')
    mountPicker()
    await flushPromises()
    expect(spy).toHaveBeenCalledWith('interests_picker_shown', { trigger: 'home_prompt' })
  })

  it('a dismissal is reported once, and a SAVE is not also a dismissal', async () => {
    // `save()` emits "saved" then "close", so a close handler alone would record every successful
    // save as a dismissal too — and since interests_dismissed is the funnel's drop-off signal, the
    // step would read as though everyone who chose interests had also abandoned it.
    const spy = vi.spyOn(analytics, 'track')
    const w = mountPicker()
    await flushPromises()
    spy.mockClear()

    // Cancel, with nothing saved.
    await w.get('[data-testid="interests-cancel"]').trigger('click')
    const names = spy.mock.calls.map((c) => c[0])
    expect(names).toContain('interests_dismissed')
    expect(names).not.toContain('interests_saved')
  })
})
