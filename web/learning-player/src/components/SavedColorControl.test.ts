import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import SavedColorControl from './SavedColorControl.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
// The palette teleports to <body> via the shared popover shell (useAnchoredMenu) so it can never
// open off-screen. Stub teleport so it renders inline for `find`, and attach to the document so the
// Escape / outside-pointer dismissal is real.
const mountControl = (color: string | null = null) =>
  mount(SavedColorControl, {
    props: { color },
    attachTo: document.body,
    global: { plugins: [i18n], stubs: { teleport: true } },
  })

afterEach(() => {
  document.body.innerHTML = ''
})

describe('SavedColorControl', () => {
  it('keeps the palette closed until the dot is tapped', async () => {
    const w = mountControl(null)
    expect(w.find('[data-testid="saved-swatch"]').exists()).toBe(false)
    await w.find('[data-testid="saved-color"]').trigger('click')
    expect(w.findAll('[data-testid="saved-swatch"]')).toHaveLength(5)
  })

  it('emits the picked token, and closes', async () => {
    const w = mountControl(null)
    await w.find('[data-testid="saved-color"]').trigger('click')
    await w.find('[aria-label="Set colour: Amber"]').trigger('click')
    expect(w.emitted('pick')?.[0]).toEqual(['amber'])
    // closes on pick
    expect(w.find('[data-testid="saved-swatch"]').exists()).toBe(false)
  })

  it('tapping the active colour clears it (emits null)', async () => {
    const w = mountControl('amber')
    await w.find('[data-testid="saved-color"]').trigger('click')
    await w.find('[aria-label="Set colour: Amber"]').trigger('click')
    expect(w.emitted('pick')?.[0]).toEqual([null])
  })

  it('Escape closes the palette without picking', async () => {
    const w = mountControl(null)
    await w.find('[data-testid="saved-color"]').trigger('click')
    expect(w.find('[data-testid="saved-swatch"]').exists()).toBe(true)
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    await w.vm.$nextTick()
    expect(w.find('[data-testid="saved-swatch"]').exists()).toBe(false)
    expect(w.emitted('pick')).toBeUndefined()
  })
})
