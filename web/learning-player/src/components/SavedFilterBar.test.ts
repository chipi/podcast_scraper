import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import SavedFilterBar from './SavedFilterBar.vue'
import { HIGHLIGHT_COLORS } from '../utils/highlightColors'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

const AVAILABLE = [
  { key: 'episodes', label: 'Episodes' },
  { key: 'highlights', label: 'Highlights' },
]

function mountBar(props: Record<string, unknown> = {}) {
  return mount(SavedFilterBar, {
    props: { availableTypes: AVAILABLE, types: [], color: null, sort: 'episode', ...props },
    global: { plugins: [i18n] },
  })
}

describe('SavedFilterBar', () => {
  it('renders a chip per available type plus All, and the WHOLE colour palette', () => {
    const w = mountBar()
    expect(w.find('[data-testid="saved-type-all"]').exists()).toBe(true)
    expect(w.find('[data-testid="saved-type-episodes"]').exists()).toBe(true)
    expect(w.find('[data-testid="saved-type-highlights"]').exists()).toBe(true)
    // The full palette, regardless of what is in use (operator 2026-09-17). Rendering only the
    // colours present made the control's SIZE a function of the data: one saved amber item left a
    // single lone dot, which reads as broken rather than as a colour filter.
    expect(w.findAll('[data-testid="saved-filter-swatch"]')).toHaveLength(HIGHLIGHT_COLORS.length)
  })

  it('leads the strip with an "any colour" ring that clears the colour filter only', () => {
    // The only way back used to be the shared "Clear", which also dropped the type and search
    // filters — one control undoing three things the user did not ask to undo.
    const w = mountBar({ color: 'amber', types: ['episodes'], search: 'x' })
    const any = w.find('[data-testid="saved-filter-swatch-any"]')
    expect(any.exists()).toBe(true)
    return any.trigger('click').then(() => {
      expect(w.emitted('update:color')?.at(-1)).toEqual([null])
      expect(w.emitted('update:types'), 'clearing the colour also cleared types').toBeUndefined()
      expect(w.emitted('update:search'), 'clearing the colour also cleared the search').toBeUndefined()
    })
  })

  it('omits the colour strip where nothing can carry a colour (Following)', () => {
    const w = mountBar({ showColors: false })
    expect(w.findAll('[data-testid="saved-filter-swatch"]')).toHaveLength(0)
    expect(w.find('[data-testid="saved-filter-swatch-any"]').exists()).toBe(false)
  })

  it('toggles a type into the v-model and back out', async () => {
    const w = mountBar()
    await w.find('[data-testid="saved-type-episodes"]').trigger('click')
    expect(w.emitted('update:types')?.at(-1)).toEqual([['episodes']])
  })

  it('the All chip clears the type selection', async () => {
    const w = mountBar({ types: ['episodes'] })
    await w.find('[data-testid="saved-type-all"]').trigger('click')
    expect(w.emitted('update:types')?.at(-1)).toEqual([[]])
  })

  it('picking a colour sets it; re-picking the active one clears it', async () => {
    const w = mountBar({ color: null })
    await w.findAll('[data-testid="saved-filter-swatch"]')[0].trigger('click')
    expect(w.emitted('update:color')?.at(-1)).toEqual(['amber'])
    const w2 = mountBar({ color: 'amber' })
    await w2.findAll('[data-testid="saved-filter-swatch"]')[0].trigger('click')
    expect(w2.emitted('update:color')?.at(-1)).toEqual([null])
  })

  it('changing sort updates the v-model', async () => {
    const w = mountBar()
    await w.find('[data-testid="saved-sort"]').setValue('title')
    expect(w.emitted('update:sort')?.at(-1)).toEqual(['title'])
  })

  it('hides entirely when there are no saved kinds', () => {
    const w = mountBar({ availableTypes: [] })
    expect(w.find('[data-testid="saved-filter-bar"]').exists()).toBe(false)
  })
})
