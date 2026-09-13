import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import SavedFilterBar from './SavedFilterBar.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

const AVAILABLE = [
  { key: 'episodes', label: 'Episodes' },
  { key: 'highlights', label: 'Highlights' },
]

function mountBar(props: Record<string, unknown> = {}) {
  return mount(SavedFilterBar, {
    props: { availableTypes: AVAILABLE, colorsPresent: ['amber', 'rose'], types: [], color: null, sort: 'episode', ...props },
    global: { plugins: [i18n] },
  })
}

describe('SavedFilterBar', () => {
  it('renders a chip per available type plus All, and swatches only for colours in use', () => {
    const w = mountBar()
    expect(w.find('[data-testid="saved-type-all"]').exists()).toBe(true)
    expect(w.find('[data-testid="saved-type-episodes"]').exists()).toBe(true)
    expect(w.find('[data-testid="saved-type-highlights"]').exists()).toBe(true)
    // colorsPresent = amber, rose → two filter swatches, not the whole palette.
    expect(w.findAll('[data-testid="saved-filter-swatch"]')).toHaveLength(2)
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
