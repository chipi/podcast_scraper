import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import ListToolbar from './ListToolbar.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
// Sort options are now caller-supplied (both Browse tabs share this bar with different keys).
const SORT_OPTIONS = [
  { value: 'newest', label: 'Newest' },
  { value: 'oldest', label: 'Oldest' },
  { value: 'az', label: 'A–Z' },
  { value: 'za', label: 'Z–A' },
]
const mountBar = (props = {}) =>
  mount(ListToolbar, {
    props: { search: '', sort: 'newest', filter: 'all', sortOptions: SORT_OPTIONS, ...props },
    global: { plugins: [i18n] },
  })

describe('ListToolbar', () => {
  it('shows its controls outright — no disclosure to find first (#2004 item 10)', () => {
    // Was collapsed behind a "Sort & filter" pill while the Shows tab, two tabs away, showed its
    // filter field and sort select outright. Same job, two interaction models, and the collapsed one
    // hid the fact that filtering was possible at all.
    const w = mountBar()
    expect(w.find('[data-testid="list-toolbar-search"]').exists()).toBe(true)
    expect(w.find('[data-testid="list-toolbar-sort"]').exists()).toBe(true)
    expect(w.findAll('button').some((b) => b.text().includes('Sort & filter'))).toBe(false)
  })

  it('renders the search + compact controls, no native selects (operator 2026-09-14)', () => {
    // The controls are now compact ToolbarMenu triggers (search wide, sort/view little circles),
    // NOT native <select>s — a select reserved width for its widest option and ate the row.
    const w = mountBar()
    expect(w.find('[data-testid="list-toolbar-search"]').exists()).toBe(true)
    expect(w.find('[data-testid="list-toolbar-sort"]').exists()).toBe(true)
    expect(w.find('[data-testid="list-toolbar-view"]').exists()).toBe(true)
    expect(w.findAll('select')).toHaveLength(0)
    // No filter control unless the caller supplies options.
    expect(w.find('[data-testid="list-toolbar-filter"]').exists()).toBe(false)
  })

  it('on a phone the search is a 🔍 circle: tapped it takes the row, × closes and clears (operator 2026-10-15)', async () => {
    const w = mountBar()
    const field = () => w.get('[data-testid="list-toolbar-search"]').element.parentElement!
    // Closed: the circle shows; the field waits for lg (CSS), the chips hold the row.
    expect(w.find('[data-testid="list-toolbar-search-open"]').exists()).toBe(true)
    expect(field().className).toContain('hidden lg:flex')
    await w.get('[data-testid="list-toolbar-search-open"]').trigger('click')
    expect(w.find('[data-testid="list-toolbar-search-open"]').exists()).toBe(false)
    expect(field().className).not.toContain('hidden')
    expect(w.get('[data-testid="list-toolbar-sort"]').element.closest('div.items-center')!.className).toContain('hidden lg:flex')
    await w.get('[data-testid="list-toolbar-search-close"]').trigger('click')
    expect(w.find('[data-testid="list-toolbar-search-open"]').exists()).toBe(true)
  })

  it('stays open while it holds text, so a filtered list never hides its query', async () => {
    const w = mountBar({ search: 'memory' })
    expect(w.find('[data-testid="list-toolbar-search-open"]').exists()).toBe(false)
    // × clears it (and with it the filter), which is what closes it.
    await w.get('[data-testid="list-toolbar-search-close"]').trigger('click')
    expect(w.emitted('update:search')?.at(-1)).toEqual([''])
  })

  it('two-way-binds search via v-model (update:search)', async () => {
    const w = mountBar()
    await w.find('input[type="search"]').setValue('memory')
    expect(w.emitted('update:search')?.at(-1)).toEqual(['memory'])
  })

  it('two-way-binds sort via v-model — open the menu, pick an option (update:sort)', async () => {
    const w = mountBar()
    await w.get('[data-testid="list-toolbar-sort"]').trigger('click') // open the sort menu
    await w.get('[data-testid="list-toolbar-sort-opt-za"]').trigger('click')
    expect(w.emitted('update:sort')?.at(-1)).toEqual(['za'])
  })

})
