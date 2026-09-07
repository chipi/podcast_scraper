import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import ListToolbar from './ListToolbar.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountBar = (props = {}) =>
  mount(ListToolbar, { props: { search: '', sort: 'newest', filter: 'all', ...props }, global: { plugins: [i18n] } })

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

  it('renders exactly two controls, on one line (#2004 item 10)', () => {
    // The ask was "the same thing as on shows": filter left, sort right, one line. My first attempt
    // kept all four controls and let them wrap into three lines — four do not fit a phone row.
    const w = mountBar({ shows: [{ id: 'f1', label: 'Show One' }] })
    expect(w.get('[data-testid="list-toolbar-search"]').exists()).toBe(true)
    expect(w.get('[data-testid="list-toolbar-sort"]').exists()).toBe(true)
    expect(w.findAll('select')).toHaveLength(1)
    expect(w.find('[data-testid="list-toolbar-filter"]').exists()).toBe(false)
    expect(w.find('[data-testid="list-toolbar-show"]').exists()).toBe(false)
  })

  it('two-way-binds search via v-model (update:search)', async () => {
    const w = mountBar()
    await w.find('input[type="search"]').setValue('memory')
    expect(w.emitted('update:search')?.at(-1)).toEqual(['memory'])
  })

  it('two-way-binds sort via v-model (update:sort)', async () => {
    const w = mountBar()
    await w.find('[data-testid="list-toolbar-sort"]').setValue('title')
    expect(w.emitted('update:sort')?.at(-1)).toEqual(['title'])
  })

})
