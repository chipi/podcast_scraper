import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import ListToolbar from './ListToolbar.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountBar = (props = {}) =>
  mount(ListToolbar, { props: { search: '', sort: 'newest', filter: 'all', ...props }, global: { plugins: [i18n] } })

describe('ListToolbar', () => {
  it('is collapsed by default and expands the controls on toggle', async () => {
    const w = mountBar()
    const toggle = w.findAll('button').find((b) => b.text().includes('Sort & filter'))!
    expect(toggle.attributes('aria-expanded')).toBe('false')
    await toggle.trigger('click')
    expect(toggle.attributes('aria-expanded')).toBe('true')
  })

  it('two-way-binds search via v-model (update:search)', async () => {
    const w = mountBar()
    await w.findAll('button').find((b) => b.text().includes('Sort & filter'))!.trigger('click')
    await w.find('input[type="search"]').setValue('memory')
    expect(w.emitted('update:search')?.at(-1)).toEqual(['memory'])
  })

  it('renders a show filter only when shows are provided', async () => {
    const w = mountBar({ shows: [{ id: 'f1', label: 'Show One' }] })
    await w.findAll('button').find((b) => b.text().includes('Sort & filter'))!.trigger('click')
    expect(w.text()).toContain('Show One')
  })
})
