import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import TopVoices from './TopVoices.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/person/:id', name: 'person', component: { template: '<div/>' } }],
})
const people = [
  { id: 'person:a', name: 'Ann', kind: 'person' as const },
  { id: 'person:b', name: 'Bo', kind: 'person' as const },
]

describe('TopVoices', () => {
  it('renders buttons that emit the person id when no route is given (card / sheet)', async () => {
    const w = mount(TopVoices, { props: { people }, global: { plugins: [i18n, router] } })
    const first = w.get('[data-testid="ec-top-voice"]')
    expect(first.element.tagName).toBe('BUTTON')
    await first.trigger('click')
    expect(w.emitted('open')?.[0]?.[0]).toBe('person:a')
  })

  it('renders links when the caller gives a route (page)', () => {
    const w = mount(TopVoices, {
      props: { people, routeFor: (id: string) => ({ name: 'person', params: { id } }) },
      global: { plugins: [i18n, router] },
    })
    expect(w.get('[data-testid="ec-top-voice"]').attributes('href')).toBe('/person/person:a')
  })

  it('renders nothing for no people', () => {
    const w = mount(TopVoices, { props: { people: [] }, global: { plugins: [i18n, router] } })
    expect(w.find('[data-testid="ec-top-voices"]').exists()).toBe(false)
  })
})
