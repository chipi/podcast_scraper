import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import AboutPageView from './AboutPageView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }

function mountAbout(page: string) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/about/:page', name: 'about-page', component: AboutPageView, props: true },
      { path: '/settings', name: 'settings', component: stub },
    ],
  })
  return mount(AboutPageView, { props: { page }, global: { plugins: [i18n, router] } })
}

describe('AboutPageView', () => {
  it('titles the page from the :page param and links back to Settings', () => {
    const w = mountAbout('privacy')
    expect(w.get('[data-testid="about-page-title"]').text()).toBe(en.about.privacy)
    expect(w.get('[data-testid="about-page"]').text()).toContain(en.about.placeholder)
    expect(w.get('a[href="/settings"]').exists()).toBe(true)
  })

  it('resolves each known page slug to its title', () => {
    expect(mountAbout('third-party').get('[data-testid="about-page-title"]').text()).toBe(en.about.thirdParty)
    expect(mountAbout('terms').get('[data-testid="about-page-title"]').text()).toBe(en.about.terms)
  })

  it('falls back to a known title for an unknown slug rather than rendering a raw key', () => {
    const text = mountAbout('bogus').get('[data-testid="about-page-title"]').text()
    expect(text).toBe(en.about.thirdParty)
    expect(text).not.toContain('about.')
  })
})
