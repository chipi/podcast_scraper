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
    expect(w.find('a[href="/settings"]').exists()).toBe(true)
  })

  it('the privacy page is the real policy (#2210), with its open items flagged as a draft', () => {
    const w = mountAbout('privacy')
    const policy = w.get('[data-testid="privacy-policy"]').text()
    expect(w.get('[data-testid="about-page"]').text()).not.toContain(en.about.placeholder)
    // The things the store declarations promise must be stated here too.
    for (const claim of ['Delete your account', 'Clear listening history', 'Settings › Privacy', 'aged 16', 'info@closelistening.app', 'do not sell']) {
      expect(policy, claim).toContain(claim)
    }
    // Unfinished parts are visible, not silent — the controller and backup retention.
    const notice = w.get('[data-testid="privacy-draft-notice"]').text()
    expect(notice).toContain('controller')
    expect(notice).toContain('backups')
  })

  it('the other legal pages are still placeholders', () => {
    const w = mountAbout('terms')
    expect(w.text()).toContain(en.about.placeholder)
    expect(w.find('[data-testid="privacy-policy"]').exists()).toBe(false)
  })

  it('resolves each known page slug to its title', () => {
    expect(mountAbout('third-party').get('[data-testid="about-page-title"]').text()).toBe(en.about.thirdParty)
    expect(mountAbout('terms').get('[data-testid="about-page-title"]').text()).toBe(en.about.terms)
  })

  it('falls back to the generic "About" for an unknown slug, not a specific page title or raw key', () => {
    const text = mountAbout('bogus').get('[data-testid="about-page-title"]').text()
    expect(text).toBe(en.about.title)
    expect(text).not.toBe(en.about.thirdParty)
    expect(text).not.toContain('about.')
  })
})
