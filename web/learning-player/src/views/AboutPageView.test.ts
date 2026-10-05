import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises } from '@vue/test-utils'
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

  it('the terms page is a real first version, with its open items flagged as a draft', () => {
    const w = mountAbout('terms')
    expect(w.text()).not.toContain(en.about.placeholder)
    const terms = w.get('[data-testid="terms-of-use"]').text()
    for (const claim of ['16 or older', 'belong to their creators', 'can be wrong', 'Delete account', 'info@closelistening.app', 'access token']) {
      expect(terms, claim).toContain(claim)
    }
    const notice = w.get('[data-testid="terms-draft-notice"]').text()
    expect(notice).toContain('being registered')
    expect(notice).toContain('not yet been reviewed by a lawyer')
  })

  describe('third-party software', () => {
    afterEach(() => vi.unstubAllGlobals())

    it('lists the generated packages with their licences', async () => {
      vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
        generated: '2026-10-05T08:00:00Z',
        packages: [
          { name: 'vue', version: '3.5.0', license: 'MIT', url: 'https://vuejs.org', text: 'MIT License …' },
          { name: 'Google Sans (font)', version: '', license: 'OFL-1.1', url: 'https://fonts.google.com', text: null },
        ],
      }))))
      const w = mountAbout('third-party')
      await flushPromises()
      const rows = w.findAll('[data-testid="third-party-entry"]')
      // Alphabetical: the font, then vue.
      expect(rows).toHaveLength(2)
      expect(rows[1].text()).toContain('vue')
      expect(rows[1].text()).toContain('MIT')
      expect(rows[1].find('pre').text()).toBe('MIT License …')
      expect(rows[0].text()).toContain('ships no licence file')
      expect(w.get('[data-testid="third-party-count"]').text()).toContain('2 libraries')
    })

    it('groups scoped packages under their scope; an unscoped package is a row of its own', async () => {
      const pkg = (name: string) => ({ name, version: '1.0.0', license: 'MIT', url: 'https://x', text: 'MIT' })
      vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
        generated: '2026-10-05T08:00:00Z',
        packages: [pkg('@babel/parser'), pkg('@babel/types'), pkg('pinia'), pkg('@vue/shared')],
      }))))
      const w = mountAbout('third-party')
      await flushPromises()
      const groups = w.findAll('[data-testid="third-party-group"]')
      expect(groups).toHaveLength(2)
      const head = (i: number) => groups[i].find('summary').text()
      expect(head(0)).toContain('@babel')
      expect(head(0)).toContain('2 libraries')
      expect(head(1)).toContain('@vue')
      expect(head(1)).toContain('1 library')
      // Members sit under their scope, named without the prefix.
      const members = groups[0].findAll('[data-testid="third-party-entry"] summary').map((m) => m.text())
      expect(members[0]).toMatch(/^›parser/)
      expect(members[1]).toMatch(/^›types/)
      // Order across groups and singles is alphabetical: @babel, @vue, pinia.
      const top = w.findAll('[data-testid="third-party"] > ul > li').map((li) => li.find('summary').text())
      expect(top.map((x) => x.replace(/^›/, '').match(/^[@a-z]+/)?.[0])).toEqual(['@babel', '@vue', 'pinia'])
    })

    it('says so when the list cannot be loaded, rather than showing an empty page', async () => {
      vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })))
      const w = mountAbout('third-party')
      await flushPromises()
      expect(w.find('[data-testid="third-party-failed"]').exists()).toBe(true)
    })
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
