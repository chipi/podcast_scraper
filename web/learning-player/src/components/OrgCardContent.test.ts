import { mount, RouterLinkStub } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'

import en from '../i18n/locales/en.json'
import type { OrgCard } from '../services/types'
import OrgCardContent from './OrgCardContent.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function org(): OrgCard {
  return {
    id: 'org:acme',
    label: 'Acme Labs',
    episode_count: 1,
    episodes: [
      { slug: 'ep-a', title: 'Episode A', podcast_title: 'My Show', feed_id: 'f1' } as never,
    ],
    related_people: [{ id: 'person:jane', name: 'Jane Doe', kind: 'person' } as never],
    related_orgs: [{ id: 'org:globex', name: 'Globex', kind: 'org' } as never],
    related_topics: [{ id: 'topic:ai', label: 'AI' } as never],
  }
}

function mountIt() {
  return mount(OrgCardContent, {
    props: { org: org() },
    global: { plugins: [i18n], stubs: { RouterLink: RouterLinkStub } },
  })
}

describe('OrgCardContent (#2031)', () => {
  it('renders the lean org body: episodes + co-occurring people / orgs / topics', () => {
    const w = mountIt()
    // Search button carries the org name; the co-occurrence sections render each kind.
    expect(w.get('[data-testid="ec-search-library"]').text()).toContain('Acme Labs')
    expect(w.text()).toContain('Episode A')
    expect(w.get('[data-testid="ec-related-orgs"]').text()).toContain('Globex')
    expect(w.text()).toContain('Jane Doe')
    expect(w.text()).toContain('AI')
  })

  it('drills into a co-occurring org in place (emits open organization)', async () => {
    const w = mountIt()
    await w.get('[data-testid="ec-related-org"]').trigger('click')
    expect(w.emitted('open')![0]).toEqual([{ kind: 'organization', id: 'org:globex' }])
  })

  it('opens a co-occurring person / topic with the right kind', async () => {
    const w = mountIt()
    const chips = w.findAll('button.text-person')
    await chips[0].trigger('click')
    expect(w.emitted('open')!.at(-1)).toEqual([{ kind: 'person', id: 'person:jane' }])
  })

  it('renders the org_web enrichment block (logo + description + facts) when present (#2035)', () => {
    const withWeb = { ...org(), web: {
      description: 'AI safety research lab',
      source: 'wikidata',
      source_url: 'https://www.wikidata.org/wiki/Q1',
      logo_url: '/api/app/organizations/org:acme/logo',
      logo_license: 'CC-BY-SA 4.0',
      founded: '2015',
      industry: 'Artificial intelligence',
      website: 'https://acme.example',
    } }
    const w = mount(OrgCardContent, {
      props: { org: withWeb },
      global: { plugins: [i18n], stubs: { RouterLink: RouterLinkStub } },
    })
    const block = w.get('[data-testid="ec-org-web"]')
    expect(block.text()).toContain('AI safety research lab')
    expect(block.text()).toContain('2015')
    expect(block.text()).toContain('Artificial intelligence')
    expect(w.get('[data-testid="ec-org-logo"]').attributes('src')).toBe(
      '/api/app/organizations/org:acme/logo',
    )
    expect(w.get('[data-testid="ec-org-website"]').attributes('href')).toBe('https://acme.example')
  })

  it('stays lean (no web block) when the org has no enrichment', () => {
    const w = mountIt() // org() has no web
    expect(w.find('[data-testid="ec-org-web"]').exists()).toBe(false)
  })
})
