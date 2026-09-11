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
})
