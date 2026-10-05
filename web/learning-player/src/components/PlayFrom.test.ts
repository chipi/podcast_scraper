import { mount, RouterLinkStub } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import PlayFrom from './PlayFrom.vue'
import en from '../i18n/locales/en.json'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
type Props = InstanceType<typeof PlayFrom>['$props']
const mountIt = (props: Props) =>
  mount(PlayFrom, { props, global: { plugins: [i18n], stubs: { RouterLink: RouterLinkStub } } })

/** The ONE jump-to-a-moment control (operator 2026-10-05): "▶ Play from 1:05" everywhere. */
describe('PlayFrom', () => {
  it('says "Play from" and the time', () => {
    expect(mountIt({ seconds: 65 }).text()).toBe('▶ Play from 1:05')
  })

  it('is a link when given a destination — one that PLAYS (?play=1)', () => {
    // An explicit ▶ starts playback (operator 2026-10-05); the moment's own query is kept.
    const to = { name: 'player', params: { slug: 'ep' }, query: { t: '65' } }
    const w = mountIt({ seconds: 65, to })
    expect(w.getComponent(RouterLinkStub).props('to')).toEqual({
      ...to,
      query: { t: '65', play: '1' },
    })
    expect(w.find('button').exists()).toBe(false)
  })

  it('is a button that emits click otherwise', async () => {
    const w = mountIt({ seconds: 65 })
    await w.get('button').trigger('click')
    expect(w.emitted('click')).toHaveLength(1)
  })

  it('never invents a 0:00 for a moment with no time', () => {
    expect(mountIt({ seconds: null, fallback: 'Open' }).text()).toBe('▶ Open')
  })
})
