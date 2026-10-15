import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import EpisodeListenChoice from './EpisodeListenChoice.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const choice = () =>
  mount(EpisodeListenChoice, {
    props: { time: 737, testid: 'x-listen', keepTestid: 'x-keep', startTestid: 'x-play-start' },
    global: { plugins: [i18n] },
  })

describe('EpisodeListenChoice (operator 2026-10-15)', () => {
  it('names both ways into the episode: on from the clip, with its time, or from the start', () => {
    const w = choice()
    expect(w.get('[data-testid="x-listen"]').text()).toContain('Listen to the full episode')
    expect(w.get('[data-testid="x-keep"]').text()).toBe('Play on from 12:17')
    expect(w.get('[data-testid="x-play-start"]').text()).toBe('Play from the start')
  })

  it('emits which one was chosen', async () => {
    const w = choice()
    await w.get('[data-testid="x-keep"]').trigger('click')
    await w.get('[data-testid="x-play-start"]').trigger('click')
    expect(w.emitted('play-on')).toHaveLength(1)
    expect(w.emitted('from-start')).toHaveLength(1)
  })
})
