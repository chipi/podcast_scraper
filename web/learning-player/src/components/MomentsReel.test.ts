import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import MomentsReel from './MomentsReel.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

const MOMENTS = [
  { insightId: 'a', text: 'First point', speaker: 'Ann', startMs: 125_000, endMs: 195_000 },
  { insightId: 'b', text: 'Second point', speaker: null, startMs: 702_000, endMs: 772_000 },
  { insightId: 'c', text: 'Third point', speaker: 'Bo', startMs: 1_200_000, endMs: 1_270_000 },
]

function make(over: Record<string, unknown> = {}) {
  return mount(MomentsReel, {
    props: {
      title: 'The Bessent Tape',
      showTitle: 'Long Horizon Notes',
      artwork: null,
      moments: MOMENTS,
      index: 1,
      done: false,
      playing: true,
      currentTime: 737,
      returnTo: 1880,
      episodeSeconds: 2890,
      ...over,
    },
    global: { plugins: [i18n] },
  })
}

describe('MomentsReel (operator 2026-10-10)', () => {
  it('leads with the current moment: which of how many, when, the point, who', () => {
    const w = make()
    const card = w.get('[data-testid="moments-current"]')
    expect(card.text()).toContain('Moment 2 of 3')
    expect(card.text()).toContain('11:42')
    expect(w.get('[data-testid="moments-text"]').text()).toBe('Second point')
    // 3 clips of 70 s = 3½ min.
    expect(w.get('[data-testid="moments-episode"]').text()).toContain('3 moments · 3½ min')
  })

  it('marks played, current and coming moments on the bar and in the index', () => {
    const w = make()
    expect(w.findAll('[data-testid="moments-segment"]').map((s) => s.attributes('data-state'))).toEqual([
      'done',
      'now',
      'todo',
    ])
    const items = w.findAll('[data-testid="moments-index-item"]')
    expect(items).toHaveLength(3)
    expect(items[1].attributes('aria-current')).toBe('true')
    expect(items[0].text()).toContain('✓')
  })

  it('every control says what it does, and emits it', async () => {
    const w = make()
    await w.get('[data-testid="moments-prev"]').trigger('click')
    await w.get('[data-testid="moments-next"]').trigger('click')
    await w.get('[data-testid="moments-toggle"]').trigger('click')
    await w.get('[data-testid="moments-keep"]').trigger('click')
    await w.get('[data-testid="moments-close"]').trigger('click')
    await w.findAll('[data-testid="moments-index-item"]')[2].trigger('click')
    expect(Object.keys(w.emitted())).toEqual(
      expect.arrayContaining(['prev', 'next', 'toggle', 'keep', 'close', 'go']),
    )
    expect(w.emitted('go')![0]).toEqual([2])
    expect(w.get('[data-testid="moments-toggle"]').attributes('aria-label')).toBe('Pause')
    expect(w.get('[data-testid="moments-keep"]').text()).toContain('12:17')
  })

  it('after the last moment: an end card with from-the-start and back-to-where-you-were', async () => {
    const w = make({ done: true, index: 2, playing: false })
    expect(w.find('[data-testid="moments-current"]').exists()).toBe(false)
    expect(w.find('[data-testid="moments-keep"]').exists()).toBe(false)
    const card = w.get('[data-testid="moments-done"]')
    expect(card.text()).toContain('3 moments · 3½ min of a 48 min episode')
    expect(w.get('[data-testid="moments-back"]').text()).toContain('31:20')
    await w.get('[data-testid="moments-from-start"]').trigger('click')
    expect(w.emitted('from-start')).toHaveLength(1)
  })
})

describe('MomentsReel end card chains through the queue', () => {
  it('offers the next queued episode\'s moments when there is one', async () => {
    const w = make({ done: true, index: 2, playing: false, nextTitle: 'The Risk Panel' })
    const btn = w.get('[data-testid="moments-next-episode"]')
    expect(btn.text()).toContain('Next in queue: its moments')
    expect(btn.text()).toContain('The Risk Panel')
    await btn.trigger('click')
    expect(w.emitted('next-episode')).toHaveLength(1)
  })
  it('and nothing when the queue has nothing after this episode', () => {
    expect(make({ done: true, index: 2 }).find('[data-testid="moments-next-episode"]').exists()).toBe(false)
  })
})

describe('MomentsReel keeps the obi: Episode · Brief · About', () => {
  it('Episode leaves the reel the way ✕ does; Brief and About open their sheets', async () => {
    const w = make({ hasBrief: true, hasAbout: true })
    const labels = w.findAll('[data-testid="moments-doors"] button').map((b) => b.text())
    expect(labels).toEqual(['Episode', 'Brief', 'About'])
    await w.get('[data-testid="moments-door-episode"]').trigger('click')
    await w.get('[data-testid="moments-door-brief"]').trigger('click')
    await w.get('[data-testid="moments-door-about"]').trigger('click')
    expect(w.emitted('close')).toHaveLength(1)
    expect(w.emitted('brief')).toHaveLength(1)
    expect(w.emitted('about')).toHaveLength(1)
  })
  it('only the doors the episode has: no description, no About', () => {
    const w = make({ hasBrief: true, hasAbout: false })
    expect(w.find('[data-testid="moments-door-about"]').exists()).toBe(false)
  })
})

describe('MomentsReel names who says it, and their role', () => {
  it('"Ann, host" when the episode knows her role; just the name when it does not', () => {
    const w = make({ index: 0, speakerRoles: { Ann: 'host' } })
    expect(w.get('[data-testid="moments-speaker"]').text()).toBe('Ann, host')
    expect(make({ index: 0 }).get('[data-testid="moments-speaker"]').text()).toBe('Ann')
  })
})
