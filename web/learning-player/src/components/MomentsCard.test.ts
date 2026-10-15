import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import MomentsCard from './MomentsCard.vue'
import MomentsIndex from './MomentsIndex.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

const MOMENTS = [
  { insightId: 'a', text: 'First point', speaker: 'Ann', startMs: 125_000, endMs: 195_000 },
  { insightId: 'b', text: 'Second point', speaker: null, startMs: 702_000, endMs: 772_000 },
  { insightId: 'c', text: 'Third point', speaker: 'Bo', startMs: 1_200_000, endMs: 1_270_000 },
]

function card(over: Record<string, unknown> = {}) {
  return mount(MomentsCard, {
    props: {
      moments: MOMENTS,
      index: 1,
      done: false,
      currentTime: 737,
      returnTo: 1880,
      episodeSeconds: 2890,
      ...over,
    },
    global: { plugins: [i18n] },
  })
}

describe('MomentsCard: the reel inside the artwork (operator 2026-10-10)', () => {
  it('leads with the current moment: which of how many, when, the point, who', () => {
    const w = card()
    expect(w.get('[data-testid="moments-current"]').text()).toContain('Moment 2 of 3')
    expect(w.get('[data-testid="moments-current"]').text()).toContain('11:42')
    expect(w.get('[data-testid="moments-text"]').text()).toBe('Second point')
  })

  it('marks played, current and coming moments across the top', () => {
    expect(card().findAll('[data-testid="moments-segment"]').map((s) => s.attributes('data-state'))).toEqual([
      'done',
      'now',
      'todo',
    ])
  })

  it('every control says what it does, and emits it', async () => {
    const w = card()
    await w.get('[data-testid="moments-prev"]').trigger('click')
    await w.get('[data-testid="moments-next"]').trigger('click')
    await w.get('[data-testid="moments-keep"]').trigger('click')
    expect(Object.keys(w.emitted())).toEqual(expect.arrayContaining(['prev', 'next', 'keep']))
    expect(w.get('[data-testid="moments-prev"]').attributes('aria-label')).toBe('Previous moment')
    expect(w.get('[data-testid="moments-keep"]').text()).toBe('Play on from 12:17')
    await w.get('[data-testid="moments-play-start"]').trigger('click')
    expect(w.emitted('from-start')).toHaveLength(1)
  })

  it("a learning day's end card offers the full episode both ways, under Back to Learn", async () => {
    const w = card({ done: true, index: 0, moments: [MOMENTS[0]], learnDay: 2, learnStatus: 'done', currentTime: 195 })
    expect(w.find('[data-testid="learn-clip-back"]').exists()).toBe(true)
    expect(w.get('[data-testid="learn-clip-keep"]').text()).toBe('Play on from 3:15')
    await w.get('[data-testid="learn-clip-keep"]').trigger('click')
    await w.get('[data-testid="learn-clip-play-start"]').trigger('click')
    expect(w.emitted('keep')).toHaveLength(1)
    expect(w.emitted('from-start')).toHaveLength(1)
  })

  it('names who says it, and their role when the episode knows it', () => {
    expect(card({ index: 0, speakerRoles: { Ann: 'host' } }).get('[data-testid="moments-speaker"]').text()).toBe(
      'Ann, host',
    )
    expect(card({ index: 0 }).get('[data-testid="moments-speaker"]').text()).toBe('Ann')
  })

  it('after the last moment the artwork shows the end card: from the start, back, next in queue', async () => {
    const w = card({ done: true, index: 2, nextTitle: 'The Risk Panel' })
    expect(w.find('[data-testid="moments-current"]').exists()).toBe(false)
    expect(w.get('[data-testid="moments-done"]').text()).toContain('3 moments · 3½ min of a 48 min episode')
    expect(w.get('[data-testid="moments-back"]').text()).toContain('31:20')
    await w.get('[data-testid="moments-from-start"]').trigger('click')
    await w.get('[data-testid="moments-back"]').trigger('click')
    const next = w.get('[data-testid="moments-next-episode"]')
    expect(next.text()).toContain('Next in queue: its moments')
    expect(next.text()).toContain('The Risk Panel')
    await next.trigger('click')
    expect(Object.keys(w.emitted())).toEqual(expect.arrayContaining(['from-start', 'back', 'next-episode']))
  })

  it('no next-in-queue when the queue has nothing after this episode', () => {
    expect(card({ done: true, index: 2 }).find('[data-testid="moments-next-episode"]').exists()).toBe(false)
  })
})

describe('MomentsIndex: every moment, under the transport', () => {
  it('lists them with their times; heard ticked, current marked; a tap goes there', async () => {
    const w = mount(MomentsIndex, {
      props: { moments: MOMENTS, index: 1, done: false, heard: new Set([MOMENTS[0].insightId]) },
      global: { plugins: [i18n] },
    })
    expect(w.get('[data-testid="moments-episode"]').text()).toContain('3 moments · 3½ min')
    const items = w.findAll('[data-testid="moments-index-item"]')
    expect(items).toHaveLength(3)
    expect(items[0].text()).toContain('✓')
    expect(items[1].attributes('aria-current')).toBe('true')
    await items[2].trigger('click')
    expect(w.emitted('go')![0]).toEqual([2])
  })

  it('ticks a moment heard to its end, ever — not one merely passed (operator 2026-10-14)', () => {
    const marks = (heard: Set<string>, index: number, done = false) =>
      mount(MomentsIndex, { props: { moments: MOMENTS, index, done, heard }, global: { plugins: [i18n] } })
        .findAll('[data-testid="moments-index-mark"]')
        .map((m) => m.attributes('data-state'))
    // Skipped past the first: it keeps its number.
    expect(marks(new Set(), 2)).toEqual(['todo', 'todo', 'now'])
    // Heard on an earlier visit, before this run reached it.
    expect(marks(new Set([MOMENTS[2].insightId]), 0)).toEqual(['now', 'todo', 'heard'])
    // The reel over: only what was heard is ticked.
    expect(marks(new Set([MOMENTS[1].insightId]), 2, true)).toEqual(['todo', 'heard', 'todo'])
  })
})
