import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import type { Segment } from '../services/types'
import type { GroundedSpan } from '../player/insights'
import TranscriptList from './TranscriptList.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountList = (props: Record<string, unknown>) =>
  mount(TranscriptList, { props, global: { plugins: [i18n] } })

const segments: Segment[] = [
  { id: 's0', start: 0, end: 2.5, text: 'Hello world.', speaker: 'person:matthew-walker' },
  { id: 's1', start: 2.5, end: 5, text: 'Second line.', speaker: null },
]

describe('TranscriptList', () => {
  it('renders every segment with its timestamp and humanised speaker', () => {
    const w = mountList({ segments, activeIndex: 0 })
    expect(w.text()).toContain('Hello world.')
    expect(w.text()).toContain('Second line.')
    expect(w.text()).toContain('0:00')
    expect(w.text()).toContain('matthew walker') // person: prefix stripped, dashes → spaces
  })

  it('marks the active segment with the accent treatment', () => {
    const w = mountList({ segments, activeIndex: 1 })
    const buttons = w.findAll('button')
    expect(buttons[1].classes()).toContain('border-accent')
    expect(buttons[0].classes()).not.toContain('border-accent')
  })

  it('emits seek with the segment start on tap', async () => {
    const w = mountList({ segments, activeIndex: 0 })
    await w.findAll('button')[1].trigger('click')
    expect(w.emitted('seek')?.[0]).toEqual([2.5])
  })

  it('highlights grounded segments, char-level underlines the quote, and emits insight on tap', async () => {
    const grounded: Record<number, GroundedSpan> = {
      0: { insightId: 'ins-1', insightText: 'A claim.', insightType: 'claim', quote: 'world' },
    }
    const w = mountList({ segments, activeIndex: -1, grounded })
    const btn = w.findAll('button')[0]
    expect(btn.classes()).toContain('border-grounded')
    // Char-level: only the matched phrase "world" is underlined, not the whole "Hello world.".
    const mark = btn.findAll('span').find((s) => s.classes().includes('decoration-grounded'))
    expect(mark?.text()).toBe('world')
    await btn.trigger('click')
    expect(w.emitted('seek')?.[0]).toEqual([0])
    expect(w.emitted('insight')?.[0]).toEqual(['ins-1'])
  })

  it('announces the active segment in an ARIA live region', () => {
    const w = mountList({ segments, activeIndex: 0 })
    const live = w.find('[aria-live="polite"]')
    expect(live.exists()).toBe(true)
    expect(live.text()).toBe('Hello world.')
  })
})
