import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
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

  it('shows a speaker name once per run, not on every consecutive segment', () => {
    const run: Segment[] = [
      { id: 'a', start: 0, end: 1, text: 'One.', speaker: 'person:amy-lawrence' },
      { id: 'b', start: 1, end: 2, text: 'Two.', speaker: 'person:amy-lawrence' },
      { id: 'c', start: 2, end: 3, text: 'Three.', speaker: 'person:bob-jones' },
    ]
    const w = mountList({ segments: run, activeIndex: -1 })
    const labels = w.findAll('.lp-speaker').map((n) => n.text())
    expect(labels).toEqual(['amy lawrence', 'bob jones']) // not repeated for the 2nd amy segment
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

  it('shows no capture affordance by default (canCapture off)', () => {
    const w = mountList({ segments, activeIndex: 0 })
    // only the per-segment seek buttons exist — no save buttons
    expect(w.findAll('button')).toHaveLength(segments.length)
    expect(w.find('[aria-label="Save highlight — your selected text, or this whole line"]').exists()).toBe(false)
  })

  it('renders a save button per line when canCapture and emits a whole-line capture on tap', async () => {
    const w = mountList({ segments, activeIndex: 0, canCapture: true })
    const saves = w.findAll('[aria-label="Save highlight — your selected text, or this whole line"]')
    expect(saves).toHaveLength(segments.length)
    await saves[1].trigger('click')
    // no selection → whole-line capture (sub is null)
    expect(w.emitted('capture')?.[0]).toEqual([{ segment: segments[1], sub: null }])
    // the seek button still works independently of the capture button
    await w.findAll('[data-testid="seg"]')[1].trigger('click')
    expect(w.emitted('seek')?.[0]).toEqual([2.5])
  })

  it('captures the selected phrase (sub-range) when text is selected within the line (FR1.2)', async () => {
    const w = mountList({ segments, activeIndex: 0, canCapture: true })
    // Select "world" (chars 6–11) within the first line's rendered text node.
    const textEl = w.findAll('span.leading-relaxed')[0].element as HTMLElement
    const textNode = textEl.querySelector('span')!.firstChild as Text
    const range = document.createRange()
    range.setStart(textNode, 6)
    range.setEnd(textNode, 11)
    vi.spyOn(window, 'getSelection').mockReturnValue({
      isCollapsed: false,
      rangeCount: 1,
      getRangeAt: () => range,
      toString: () => 'world',
    } as unknown as Selection)
    await w.findAll('[aria-label="Save highlight — your selected text, or this whole line"]')[0].trigger('click')
    expect(w.emitted('capture')?.[0]).toEqual([
      { segment: segments[0], sub: { char_start: 6, char_end: 11, quote_text: 'world' } },
    ])
    vi.restoreAllMocks()
  })

  it('reflects saved state via aria-pressed + the saved label', () => {
    const w = mountList({
      segments,
      activeIndex: -1,
      canCapture: true,
      savedSegmentIds: new Set(['s0']),
    })
    expect(w.find('[aria-label="Line saved — tap to remove"]').exists()).toBe(true)
    const pressed = w.findAll('[aria-pressed="true"]')
    expect(pressed).toHaveLength(1)
  })
})
