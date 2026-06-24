import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import type { Segment } from '../services/types'
import TranscriptList from './TranscriptList.vue'

const segments: Segment[] = [
  { id: 's0', start: 0, end: 2.5, text: 'Hello world.', speaker: 'person:matthew-walker' },
  { id: 's1', start: 2.5, end: 5, text: 'Second line.', speaker: null },
]

describe('TranscriptList', () => {
  it('renders every segment with its timestamp and humanised speaker', () => {
    const w = mount(TranscriptList, { props: { segments, activeIndex: 0 } })
    expect(w.text()).toContain('Hello world.')
    expect(w.text()).toContain('Second line.')
    expect(w.text()).toContain('0:00')
    expect(w.text()).toContain('matthew walker') // person: prefix stripped, dashes → spaces
  })

  it('marks the active segment with the accent treatment', () => {
    const w = mount(TranscriptList, { props: { segments, activeIndex: 1 } })
    const buttons = w.findAll('button')
    expect(buttons[1].classes()).toContain('border-accent')
    expect(buttons[0].classes()).not.toContain('border-accent')
  })

  it('emits seek with the segment start on tap', async () => {
    const w = mount(TranscriptList, { props: { segments, activeIndex: 0 } })
    await w.findAll('button')[1].trigger('click')
    expect(w.emitted('seek')?.[0]).toEqual([2.5])
  })

  it('announces the active segment in an ARIA live region', () => {
    const w = mount(TranscriptList, { props: { segments, activeIndex: 0 } })
    const live = w.find('[aria-live="polite"]')
    expect(live.exists()).toBe(true)
    expect(live.text()).toBe('Hello world.')
  })
})
