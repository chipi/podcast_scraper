// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { TopPersonItem } from '../../api/corpusPersonsApi'
import TopVoices from './TopVoices.vue'

const ann: TopPersonItem = {
  person_id: 'person:ann',
  display_name: 'Ann Lee',
  episode_count: 4,
  insight_count: 12,
  top_topics: ['ai', 'chips'],
}
const bob: TopPersonItem = {
  person_id: 'person:bob',
  display_name: 'Bob Ray',
  episode_count: 2,
  insight_count: 3,
  top_topics: [],
}

function mountVoices(props: { persons: TopPersonItem[]; loading?: boolean; error?: string | null }) {
  return mount(TopVoices, { props: { loading: false, error: null, ...props } })
}

describe('TopVoices', () => {
  it('renders no person cards while loading, even with data', () => {
    const w = mountVoices({ persons: [ann], loading: true })
    expect(w.text()).toContain('Loading…')
    expect(w.text()).not.toContain('4 episodes · 12 insights')
  })

  it('shows the error instead of the empty state', () => {
    const w = mountVoices({ persons: [], error: 'HTTP 500' })
    expect(w.text()).toContain('HTTP 500')
    expect(w.text()).not.toContain('No speaker intelligence')
  })

  it('shows the empty state and no insight line when there are no persons', () => {
    const w = mountVoices({ persons: [] })
    expect(w.text()).toContain('No speaker intelligence found for this corpus.')
    expect(w.text()).not.toContain('speakers with grounded insights')
  })

  it('renders one card per person with counts and topic chips', () => {
    const w = mountVoices({ persons: [ann, bob] })
    const text = w.text()
    expect(text).toContain('Ann Lee')
    expect(text).toContain('4 episodes · 12 insights')
    expect(text).toContain('ai')
    expect(text).toContain('chips')
    expect(text).toContain('Bob Ray')
    expect(text).toContain('2 episodes · 3 insights')
  })

  it('summarises the leading speaker in the insight line', () => {
    const w = mountVoices({ persons: [ann, bob] })
    expect(w.text()).toContain(
      '2 speakers with grounded insights — Ann Lee leads with 12 insights across 4 episodes.',
    )
  })
})
