// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import PersonLandingView from './PersonLandingView.vue'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

// #1055 — the relational connections fetches (topics + co-speakers) are mocked; we assert
// the Connections section renders what they return.
const fetchPositions = vi.fn()
const fetchPersonTopics = vi.fn()
const fetchCoSpeakers = vi.fn()
vi.mock('../../api/relationalApi', () => ({
  fetchPositions: (...a: unknown[]) => fetchPositions(...a),
  fetchPersonTopics: (...a: unknown[]) => fetchPersonTopics(...a),
  fetchCoSpeakers: (...a: unknown[]) => fetchCoSpeakers(...a),
}))
const fetchPersonProfile = vi.fn()
vi.mock('../../api/cilApi', () => ({
  fetchPersonProfile: (...a: unknown[]) => fetchPersonProfile(...a),
}))

const STUBS = {
  SubjectTimelineChart: { name: 'SubjectTimelineChart', template: '<div data-stub="timeline" />' },
}

function rel(id: string, text: string, type: string) {
  return { id, type, text, show_id: '', episode_id: '' }
}

async function mountWith(): Promise<ReturnType<typeof mount>> {
  const w = mount(PersonLandingView, { attachTo: document.body, global: { stubs: STUBS } })
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  useSubjectStore().focusPerson('person:alice')
  await flushPromises()
  await flushPromises()
  return w
}

describe('PersonLandingView — connections (#1055)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchPositions.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchPersonProfile.mockResolvedValue({ subject: 'person:alice', profile: {} })
    fetchPersonTopics.mockResolvedValue({
      subject: 'person:alice',
      results: [rel('topic:ai', 'AI ethics', 'topic'), rel('topic:reg', 'AI regulation', 'topic')],
    })
    fetchCoSpeakers.mockResolvedValue({
      subject: 'person:alice',
      results: [rel('person:bob', 'Bob', 'person')],
    })
  })

  it('renders the person topics as chips', async () => {
    const w = await mountWith()
    const topics = w.get('[data-testid="person-landing-topics"]')
    const chips = w.findAll('[data-testid="person-landing-topic-chip"]')
    expect(chips.map((c) => c.text())).toEqual(['AI ethics', 'AI regulation'])
    expect(topics.exists()).toBe(true)
  })

  it('renders co-speakers as chips', async () => {
    const w = await mountWith()
    const chips = w.findAll('[data-testid="person-landing-co-speaker-chip"]')
    expect(chips.map((c) => c.text())).toEqual(['Bob'])
  })

  it('shows empty-state copy when there are no co-speakers', async () => {
    fetchCoSpeakers.mockResolvedValue({ subject: 'person:alice', results: [] })
    const w = await mountWith()
    expect(w.find('[data-testid="person-landing-co-speakers-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="person-landing-co-speaker-chip"]').exists()).toBe(false)
  })

  it('queries the relational connection endpoints with the corpus path + person id', async () => {
    await mountWith()
    expect(fetchPersonTopics).toHaveBeenCalledWith('/corpus', 'person:alice')
    expect(fetchCoSpeakers).toHaveBeenCalledWith('/corpus', 'person:alice')
  })
})
