// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import PersonLandingView from './PersonLandingView.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import type { ParsedArtifact } from '../../types/artifact'

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

// #1048 — Person Landing shell tab restructure: Person Profile + Position
// Tracker pair (placeholder), identity-header additions (role + episode count
// + organization chips), and ABOUT∩MENTIONS_PERSON ranked-topic overview.
function makeArtifactWithPersonAndTopics(): ParsedArtifact {
  // person:alice mentioned in two insights, one ABOUT topic:ai (twice via
  // two ABOUT edges from the same insight → collapses to 1 pair), the other
  // ABOUT topic:reg. Also one MENTIONS_ORG to org:openai paired with the
  // ai-insight. SPOKE_IN edge to one episode for the episode-count signal.
  return {
    id: 'a1',
    kind: 'gi',
    data: {
      nodes: [
        { id: 'person:alice', type: 'Person', properties: { name: 'Alice', role: 'host' } },
        { id: 'insight:i1', type: 'Insight', properties: {} },
        { id: 'insight:i2', type: 'Insight', properties: {} },
        { id: 'topic:ai', type: 'Topic', properties: { name: 'AI ethics' } },
        { id: 'topic:reg', type: 'Topic', properties: { name: 'AI regulation' } },
        { id: 'org:openai', type: 'Organization', properties: { name: 'OpenAI' } },
        { id: 'ep:001', type: 'Episode', properties: {} },
      ],
      edges: [
        { from: 'insight:i1', to: 'person:alice', type: 'MENTIONS_PERSON' },
        { from: 'insight:i2', to: 'person:alice', type: 'MENTIONS_PERSON' },
        { from: 'insight:i1', to: 'topic:ai', type: 'ABOUT' },
        { from: 'insight:i1', to: 'topic:ai', type: 'ABOUT' }, // duplicate pair → collapses
        { from: 'insight:i2', to: 'topic:reg', type: 'ABOUT' },
        { from: 'insight:i1', to: 'org:openai', type: 'MENTIONS_ORG' },
        { from: 'person:alice', to: 'ep:001', type: 'SPOKE_IN' },
      ],
    },
  } as unknown as ParsedArtifact
}

describe('PersonLandingView — #1048 shell (Person Profile + Position Tracker)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchPositions.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchPersonProfile.mockResolvedValue({ subject: 'person:alice', profile: {} })
    fetchPersonTopics.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchCoSpeakers.mockResolvedValue({ subject: 'person:alice', results: [] })
  })

  async function mountWithArtifact(): Promise<ReturnType<typeof mount>> {
    const w = mount(PersonLandingView, { attachTo: document.body, global: { stubs: STUBS } })
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    // Mirrors TopicEntityView.test.ts injection pattern.
    useArtifactsStore().parsedList = [makeArtifactWithPersonAndTopics()]
    useSubjectStore().focusPerson('person:alice')
    await flushPromises()
    await flushPromises()
    return w
  }

  it('renders the Person Profile + Position Tracker tab pair (no Positions tab)', async () => {
    const w = await mountWithArtifact()
    expect(w.find('[data-testid="person-landing-tab-profile"]').text()).toBe('Person Profile')
    expect(w.find('[data-testid="person-landing-tab-position-tracker"]').text()).toBe(
      'Position Tracker',
    )
    expect(w.find('[data-testid="person-landing-tab-positions"]').exists()).toBe(false)
  })

  it('shows a role badge with the Person.role value', async () => {
    const w = await mountWithArtifact()
    const badge = w.get('[data-testid="person-landing-role"]')
    expect(badge.text()).toBe('Host')
    expect(badge.attributes('data-role')).toBe('host')
  })

  it('renders the episode-count signal from SPOKE_IN edges', async () => {
    const w = await mountWithArtifact()
    expect(w.get('[data-testid="person-landing-episode-count"]').text()).toContain('1 episode')
  })

  it('renders organization chips from MENTIONS_PERSON ∩ MENTIONS_ORG insights', async () => {
    const w = await mountWithArtifact()
    const chips = w.findAll('[data-testid="person-landing-organization-chip"]')
    expect(chips.map((c) => c.text())).toEqual(['OpenAI'])
  })

  it('ranks topics by collapsed ABOUT∩MENTIONS_PERSON pairs (duplicate edges collapse)', async () => {
    const w = await mountWithArtifact()
    const rows = w.findAll('[data-testid="person-landing-ranked-topic-row"]')
    expect(rows.length).toBe(2)
    // Both topics tied at 1 (duplicate ABOUT collapsed); alphabetic tiebreak
    // puts "AI ethics" before "AI regulation".
    expect(rows[0].text()).toContain('AI ethics')
    expect(rows[0].text()).toContain('1')
    expect(rows[1].text()).toContain('AI regulation')
    expect(rows[1].text()).toContain('1')
  })

  it('Position Tracker tab renders the panel with no-topic state when no topic selected', async () => {
    const w = await mountWithArtifact()
    await w.get('[data-testid="person-landing-tab-position-tracker"]').trigger('click')
    expect(w.find('[data-testid="person-landing-panel-position-tracker"]').isVisible()).toBe(true)
    expect(w.find('[data-testid="position-tracker-panel"]').exists()).toBe(true)
    // #1049 state 1 — no Topic selected.
    expect(w.find('[data-testid="position-tracker-no-topic"]').exists()).toBe(true)
  })

  it('clicking a Top Topic row pivots to Position Tracker for that (Person, Topic) pair', async () => {
    const w = await mountWithArtifact()
    // Click the first ranked-topic button (AI ethics, alphabetic tiebreak first).
    const buttons = w.findAll('[data-testid="person-landing-ranked-topic-button"]')
    expect(buttons.length).toBeGreaterThan(0)
    await buttons[0].trigger('click')
    // Tab switched + subject store carries the topic id.
    expect(useSubjectStore().positionTrackerTopicId).toBe('topic:ai')
    expect(w.find('[data-testid="person-landing-panel-position-tracker"]').isVisible()).toBe(true)
    // No-topic placeholder gone; arc state visible (the fixture has matching insights).
    expect(w.find('[data-testid="position-tracker-no-topic"]').exists()).toBe(false)
    expect(w.find('[data-testid="position-tracker-topic-name"]').text()).toBe('AI ethics')
  })
})
