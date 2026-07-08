// @vitest-environment happy-dom
//
// #1049 — PositionTrackerPanel covers the three UXS-009 states:
// no Topic selected, Topic selected but zero arc rows, and Topic selected
// with arc rows + insight_type filter chips.
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// ADR-108 sentiment tint fetches the corpus-wide position arc; mock it (default: no sentiment,
// so the existing UXS-009 tests are unaffected — cards fall back to the neutral tint).
const fetchPersonPositions = vi.fn()
vi.mock('../../api/cilApi', () => ({
  fetchPersonPositions: (...a: unknown[]) => fetchPersonPositions(...a),
}))

import PositionTrackerPanel from './PositionTrackerPanel.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import type { ParsedArtifact } from '../../types/artifact'

function makeArc(): ParsedArtifact {
  return {
    id: 'a1',
    kind: 'gi',
    data: {
      nodes: [
        { id: 'person:alice', type: 'Person', properties: { name: 'Alice' } },
        { id: 'topic:ai', type: 'Topic', properties: { name: 'AI ethics' } },
        {
          id: 'i_claim',
          type: 'Insight',
          properties: {
            text: 'We must regulate AI.',
            insight_type: 'claim',
            position_hint: 0.4,
          },
        },
        {
          id: 'i_obs',
          type: 'Insight',
          properties: {
            text: 'AI labs trail in red-teaming.',
            insight_type: 'observation',
            position_hint: 0.7,
          },
        },
        {
          id: 'ep:1',
          type: 'Episode',
          properties: { publish_date: '2026-02-01' },
        },
        { id: 'q1', type: 'Quote', properties: { text: 'we should write the rules now' } },
      ],
      edges: [
        { type: 'MENTIONS_PERSON', from: 'i_claim', to: 'person:alice' },
        { type: 'MENTIONS_PERSON', from: 'i_obs', to: 'person:alice' },
        { type: 'ABOUT', from: 'i_claim', to: 'topic:ai' },
        { type: 'ABOUT', from: 'i_obs', to: 'topic:ai' },
        { type: 'IN_EPISODE', from: 'i_claim', to: 'ep:1' },
        { type: 'IN_EPISODE', from: 'i_obs', to: 'ep:1' },
        { type: 'SUPPORTED_BY', from: 'i_claim', to: 'q1' },
      ],
    },
  } as unknown as ParsedArtifact
}

async function mountPanel(opts?: {
  topic?: string | null
  withArtifact?: boolean
  personId?: string
}): Promise<ReturnType<typeof mount>> {
  const personId = opts?.personId ?? 'person:alice'
  const artifacts = useArtifactsStore()
  if (opts?.withArtifact !== false) {
    artifacts.parsedList = [makeArc()]
  }
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  const subject = useSubjectStore()
  subject.focusPerson(personId)
  const w = mount(PositionTrackerPanel, {
    attachTo: document.body,
    props: { personIdOverride: personId },
  })
  if (opts?.topic !== undefined && opts?.topic !== null) {
    subject.selectTopicForPositionTracker(opts.topic)
  }
  await flushPromises()
  return w
}

describe('PositionTrackerPanel (#1049)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchPersonPositions.mockReset()
    fetchPersonPositions.mockResolvedValue({ path: '/c', person_id: '', topic_id: '', episodes: [] })
  })

  it('ADR-108: tints each arc card by its insight sentiment (best-effort join)', async () => {
    fetchPersonPositions.mockResolvedValue({
      path: '/c',
      person_id: 'person:alice',
      topic_id: 'topic:ai',
      episodes: [
        {
          episode_id: 'ep:1',
          publish_date: '2026-02-01',
          insights: [
            { id: 'i_claim', sentiment: { compound: 0.6, label: 'positive' } },
            { id: 'i_obs', sentiment: { compound: -0.5, label: 'negative' } },
          ],
        },
      ],
    })
    const w = await mountPanel({ topic: 'topic:ai' })
    await flushPromises()
    const rows = w.findAll('[data-testid="position-tracker-row"]')
    const claim = rows.find((r) => r.attributes('data-insight-type') === 'claim')
    const obs = rows.find((r) => r.attributes('data-insight-type') === 'observation')
    expect(claim?.classes()).toContain('bg-emerald-900/10') // positive
    expect(obs?.classes()).toContain('bg-rose-900/10') // negative
  })

  it('UXS-009 state 1: no Topic selected → placeholder copy', async () => {
    const w = await mountPanel()
    expect(w.find('[data-testid="position-tracker-no-topic"]').exists()).toBe(true)
    expect(w.find('[data-testid="position-tracker-arc"]').exists()).toBe(false)
  })

  it('UXS-009 state 2: Topic selected, zero arc rows → informative empty state', async () => {
    // Wrong topic id means the intersection is empty.
    const w = await mountPanel({ topic: 'topic:unknown' })
    expect(w.find('[data-testid="position-tracker-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="position-tracker-arc"]').exists()).toBe(false)
  })

  it('UXS-009 state 3: populated arc with rows ordered by date then position_hint', async () => {
    const w = await mountPanel({ topic: 'topic:ai' })
    const arc = w.get('[data-testid="position-tracker-arc"]')
    expect(arc.exists()).toBe(true)
    expect(w.get('[data-testid="position-tracker-topic-name"]').text()).toBe('AI ethics')
    const rows = w.findAll('[data-testid="position-tracker-row"]')
    expect(rows.length).toBe(2)
    // 0.4 sorts before 0.7 within the same date.
    expect(rows[0].get('[data-testid="position-tracker-row-text"]').text()).toBe(
      'We must regulate AI.',
    )
    expect(rows[1].get('[data-testid="position-tracker-row-text"]').text()).toBe(
      'AI labs trail in red-teaming.',
    )
  })

  it('insight_type filter chips narrow the visible rows', async () => {
    const w = await mountPanel({ topic: 'topic:ai' })
    expect(w.findAll('[data-testid="position-tracker-row"]').length).toBe(2)
    // Activate the "claim" filter.
    await w.get('[data-testid="position-tracker-filter-claim"]').trigger('click')
    const rows = w.findAll('[data-testid="position-tracker-row"]')
    expect(rows.length).toBe(1)
    expect(rows[0].attributes('data-insight-type')).toBe('claim')
  })

  it('filter that matches no rows surfaces a filter-empty hint without losing the chips', async () => {
    const w = await mountPanel({ topic: 'topic:ai' })
    await w.get('[data-testid="position-tracker-filter-question"]').trigger('click')
    expect(w.find('[data-testid="position-tracker-filter-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="position-tracker-filters"]').exists()).toBe(true)
  })

  it('switching to a new (Person, Topic) pair resets active filters', async () => {
    const w = await mountPanel({ topic: 'topic:ai' })
    await w.get('[data-testid="position-tracker-filter-claim"]').trigger('click')
    // Clear + reselect to simulate a different pair landing.
    const subject = useSubjectStore()
    subject.clearPositionTrackerTopic()
    await flushPromises()
    subject.selectTopicForPositionTracker('topic:ai')
    await flushPromises()
    // The chip is no longer pressed.
    expect(
      w.get('[data-testid="position-tracker-filter-claim"]').attributes('aria-pressed'),
    ).toBe('false')
  })

  it('Clear button returns the panel to the no-topic state', async () => {
    const w = await mountPanel({ topic: 'topic:ai' })
    expect(w.find('[data-testid="position-tracker-arc"]').exists()).toBe(true)
    await w.get('[data-testid="position-tracker-clear-topic"]').trigger('click')
    expect(useSubjectStore().positionTrackerTopicId).toBeNull()
    expect(w.find('[data-testid="position-tracker-no-topic"]').exists()).toBe(true)
  })

  it('renders supporting quote excerpts when SUPPORTED_BY exists', async () => {
    const w = await mountPanel({ topic: 'topic:ai' })
    const quotes = w.findAll('[data-testid="position-tracker-row-quote"]')
    expect(quotes.length).toBe(1)
    expect(quotes[0].text()).toBe('we should write the rules now')
  })

  it('insight_type filter matches case-insensitively (defensive vs capitalized artifact values)', async () => {
    // Override the default artifact: same shape but insight_type capitalized.
    setActivePinia(createPinia())
    const personId = 'person:a'
    useArtifactsStore().parsedList = [
      {
        id: 'caps',
        kind: 'gi',
        data: {
          nodes: [
            { id: 'person:a', type: 'Person', properties: { name: 'A' } },
            { id: 'topic:t', type: 'Topic', properties: { name: 'T' } },
            {
              id: 'i_caps',
              type: 'Insight',
              properties: { text: 'caps', insight_type: 'Claim', position_hint: 0.1 },
            },
            { id: 'ep', type: 'Episode', properties: { publish_date: '2026-04-01' } },
          ],
          edges: [
            { type: 'MENTIONS_PERSON', from: 'i_caps', to: 'person:a' },
            { type: 'ABOUT', from: 'i_caps', to: 'topic:t' },
            { type: 'IN_EPISODE', from: 'i_caps', to: 'ep' },
          ],
        },
      } as unknown as ParsedArtifact,
    ]
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    const subject = useSubjectStore()
    subject.focusPerson(personId)
    const w = mount(PositionTrackerPanel, {
      attachTo: document.body,
      props: { personIdOverride: personId },
    })
    subject.selectTopicForPositionTracker('topic:t')
    await flushPromises()
    // Row visible by default.
    expect(w.findAll('[data-testid="position-tracker-row"]').length).toBe(1)
    // Click the (lowercase-labelled) claim chip — capitalized row still matches.
    await w.get('[data-testid="position-tracker-filter-claim"]').trigger('click')
    expect(w.findAll('[data-testid="position-tracker-row"]').length).toBe(1)
  })
})
