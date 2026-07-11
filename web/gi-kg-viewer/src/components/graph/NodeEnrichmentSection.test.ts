// @vitest-environment happy-dom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import NodeEnrichmentSection from './NodeEnrichmentSection.vue'
import { useShellStore } from '../../stores/shell'
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'

vi.mock('../../composables/useEnrichmentEnvelopeCache', () => ({
  fetchCachedCorpusEnvelope: vi.fn(),
}))

const fetchEnvelope = vi.mocked(fetchCachedCorpusEnvelope)

/** Resolve `temporal_velocity` with one matching topic row, everything else empty. */
function velocityFor(topicId: string) {
  fetchEnvelope.mockImplementation((_root: string, enricher: string) => {
    if (enricher === 'temporal_velocity') {
      return Promise.resolve({
        data: { topics: [{ topic_id: topicId, velocity_last_over_6mo: 2, total: 10 }] },
      } as never)
    }
    return Promise.resolve({ data: { pairs: [] } } as never)
  })
}

async function mountFor(props: { nodeId: string; nodeType: string }) {
  const w = mount(NodeEnrichmentSection, { props })
  useShellStore().corpusPath = '/corpus'
  // The watcher fires immediately, but corpusPath was empty on that first pass;
  // nudge a re-load now that the corpus is set, then flush the async chain.
  await w.setProps({ nodeId: props.nodeId + ' ' }) // change → reload
  await w.setProps({ nodeId: props.nodeId })
  for (let i = 0; i < 6; i++) await w.vm.$nextTick()
  return w
}

/** Last boolean the component emitted on `has-content`. */
function lastHasContent(w: ReturnType<typeof mount>): boolean | undefined {
  const events = w.emitted('has-content') as Array<[boolean]> | undefined
  return events?.at(-1)?.[0]
}

describe('NodeEnrichmentSection — has-content reporting', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchEnvelope.mockReset()
  })

  it('emits has-content=true for a topic that has a velocity signal', async () => {
    velocityFor('topic:ai')
    const w = await mountFor({ nodeId: 'topic:ai', nodeType: 'topic' })
    expect(lastHasContent(w)).toBe(true)
  })

  it('emits has-content=false for a topic with no matching signals', async () => {
    velocityFor('topic:something-else')
    const w = await mountFor({ nodeId: 'topic:ai', nodeType: 'topic' })
    expect(lastHasContent(w)).toBe(false)
  })

  it('emits has-content=false for an unsupported node type', async () => {
    fetchEnvelope.mockResolvedValue({ data: {} } as never)
    const w = await mountFor({ nodeId: 'org:acme', nodeType: 'org' })
    expect(lastHasContent(w)).toBe(false)
  })
})

// ADR-108 — the consensus row shows the two corroborating claims (persisted by
// the topic_consensus enricher), oriented to the focused person.
function consensusEnvelope() {
  fetchEnvelope.mockImplementation((_root: string, enricher: string) => {
    if (enricher === 'topic_consensus') {
      return Promise.resolve({
        data: {
          consensus: [
            {
              topic_id: 'topic:venture-capital',
              person_a_id: 'person:alice',
              person_a_name: 'Alice',
              person_b_id: 'person:bob',
              person_b_name: 'Bob',
              insight_a_text: 'Most VC returns concentrate in a handful of funds.',
              insight_b_text: 'A small number of funds capture the bulk of venture returns.',
            },
          ],
        },
      } as never)
    }
    return Promise.resolve({ data: {} } as never)
  })
}

describe('NodeEnrichmentSection — consensus claims (ADR-108)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchEnvelope.mockReset()
  })

  it('renders both corroborating claims, attributed, oriented to the focused person', async () => {
    consensusEnvelope()
    const w = await mountFor({ nodeId: 'person:alice', nodeType: 'person' })
    const claims = w.find('[data-testid="node-enrichment-consensus-claims"]')
    expect(claims.exists()).toBe(true)
    const text = claims.text()
    // Focused = person_a → self claim is insight_a_text; counterpart Bob's is insight_b_text.
    expect(text).toContain('Alice:')
    expect(text).toContain('Most VC returns concentrate in a handful of funds.')
    expect(text).toContain('Bob:')
    expect(text).toContain('A small number of funds capture the bulk of venture returns.')
  })

  it('flips claim orientation when the counterpart is focused', async () => {
    consensusEnvelope()
    const w = await mountFor({ nodeId: 'person:bob', nodeType: 'person' })
    const claims = w.find('[data-testid="node-enrichment-consensus-claims"]')
    expect(claims.exists()).toBe(true)
    // Focused = person_b → their claim is insight_b_text; the row's counterpart is Alice.
    const text = claims.text()
    expect(text).toContain('Bob:')
    expect(text).toContain('A small number of funds capture the bulk of venture returns.')
    expect(text).toContain('Alice:')
    expect(text).toContain('Most VC returns concentrate in a handful of funds.')
  })
})

// The has-content tests above prove the signal LOADS; these prove the rendered
// VALUE is correct (velocity number + rising tint, grounding %, co-appears order).
describe('NodeEnrichmentSection — renders the signal values', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchEnvelope.mockReset()
  })

  it('renders the velocity value + a rising (emerald) badge for a hot topic', async () => {
    velocityFor('topic:ai') // velocity 2 (> 1.5) → rising
    const w = await mountFor({ nodeId: 'topic:ai', nodeType: 'topic' })
    const badge = w.find('[data-testid="node-enrichment-velocity"]')
    expect(badge.exists()).toBe(true)
    expect(badge.text()).toContain('2.00×')
    expect(badge.text()).toContain('10 mentions')
    // > 1.5 → the "rising" emerald tint (not the neutral/rose class).
    expect(badge.html()).toContain('emerald')
  })

  it('renders the grounding rate as a percentage + grounded/total for a person', async () => {
    fetchEnvelope.mockImplementation((_r: string, enricher: string) => {
      if (enricher === 'grounding_rate')
        return Promise.resolve({
          data: {
            persons: [
              { person_id: 'person:alice', grounded_insights: 17, total_insights: 20, rate: 0.85 },
            ],
          },
        } as never)
      return Promise.resolve({ data: {} } as never)
    })
    const w = await mountFor({ nodeId: 'person:alice', nodeType: 'person' })
    const g = w.find('[data-testid="node-enrichment-grounding"]')
    expect(g.exists()).toBe(true)
    expect(g.text()).toContain('85%')
    expect(g.text()).toContain('17/20 insights grounded')
  })

  it('renders co-appearance chips sorted by shared-episode count for a person', async () => {
    fetchEnvelope.mockImplementation((_r: string, enricher: string) => {
      if (enricher === 'guest_coappearance')
        return Promise.resolve({
          data: {
            pairs: [
              { person_a_id: 'person:alice', person_b_id: 'person:bob', person_b_name: 'Bob', episode_count: 4 },
              { person_a_id: 'person:amy', person_b_id: 'person:alice', person_a_name: 'Amy', episode_count: 9 },
            ],
          },
        } as never)
      return Promise.resolve({ data: {} } as never)
    })
    const w = await mountFor({ nodeId: 'person:alice', nodeType: 'person' })
    const co = w.find('[data-testid="node-enrichment-coappearance"]')
    expect(co.exists()).toBe(true)
    const chips = co.findAll('button')
    // Sorted by episode_count desc: Amy (9) before Bob (4).
    expect(chips[0].text()).toContain('Amy')
    expect(chips[0].text()).toContain('9')
    expect(chips[1].text()).toContain('Bob')
    expect(chips[1].text()).toContain('4')
  })
})
