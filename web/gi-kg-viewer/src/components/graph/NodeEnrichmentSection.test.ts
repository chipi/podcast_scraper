// @vitest-environment happy-dom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import NodeEnrichmentSection from './NodeEnrichmentSection.vue'
import { useShellStore } from '../../stores/shell'
import { getCorpusEntitySignals } from '../../api/enrichmentApi'

// The node panel now fetches the lean, server-pre-filtered `/api/corpus/entity-signals` (one call)
// instead of downloading each full corpus envelope. The mock returns the flat `signals` shape
// (enricher -> { list_key: [...] }), already filtered to the focused entity.
vi.mock('../../api/enrichmentApi', () => ({
  getCorpusEntitySignals: vi.fn(),
}))

const getSignals = vi.mocked(getCorpusEntitySignals)

/** Resolve `temporal_velocity` with one matching topic row, everything else empty. */
function velocityFor(topicId: string) {
  getSignals.mockResolvedValue({
    temporal_velocity: { topics: [{ topic_id: topicId, velocity_last_over_6mo: 2, total: 10 }] },
  } as never)
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
    getSignals.mockReset()
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
    getSignals.mockResolvedValue({} as never)
    const w = await mountFor({ nodeId: 'org:acme', nodeType: 'org' })
    expect(lastHasContent(w)).toBe(false)
  })
})

// ADR-108 — the consensus row shows the two corroborating claims (persisted by
// the topic_consensus enricher), oriented to the focused person.
function consensusEnvelope() {
  getSignals.mockResolvedValue({
    topic_consensus: {
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

describe('NodeEnrichmentSection — consensus claims (ADR-108)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    getSignals.mockReset()
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
    getSignals.mockReset()
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

  it('does NOT render a per-person grounding rate — the metric is per-episode (#1927)', async () => {
    // Replaces a test that asserted an 85% badge on a person card. grounding_rate was per-Person
    // and returned exactly 1.0 for all 689 people in the real corpus: an insight is grounded
    // exactly when a supporting quote exists, and the quote carries the speaker, so an ungrounded
    // insight has no speaker and the denominator could only ever equal the numerator. The badge
    // could never have shown anything but 100%. The metric is per-EPISODE now and appears on the
    // Show rail; a person card has nothing to show.
    getSignals.mockResolvedValue({
      grounding_rate: {
        persons: [
          { person_id: 'person:alice', grounded_insights: 17, total_insights: 20, rate: 0.85 },
        ],
      },
    } as never)
    const w = await mountFor({ nodeId: 'person:alice', nodeType: 'person' })
    expect(w.find('[data-testid="node-enrichment-grounding"]').exists()).toBe(false)
  })

  it('renders co-appearance chips sorted by shared-episode count for a person', async () => {
    getSignals.mockResolvedValue({
      guest_coappearance: {
        pairs: [
          { person_a_id: 'person:alice', person_b_id: 'person:bob', person_b_name: 'Bob', episode_count: 4 },
          { person_a_id: 'person:amy', person_b_id: 'person:alice', person_a_name: 'Amy', episode_count: 9 },
        ],
      },
    } as never)
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
