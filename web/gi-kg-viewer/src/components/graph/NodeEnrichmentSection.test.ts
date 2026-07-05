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

// N7 — the contradiction row shows the two opposing claims (persisted by the
// nli_contradiction enricher v1.1.0), oriented to the focused person.
function contradictionEnvelope() {
  fetchEnvelope.mockImplementation((_root: string, enricher: string) => {
    if (enricher === 'nli_contradiction') {
      return Promise.resolve({
        data: {
          contradictions: [
            {
              topic_id: 'topic:venture-capital',
              person_a_id: 'person:alice',
              person_a_name: 'Alice',
              person_b_id: 'person:bob',
              person_b_name: 'Bob',
              insight_a_text: 'VC returns concentrate in a few funds.',
              insight_b_text: 'We built the company without venture capital.',
            },
          ],
        },
      } as never)
    }
    return Promise.resolve({ data: {} } as never)
  })
}

describe('NodeEnrichmentSection — contradiction claims (N7)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchEnvelope.mockReset()
  })

  it('renders both opposing claims, attributed, oriented to the focused person', async () => {
    contradictionEnvelope()
    const w = await mountFor({ nodeId: 'person:alice', nodeType: 'person' })
    const claims = w.find('[data-testid="node-enrichment-contradiction-claims"]')
    expect(claims.exists()).toBe(true)
    const text = claims.text()
    // Focused = person_a → self claim is insight_a_text; counterpart Bob's is insight_b_text.
    expect(text).toContain('Alice:')
    expect(text).toContain('VC returns concentrate in a few funds.')
    expect(text).toContain('Bob:')
    expect(text).toContain('We built the company without venture capital.')
  })

  it('flips claim orientation when the counterpart is focused', async () => {
    contradictionEnvelope()
    const w = await mountFor({ nodeId: 'person:bob', nodeType: 'person' })
    const claims = w.find('[data-testid="node-enrichment-contradiction-claims"]')
    expect(claims.exists()).toBe(true)
    // Focused = person_b → their claim is insight_b_text; the row's counterpart is Alice.
    const text = claims.text()
    expect(text).toContain('Bob:')
    expect(text).toContain('We built the company without venture capital.')
    expect(text).toContain('Alice:')
    expect(text).toContain('VC returns concentrate in a few funds.')
  })
})
