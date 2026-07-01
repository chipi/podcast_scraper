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
