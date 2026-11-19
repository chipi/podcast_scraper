// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h } from 'vue'

import TopicEntityView from './TopicEntityView.vue'
import { topicViewSections } from '../../extensions'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import type { ParsedArtifact, RawGraphEdge, RawGraphNode } from '../../types/artifact'

// The relational layer is only reached through registered sections; nothing here may call it.
const relational = vi.fn()
vi.mock('../../api/relationalApi', () => ({
  fetchCrossShow: relational,
  fetchWhoSaid: relational,
  fetchTopicEntities: relational,
  fetchRelatedTopics: relational,
}))

// SubjectTimelineChart wraps Chart.js (canvas) — stub to a passthrough.
const STUBS = {
  SubjectTimelineChart: { name: 'SubjectTimelineChart', template: '<div data-stub="timeline-chart" />' },
}

function artifactOf(nodes: RawGraphNode[], edges: RawGraphEdge[] = []): ParsedArtifact {
  return {
    name: 'kg',
    kind: 'kg',
    episodeId: null,
    nodes: nodes.length,
    edges: edges.length,
    nodeTypes: {},
    data: { nodes, edges },
  }
}

/** Topic + one dated Insight + one Quote linked via ABOUT, plus the Episode. */
function topicWithMentions(): ParsedArtifact {
  return artifactOf(
    [
      {
        id: 'topic:ai',
        type: 'Topic',
        properties: {
          label: 'Artificial Intelligence',
          aliases: ['AI', 'ML'],
          description: 'The study of machine intelligence.',
        },
      },
      {
        id: 'insight:1',
        type: 'Insight',
        properties: { text: 'AI is advancing fast.', episode_id: 'ep-1' },
      },
      { id: 'quote:1', type: 'Quote', properties: { text: 'A quote about AI.', episode_id: 'ep-1' } },
      {
        id: '__unified_ep__:ep-1',
        type: 'Episode',
        properties: { episode_title: 'Episode One', publish_date: '2026-01-15' },
      },
    ],
    [
      { from: 'topic:ai', to: 'insight:1', type: 'ABOUT' },
      { from: 'topic:ai', to: 'quote:1', type: 'MENTIONS' },
    ],
  )
}

async function mountTopic(
  art: ParsedArtifact | null,
  topicId = 'topic:ai',
  opts: { health?: boolean } = {},
) {
  // TopicEntityView is now embedded-only (the standalone rail was retired); it
  // takes its subject id from the ``subjectIdOverride`` prop rather than the
  // subject store (``focusTopic`` now opens the unified NodeDetail node view).
  // Seed the artifacts + corpus path BEFORE mount so the immediate relational
  // fetch (watch subjectId, {immediate:true}) sees the root on first run.
  const artifacts = useArtifactsStore()
  if (art) artifacts.parsedList = [art]
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = opts.health === false ? null : 'ok'
  const subject = useSubjectStore()
  const w = mount(TopicEntityView, {
    attachTo: document.body,
    props: { subjectIdOverride: topicId },
    global: { stubs: STUBS },
  })
  for (let i = 0; i < 8; i++) await w.vm.$nextTick()
  return { w, artifacts, shell, subject }
}

describe('TopicEntityView.vue', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('renders the topic kind label, name, aliases and description', async () => {
    const { w } = await mountTopic(topicWithMentions())
    expect(w.get('[data-testid="topic-entity-view-kind"]').text()).toBe('Topic')
    expect(w.get('[data-testid="topic-entity-view-name"]').text()).toBe('Artificial Intelligence')
    expect(w.get('[data-testid="topic-entity-view-aliases"]').text()).toContain('AI, ML')
    expect(w.get('[data-testid="topic-entity-view-description"]').text()).toContain(
      'machine intelligence',
    )
  })

  it('falls back to the raw subject id when the node is not in the loaded slice', async () => {
    const { w } = await mountTopic(artifactOf([]), 'topic:missing')
    expect(w.get('[data-testid="topic-entity-view-name"]').text()).toBe('topic:missing')
    // No aliases / description rows for an unresolved subject.
    expect(w.find('[data-testid="topic-entity-view-aliases"]').exists()).toBe(false)
  })

  it('labels a Person/Entity node as "Entity"', async () => {
    const art = artifactOf([
      { id: 'person:ada', type: 'Person', properties: { name: 'Ada Lovelace' } },
    ])
    const { w } = await mountTopic(art, 'person:ada')
    expect(w.get('[data-testid="topic-entity-view-kind"]').text()).toBe('Entity')
    expect(w.get('[data-testid="topic-entity-view-name"]').text()).toBe('Ada Lovelace')
  })

  it('labels an Organization node as "Entity" (RFC-097 v3.0 typed node)', async () => {
    /** Organization is a first-class node type per RFC-097 v3.0. The shared
     * Person/Entity rail handles both — the kind chip should say "Entity"
     * (not the default "Subject" fallback). Regression guard against the
     * subjectKindLabel computed property dropping Organization.
     */
    const art = artifactOf([
      { id: 'org:acme', type: 'Organization', properties: { name: 'Acme Corp' } },
    ])
    const { w } = await mountTopic(art, 'org:acme')
    expect(w.get('[data-testid="topic-entity-view-kind"]').text()).toBe('Entity')
    expect(w.get('[data-testid="topic-entity-view-name"]').text()).toBe('Acme Corp')
  })

  // NOTE: the graph-scoped mentions list/stats/empty-state were removed when the
  // node view merged to ONE corpus-wide "Mentions" section (the API timeline).
  // Typed-MENTIONS-edge resolution stays covered in subjectMentionsTimeline.test.ts.

  it('emits goGraph when "Open in graph" is clicked', async () => {
    const { w } = await mountTopic(topicWithMentions())
    await w.get('[data-testid="topic-entity-view-go-graph"]').trigger('click')
    expect(w.emitted('goGraph')).toHaveLength(1)
  })

  it('emits prefillSemanticSearch with the subject name', async () => {
    const { w } = await mountTopic(topicWithMentions())
    await w.get('[data-testid="topic-entity-view-prefill-search"]').trigger('click')
    expect(w.emitted('prefillSemanticSearch')![0]).toEqual([{ query: 'Artificial Intelligence' }])
  })

  // ── Relational sections (FR4.2) ─────────────────────────────────────────────

  // ── Topic view sections ─────────────────────────────────────────────────────

  describe('topicViewSections', () => {
    afterEach(() => topicViewSections.reset())

    it('renders a section after the description with the id it was given, and calls no relational route', async () => {
      topicViewSections.register({
        id: 's',
        region: 'after-description',
        component: defineComponent({
          props: ['topicId'],
          setup: (p) => () => h('span', { 'data-testid': 'ext-topic' }, String(p.topicId)),
        }),
      })
      const { w } = await mountTopic(topicWithMentions(), 'g:topic:ai')
      expect(w.get('[data-testid="ext-topic"]').text()).toBe('g:topic:ai')
      const html = w.html()
      expect(html.indexOf('topic-entity-view-description')).toBeLessThan(html.indexOf('ext-topic'))
      expect(relational).not.toHaveBeenCalled()
    })

    it('renders nothing extra when nothing is registered', async () => {
      const { w } = await mountTopic(topicWithMentions())
      expect(w.find('[data-testid^="tev-"]').exists()).toBe(false)
      expect(relational).not.toHaveBeenCalled()
    })
  })
})
