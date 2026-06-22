// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import TopicEntityView from './TopicEntityView.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import type { ParsedArtifact, RawGraphEdge, RawGraphNode } from '../../types/artifact'

// ── API mocks ────────────────────────────────────────────────────────────────
// The relational PRD-033 FR4.2 fetches (cross-show / who-said / topic-entities)
// are mocked; the timeline + mention list come from the in-memory graph slice.
const fetchCrossShow = vi.fn()
const fetchWhoSaid = vi.fn()
const fetchTopicEntities = vi.fn()
const fetchRelatedTopics = vi.fn()
vi.mock('../../api/relationalApi', () => ({
  fetchCrossShow: (...a: unknown[]) => fetchCrossShow(...a),
  fetchWhoSaid: (...a: unknown[]) => fetchWhoSaid(...a),
  fetchTopicEntities: (...a: unknown[]) => fetchTopicEntities(...a),
  fetchRelatedTopics: (...a: unknown[]) => fetchRelatedTopics(...a),
}))

// SubjectTimelineChart wraps Chart.js (canvas) — stub to a passthrough.
const STUBS = {
  SubjectTimelineChart: { name: 'SubjectTimelineChart', template: '<div data-stub="timeline-chart" />' },
}

function artifactOf(nodes: RawGraphNode[], edges: RawGraphEdge[] = []): ParsedArtifact {
  return {
    name: 'gi',
    kind: 'gi',
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
  const w = mount(TopicEntityView, { attachTo: document.body, global: { stubs: STUBS } })
  const artifacts = useArtifactsStore()
  if (art) artifacts.parsedList = [art]
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = opts.health === false ? null : 'ok'
  const subject = useSubjectStore()
  subject.focusTopic(topicId)
  for (let i = 0; i < 8; i++) await w.vm.$nextTick()
  return { w, artifacts, shell, subject }
}

describe('TopicEntityView.vue', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    fetchCrossShow.mockResolvedValue({ subject: 'topic:ai', groups: {} })
    fetchWhoSaid.mockResolvedValue({ subject: 'topic:ai', groups: {} })
    fetchTopicEntities.mockResolvedValue({ subject: 'topic:ai', results: [] })
    fetchRelatedTopics.mockResolvedValue({ subject: 'topic:ai', results: [] })
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

  it('computes the mentions stats line from the linked insight + quote', async () => {
    const { w } = await mountTopic(topicWithMentions())
    const stats = w.get('[data-testid="topic-entity-view-stats"]').text()
    expect(stats).toContain('2 dated mentions')
    expect(stats).toContain('1 insight')
    expect(stats).toContain('1 quote')
  })

  it('renders the mentions list with type chips and episode metadata', async () => {
    const { w } = await mountTopic(topicWithMentions())
    const list = w.get('[data-testid="topic-entity-view-mentions"]')
    expect(list.findAll('li')).toHaveLength(2)
    expect(list.text()).toContain('AI is advancing fast.')
    expect(list.text()).toContain('A quote about AI.')
    expect(list.text()).toContain('Episode One')
    expect(list.text()).toContain('2026-01-15')
  })

  it('renders the empty state when nothing links to the subject', async () => {
    const art = artifactOf([{ id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } }])
    const { w } = await mountTopic(art)
    expect(w.find('[data-testid="topic-entity-view-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="topic-entity-view-mentions"]').exists()).toBe(false)
  })

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

  it('renders cross-show rows from the relational layer', async () => {
    fetchCrossShow.mockResolvedValue({
      subject: 'topic:ai',
      groups: {
        'podcast:show-a': [
          { id: 'i:9', type: 'Insight', text: 'Show A take on AI.', show_id: 'show-a', episode_id: 'e' },
        ],
      },
    })
    const { w } = await mountTopic(topicWithMentions())
    expect(w.find('[data-testid="tev-cross-show"]').exists()).toBe(true)
    expect(w.findAll('[data-testid="tev-cross-show-row"]')).toHaveLength(1)
    expect(w.text()).toContain('Show A take on AI.')
  })

  it('renders who-said voices and focuses the person on click', async () => {
    fetchWhoSaid.mockResolvedValue({
      subject: 'topic:ai',
      groups: {
        'person:ada': [
          { id: 'i:1', type: 'Insight', text: 'Ada on AI.', show_id: 's', episode_id: 'e' },
        ],
      },
    })
    const { w, subject } = await mountTopic(topicWithMentions())
    const voice = w.get('[data-testid="tev-voice-link"]')
    expect(voice.text()).toContain('ada')
    await voice.trigger('click')
    expect(subject.kind).toBe('person')
    expect(subject.personId).toBe('person:ada')
  })

  it('renders entity chips and focuses an entity on click', async () => {
    fetchTopicEntities.mockResolvedValue({
      subject: 'topic:ai',
      results: [{ id: 'org:openai', type: 'org', text: 'OpenAI', show_id: '', episode_id: '' }],
    })
    const { w, subject } = await mountTopic(topicWithMentions())
    const chip = w.get('[data-testid="tev-entity-chip"]')
    expect(chip.text()).toBe('OpenAI')
    await chip.trigger('click')
    expect(subject.kind).toBe('topic') // focusEntity aliases focusTopic
    expect(subject.topicId).toBe('org:openai')
  })

  it('surfaces a relational error in the cross-show section', async () => {
    fetchCrossShow.mockResolvedValue({ subject: 'topic:ai', groups: {}, error: 'relational boom' })
    const { w } = await mountTopic(topicWithMentions())
    expect(w.find('[data-testid="tev-cross-show"]').text()).toContain('relational boom')
  })

  it('skips relational fetches when health is unset', async () => {
    await mountTopic(topicWithMentions(), 'topic:ai', { health: false })
    expect(fetchCrossShow).not.toHaveBeenCalled()
    expect(fetchWhoSaid).not.toHaveBeenCalled()
    expect(fetchTopicEntities).not.toHaveBeenCalled()
    expect(fetchRelatedTopics).not.toHaveBeenCalled()
  })

  it('renders related topics as chips (#1055)', async () => {
    fetchRelatedTopics.mockResolvedValue({
      subject: 'topic:ai',
      results: [
        { id: 'topic:ml', type: 'topic', text: 'Machine Learning', show_id: '', episode_id: '' },
        { id: 'topic:safety', type: 'topic', text: 'AI safety', show_id: '', episode_id: '' },
      ],
    })
    const { w } = await mountTopic(topicWithMentions())
    const chips = w.findAll('[data-testid="tev-related-topic-chip"]')
    expect(chips.map((c) => c.text())).toEqual(['Machine Learning', 'AI safety'])
    expect(fetchRelatedTopics).toHaveBeenCalledWith('/corpus', 'topic:ai')
  })

  it('hides the related-topics section when empty', async () => {
    const { w } = await mountTopic(topicWithMentions())
    expect(w.find('[data-testid="tev-related-topics"]').exists()).toBe(false)
  })

  // ---------------------------------------------------------------------------
  // RFC-097 v3.0 typed MENTIONS family — when the subject is a Person or an
  // Organization, the mentions list must surface insights linked via the
  // typed variants (MENTIONS_PERSON / MENTIONS_ORG), not just legacy MENTIONS.
  // ---------------------------------------------------------------------------

  it('Person subject surfaces insights linked via typed MENTIONS_PERSON', async () => {
    const art = artifactOf(
      [
        {
          id: 'person:ada',
          type: 'Person',
          properties: { name: 'Ada Lovelace' },
        },
        {
          id: 'insight:typed',
          type: 'Insight',
          properties: { text: 'Ada championed analytical engines.', episode_id: 'ep-x' },
        },
        {
          id: '__unified_ep__:ep-x',
          type: 'Episode',
          properties: { episode_title: 'Computing Pioneers', publish_date: '2026-04-10' },
        },
      ],
      // Typed variant (RFC-097 v3.0). Pre-v3 corpora would have used the
      // legacy generic MENTIONS — the test below covers that fallback.
      [{ from: 'insight:typed', to: 'person:ada', type: 'MENTIONS_PERSON' }],
    )
    const { w } = await mountTopic(art, 'person:ada')
    const list = w.get('[data-testid="topic-entity-view-mentions"]')
    expect(list.text()).toContain('Ada championed analytical engines.')
    expect(list.text()).toContain('Computing Pioneers')
    expect(list.text()).toContain('2026-04-10')
  })

  it('Organization subject surfaces insights linked via typed MENTIONS_ORG', async () => {
    const art = artifactOf(
      [
        {
          id: 'org:acme',
          type: 'Organization',
          properties: { name: 'Acme Corp' },
        },
        {
          id: 'insight:org',
          type: 'Insight',
          properties: { text: 'Acme delivered Q3 outperformance.', episode_id: 'ep-y' },
        },
        {
          id: '__unified_ep__:ep-y',
          type: 'Episode',
          properties: { episode_title: 'Earnings Recap', publish_date: '2026-05-22' },
        },
      ],
      [{ from: 'insight:org', to: 'org:acme', type: 'MENTIONS_ORG' }],
    )
    const { w } = await mountTopic(art, 'org:acme')
    const list = w.get('[data-testid="topic-entity-view-mentions"]')
    expect(list.text()).toContain('Acme delivered Q3 outperformance.')
    expect(list.text()).toContain('Earnings Recap')
    expect(list.text()).toContain('2026-05-22')
  })

  it('mid-migration corpus: subject linked by BOTH typed AND legacy MENTIONS surfaces both insights', async () => {
    /**
     * Reflects a real-world mid-migration shape: one insight uses the
     * typed variant, another still carries the legacy generic. The
     * mentions list MUST surface both — the typed family is one
     * semantic unit per the search-layer contract (see
     * ``relational_queries._MENTIONS_FAMILY``).
     */
    const art = artifactOf(
      [
        {
          id: 'person:linus',
          type: 'Person',
          properties: { name: 'Linus' },
        },
        {
          id: 'insight:typed',
          type: 'Insight',
          properties: { text: 'Linus advocates strict review.', episode_id: 'ep-a' },
        },
        {
          id: 'insight:legacy',
          type: 'Insight',
          properties: {
            text: 'Linus mentioned a deprecated API.',
            episode_id: 'ep-b',
          },
        },
        {
          id: '__unified_ep__:ep-a',
          type: 'Episode',
          properties: { episode_title: 'EpA', publish_date: '2026-06-01' },
        },
        {
          id: '__unified_ep__:ep-b',
          type: 'Episode',
          properties: { episode_title: 'EpB', publish_date: '2026-06-02' },
        },
      ],
      [
        { from: 'insight:typed', to: 'person:linus', type: 'MENTIONS_PERSON' },
        { from: 'insight:legacy', to: 'person:linus', type: 'MENTIONS' },
      ],
    )
    const { w } = await mountTopic(art, 'person:linus')
    const list = w.get('[data-testid="topic-entity-view-mentions"]')
    const items = list.findAll('li')
    expect(items.length).toBe(2)
    expect(list.text()).toContain('Linus advocates strict review.')
    expect(list.text()).toContain('Linus mentioned a deprecated API.')
  })
})
