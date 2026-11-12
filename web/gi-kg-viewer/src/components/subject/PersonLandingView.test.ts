// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import PersonLandingView from './PersonLandingView.vue'
import { invalidateRelationalCache } from '../../composables/useRelationalCache'
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

// FB10 "Appears in shows" loads the episode list; stub it so tests stay offline.
vi.mock('../../api/corpusLibraryApi', () => ({
  fetchCorpusEpisodes: () => Promise.resolve({ items: [] }),
}))

const STUBS = {
  SubjectTimelineChart: { name: 'SubjectTimelineChart', template: '<div data-stub="timeline" />' },
}

function rel(id: string, text: string, type: string) {
  return { id, type, text, show_id: '', episode_id: '' }
}

function makePersonArtifact(nodeId = 'person:alice'): ParsedArtifact {
  return {
    id: 'a1',
    kind: 'gi',
    data: {
      nodes: [
        { id: nodeId, type: 'Person', properties: { name: 'Alice' } },
      ],
      edges: [],
    },
  } as unknown as ParsedArtifact
}

async function mountWith(idOverride = 'person:alice'): Promise<ReturnType<typeof mount>> {
  const personId = idOverride
  const artifacts = useArtifactsStore()
  artifacts.parsedList = [makePersonArtifact(personId)]
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  const subject = useSubjectStore()
  subject.focusPerson(personId)
  const w = mount(PersonLandingView, {
    attachTo: document.body,
    props: { subjectIdOverride: personId },
    global: { stubs: STUBS },
  })
  await flushPromises()
  await flushPromises()
  return w
}

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
    // #1075 chunk 4 — clear the module-level relational panel cache
    // between tests so a prior test's stub response doesn't survive
    // into the next test's mock setup.
    invalidateRelationalCache()
    fetchPositions.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchPersonProfile.mockResolvedValue({ subject: 'person:alice', profile: {} })
    fetchPersonTopics.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchCoSpeakers.mockResolvedValue({ subject: 'person:alice', results: [] })
  })

  async function mountWithArtifact(): Promise<ReturnType<typeof mount>> {
    const personId = 'person:alice'
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    // Mirrors TopicEntityView.test.ts injection pattern.
    useArtifactsStore().parsedList = [makeArtifactWithPersonAndTopics()]
    const subject = useSubjectStore()
    subject.focusPerson(personId)
    const w = mount(PersonLandingView, {
      attachTo: document.body,
      props: { subjectIdOverride: personId },
      global: { stubs: STUBS },
    })
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

// #1050 — UXS-010 Person Profile aggregate sections.
function makeArtifactForPersonProfile(): ParsedArtifact {
  return {
    id: 'a1',
    kind: 'gi',
    data: {
      nodes: [
        { id: 'person:alice', type: 'Person', properties: { name: 'Alice' } },
        { id: 'topic:ai', type: 'Topic', properties: { name: 'AI ethics' } },
        { id: 'topic:reg', type: 'Topic', properties: { name: 'AI regulation' } },
        {
          id: 'insight:i1',
          type: 'Insight',
          properties: { text: 'first', insight_type: 'claim' },
        },
        {
          id: 'insight:i2',
          type: 'Insight',
          properties: { text: 'second', insight_type: 'observation' },
        },
        {
          id: 'insight:i3',
          type: 'Insight',
          properties: { text: 'third', insight_type: 'recommendation' },
        },
        {
          id: 'ep:1',
          type: 'Episode',
          properties: { episode_title: 'Pilot', publish_date: '2026-02-15' },
        },
        {
          id: 'ep:2',
          type: 'Episode',
          properties: { episode_title: 'Sequel', publish_date: '2026-03-10' },
        },
      ],
      edges: [
        { type: 'MENTIONS_PERSON', from: 'insight:i1', to: 'person:alice' },
        { type: 'MENTIONS_PERSON', from: 'insight:i2', to: 'person:alice' },
        { type: 'MENTIONS_PERSON', from: 'insight:i3', to: 'person:alice' },
        { type: 'ABOUT', from: 'insight:i1', to: 'topic:ai' },
        { type: 'ABOUT', from: 'insight:i2', to: 'topic:ai' },
        { type: 'ABOUT', from: 'insight:i3', to: 'topic:reg' },
        { type: 'SPOKE_IN', from: 'person:alice', to: 'ep:1' },
        { type: 'SPOKE_IN', from: 'person:alice', to: 'ep:2' },
      ],
    },
  } as unknown as ParsedArtifact
}

// N2/N3/N6 — the /brief ``topics`` map: the person's insights grouped by topic
// (server-side, corpus-wide). Mirrors makeArtifactForPersonProfile's insights so
// the graph fixture still supplies topic node labels ("AI ethics" / "AI regulation").
function makePersonProfileTopics(): Record<string, unknown[]> {
  return {
    'topic:ai': [
      { insight: { id: 'insight:i1', properties: { text: 'first', insight_type: 'claim' } } },
      { insight: { id: 'insight:i2', properties: { text: 'second', insight_type: 'observation' } } },
    ],
    'topic:reg': [
      { insight: { id: 'insight:i3', properties: { text: 'third', insight_type: 'recommendation' } } },
    ],
  }
}

describe('PersonLandingView — #1050 Person Profile aggregate (PRD-029 / UXS-010)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    // #1075 chunk 4 — clear the module-level relational panel cache
    // between tests so a prior test's stub response doesn't survive
    // into the next test's mock setup.
    invalidateRelationalCache()
    fetchPositions.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchPersonProfile.mockResolvedValue({
      subject: 'person:alice',
      profile: {},
      topics: makePersonProfileTopics(),
    })
    fetchPersonTopics.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchCoSpeakers.mockResolvedValue({ subject: 'person:alice', results: [] })
  })

  async function mountWithProfileArt(): Promise<ReturnType<typeof mount>> {
    const personId = 'person:alice'
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    useArtifactsStore().parsedList = [makeArtifactForPersonProfile()]
    const subject = useSubjectStore()
    subject.focusPerson(personId)
    const w = mount(PersonLandingView, {
      attachTo: document.body,
      props: { subjectIdOverride: personId },
      global: { stubs: STUBS },
    })
    await flushPromises()
    await flushPromises()
    return w
  }

  it('renames the Top Topics heading to "Topics discussed" (UXS-010)', async () => {
    const w = await mountWithProfileArt()
    const heading = w
      .get('[data-testid="person-landing-ranked-topics"]')
      .get('h3')
    expect(heading.text()).toBe('Topics discussed')
  })

  it('renders "Insights voiced" grouped by Topic with one group per Topic', async () => {
    const w = await mountWithProfileArt()
    const groups = w.findAll('[data-testid="person-landing-insights-voiced-group"]')
    expect(groups.length).toBe(2)
    // Order: count desc (AI ethics 2 > AI regulation 1), alphabetic tiebreak.
    const topicNames = groups.map((g) =>
      g.get('[data-testid="person-landing-insights-voiced-topic-button"]').text().split('\n')[0].trim(),
    )
    expect(topicNames[0]).toContain('AI ethics')
    expect(topicNames[1]).toContain('AI regulation')
    // Counts on each header.
    const counts = groups.map((g) =>
      g.get('[data-testid="person-landing-insights-voiced-topic-count"]').text(),
    )
    expect(counts).toEqual(['2', '1'])
  })

  it('Topic group header click pivots to Position Tracker for that pair', async () => {
    const w = await mountWithProfileArt()
    const headers = w.findAll('[data-testid="person-landing-insights-voiced-topic-button"]')
    await headers[1].trigger('click')
    expect(useSubjectStore().positionTrackerTopicId).toBe('topic:reg')
    expect(
      w.get('[data-testid="position-tracker-topic-name"]').text(),
    ).toBe('AI regulation')
  })

  it('insights under a group are collapsed by default and reveal on toggle', async () => {
    const w = await mountWithProfileArt()
    const group = w.findAll('[data-testid="person-landing-insights-voiced-group"]')[0]
    expect(group.find('[data-testid="person-landing-insights-voiced-rows"]').exists()).toBe(false)
    await group.get('[data-testid="person-landing-insights-voiced-toggle"]').trigger('click')
    const rows = group.findAll('[data-testid="person-landing-insights-voiced-row"]')
    expect(rows.length).toBe(2)
    expect(rows[0].get('[data-testid="person-landing-insights-voiced-row-text"]').text()).toBe('first')
  })

})

// New describe block for view prop gating + positions lens
describe('PersonLandingView — view prop gating + positions lens', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    invalidateRelationalCache()
    fetchPositions.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchPersonProfile.mockResolvedValue({
      subject: 'person:alice',
      profile: {},
      topics: makePersonProfileTopics(),
    })
    fetchPersonTopics.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchCoSpeakers.mockResolvedValue({ subject: 'person:alice', results: [] })
  })

  async function mountViewProp(
    view: 'full' | 'profile' | 'positions',
    art?: ParsedArtifact,
  ): Promise<ReturnType<typeof mount>> {
    const personId = 'person:alice'
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    useArtifactsStore().parsedList = [art ?? makeArtifactForPersonProfile()]
    const subject = useSubjectStore()
    subject.focusPerson(personId)
    const w = mount(PersonLandingView, {
      attachTo: document.body,
      props: { subjectIdOverride: personId, view },
      global: { stubs: STUBS },
    })
    await flushPromises()
    await flushPromises()
    return w
  }

  it('view="profile" hides Topics discussed, Insights voiced, and positions sections', async () => {
    const w = await mountViewProp('profile')
    expect(w.find('[data-testid="person-landing-ranked-topics"]').exists()).toBe(false)
    expect(w.find('[data-testid="person-landing-insights-voiced"]').exists()).toBe(false)
    expect(w.find('[data-testid="person-landing-stated"]').exists()).toBe(false)
    expect(w.find('[data-testid="person-landing-positions"]').exists()).toBe(false)
  })

  it('view="profile" carries no GI of its own: episodes, connections and shows are sections', async () => {
    const w = await mountViewProp('profile')
    expect(w.find('[data-testid="person-landing-panel-profile"]').exists()).toBe(true)
    expect(w.find('[data-testid="person-landing-episodes-appeared"]').exists()).toBe(false)
    expect(w.find('[data-testid="person-landing-connections"]').exists()).toBe(false)
    expect(w.find('[data-testid="person-landing-shows"]').exists()).toBe(false)
    expect(w.find('[data-testid="person-landing-episode-count"]').exists()).toBe(false)
    expect(w.find('[data-testid="person-landing-organizations"]').exists()).toBe(false)
  })

  it('view="positions" shows the lens toggle with By topic and All positions buttons', async () => {
    const w = await mountViewProp('positions')
    expect(w.find('[data-testid="person-landing-positions-lens"]').exists()).toBe(true)
    expect(w.find('[data-testid="person-landing-positions-lens-by-topic"]').text()).toBe('By topic')
    expect(w.find('[data-testid="person-landing-positions-lens-all"]').text()).toBe('All positions')
  })

  it('view="positions" default lens shows insights-voiced section (by_topic)', async () => {
    const w = await mountViewProp('positions')
    // by_topic is the default; insights-voiced should be visible
    const byTopic = w.find('[data-testid="person-landing-positions-by-topic"]')
    expect(byTopic.exists()).toBe(true)
    // The insights-voiced section should exist inside
    expect(byTopic.find('[data-testid="person-landing-insights-voiced"]').exists()).toBe(true)
  })

  it('view="positions" switching to all lens shows stated/positions sections', async () => {
    const w = await mountViewProp('positions')
    await w.get('[data-testid="person-landing-positions-lens-all"]').trigger('click')
    const allDiv = w.find('[data-testid="person-landing-positions-all"]')
    expect(allDiv.exists()).toBe(true)
    // positions-empty paragraph since fixture has no SPOKEN_BY edges for alice
    expect(allDiv.find('[data-testid="person-landing-positions-empty"]').exists()).toBe(true)
  })

  it('view="positions" with 12 corpus quotes shows paginator on all lens', async () => {
    // The "All positions" lens paginates the corpus-wide quotes (server /brief),
    // not the client-graph SPOKEN_BY walk — mock 12 quotes on the profile.
    const quotes = Array.from({ length: 12 }, (_v, i) => ({
      quote: { id: `quote:q${i + 1}`, properties: { text: `Quote ${i + 1}` } },
      episode_id: `ep${i + 1}`,
    }))
    fetchPersonProfile.mockResolvedValue({ subject: 'person:alice', quotes })

    const w = await mountViewProp('positions')
    await w.get('[data-testid="person-landing-positions-lens-all"]').trigger('click')
    // Paginator should appear (12 rows > POSITIONS_PAGE_SIZE=10)
    expect(w.find('[data-testid="person-landing-positions-pager"]').exists()).toBe(true)
    // First page shows 10 rows.
    expect(w.findAll('[data-testid="person-landing-corpus-row"]').length).toBe(10)
    // page info shows 1 / 2
    expect(w.get('[data-testid="person-landing-positions-pager-info"]').text()).toBe('1 / 2')
    // click next
    await w.get('[data-testid="person-landing-positions-pager-next"]').trigger('click')
    expect(w.get('[data-testid="person-landing-positions-pager-info"]').text()).toBe('2 / 2')
    expect(w.findAll('[data-testid="person-landing-corpus-row"]').length).toBe(2)
  })
})

describe('PersonLandingView — personProfileSections', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    invalidateRelationalCache()
    fetchPositions.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchPersonProfile.mockResolvedValue({ subject: 'person:alice', profile: {} })
    fetchPersonTopics.mockResolvedValue({ subject: 'person:alice', results: [] })
    fetchCoSpeakers.mockResolvedValue({ subject: 'person:alice', results: [] })
  })

  async function mountView(view: 'full' | 'positions', props: Record<string, unknown> = {}) {
    const { personProfileSections } = await import('../../extensions')
    for (const region of ['header', 'profile-head', 'profile', 'positions', 'full'] as const) {
      personProfileSections.register({
        id: region,
        region,
        component: {
          props: ['personId', 'view'],
          template: `<span data-testid="ext-${region}">{{ personId }}/{{ view }}</span>`,
        },
      })
    }
    const art = makePersonArtifact()
    Object.assign((art.data as { nodes: { properties: Record<string, unknown> }[] }).nodes[0].properties, props)
    useArtifactsStore().parsedList = [art]
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    useSubjectStore().focusPerson('person:alice')
    const w = mount(PersonLandingView, {
      attachTo: document.body,
      props: { subjectIdOverride: 'person:alice', view },
      global: { stubs: STUBS },
    })
    await flushPromises()
    return w
  }

  afterEach(async () => {
    const { personProfileSections } = await import('../../extensions')
    personProfileSections.reset()
  })

  const shown = (w: ReturnType<typeof mount>) =>
    ['header', 'profile-head', 'profile', 'positions', 'full'].filter((r) =>
      w.find(`[data-testid="ext-${r}"]`).exists(),
    )

  it('the full view shows the header, profile-head, profile and full sections, with the person', async () => {
    const w = await mountView('full')
    expect(shown(w)).toEqual(['header', 'profile-head', 'profile', 'full'])
    expect(w.find('[data-testid="ext-header"]').text()).toBe('person:alice/full')
    w.unmount()
  })

  it('a profile-head section renders after the role and before the aliases', async () => {
    const w = await mountView('full', { aliases: ['Al'], role: 'host' })
    const html = w.html()
    expect(html.indexOf('person-landing-role-embedded')).toBeLessThan(html.indexOf('ext-profile-head'))
    expect(html.indexOf('ext-profile-head')).toBeLessThan(html.indexOf('person-landing-aliases'))
    w.unmount()
  })

  it('the positions view shows the header and positions sections', async () => {
    const w = await mountView('positions')
    expect(shown(w)).toEqual(['header', 'positions'])
    w.unmount()
  })
})
