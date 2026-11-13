// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { defineComponent, h } from 'vue'

import type { SearchHit } from '../../api/searchApi'
import { resultCardBadges } from '../../extensions'
import { useSubjectStore } from '../../stores/subject'
import ResultCard from './ResultCard.vue'

// HelpTip is not used by ResultCard, but PodcastCover / heavy children aren't
// either — the card is self-contained, so no stubs are required. Stores are
// REAL (subject / artifacts / graphExplorer), per the reference pattern.

function hitOf(overrides: Partial<SearchHit> = {}): SearchHit {
  return {
    score: 0.1234,
    metadata: {},
    text: 'Some matched passage text.',
    ...overrides,
  } as SearchHit
}

function mountCard(props: { hit: SearchHit; libraryOpensEnabled?: boolean }) {
  return mount(ResultCard, {
    props: { libraryOpensEnabled: false, ...props },
    attachTo: document.body,
  })
}

describe('ResultCard', () => {
  beforeEach(() => setActivePinia(createPinia()))
  afterEach(() => resultCardBadges.reset())

  // --- Basic render --------------------------------------------------------

  it('renders the doc_type on the leading icon, the hit text and the formatted score', () => {
    // 2026-07-22 UX cleanup: doc_type text badge retired; the doc_type
    // lives on the leading icon via ``data-doc-type`` + the letter glyph.
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'insight' }, text: 'Hello world.', score: 0.5 }),
    })
    expect(
      w.get('[data-testid="search-result-tier"]').attributes('data-doc-type'),
    ).toBe('insight')
    expect(w.text()).toContain('Hello world.')
    // score.toFixed(4)
    expect(w.text()).toContain('0.5000')
  })

  it('shows "(no text)" placeholder when text is empty', () => {
    const w = mountCard({ hit: hitOf({ metadata: {}, text: '' }) })
    expect(w.text()).toContain('(no text)')
  })

  it('carries the retrieval-tier label as data-tier on the row icon (2026-07-22 UX)', () => {
    // Text tier badges were retired in favor of a single leading icon
    // (SearchResultRowIcon); the tier label lives on the icon as
    // ``data-tier`` so downstream specs can still assert per-tier.
    const insight = mountCard({ hit: hitOf({ source_tier: 'insight' }) })
    expect(insight.get('[data-testid="search-result-tier"]').attributes('data-tier')).toBe(
      'Insight',
    )

    const segment = mountCard({ hit: hitOf({ source_tier: 'segment' }) })
    expect(segment.get('[data-testid="search-result-tier"]').attributes('data-tier')).toBe(
      'Transcript',
    )

    const aux = mountCard({ hit: hitOf({ source_tier: 'aux' }) })
    expect(aux.get('[data-testid="search-result-tier"]').attributes('data-tier')).toBe(
      'Reference',
    )

    // Unknown tier -> Reference fallback.
    const unknown = mountCard({ hit: hitOf({ source_tier: 'mystery' }) })
    expect(unknown.get('[data-testid="search-result-tier"]').attributes('data-tier')).toBe(
      'Reference',
    )
  })

  it('defaults the tier to Reference when source_tier is absent', () => {
    const w = mountCard({ hit: hitOf({}) })
    expect(w.get('[data-testid="search-result-tier"]').attributes('data-tier')).toBe(
      'Reference',
    )
  })

  // --- Topic cluster summary line ------------------------------------------

  it('renders the topic-cluster summary joining label and parent id', () => {
    const w = mountCard({
      hit: hitOf({
        metadata: {
          topic_cluster: {
            canonical_label: 'Artificial Intelligence',
            graph_compound_parent_id: 'topic:ai',
          },
        },
      }),
    })
    expect(w.text()).toContain('Topic cluster: Artificial Intelligence · topic:ai')
  })

  it('omits the topic-cluster line when metadata has no usable cluster', () => {
    const w = mountCard({ hit: hitOf({ metadata: { topic_cluster: {} } }) })
    expect(w.text()).not.toContain('Topic cluster:')
  })

  // --- Action buttons + emits ----------------------------------------------

  it('renders the G (graph) button for a focusable hit and emits focus on click', async () => {
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'insight', source_id: 'insight:1' } }),
    })
    const g = w.get('button[aria-label="Show on graph"]')
    await g.trigger('click')
    expect(w.emitted('focus')).toHaveLength(1)
    expect((w.emitted('focus')![0][0] as SearchHit).metadata.source_id).toBe('insight:1')
  })

  it('does not render the G button when the hit is not focusable', () => {
    // doc_type not in the focusable set.
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'aux_note', source_id: 'x:1' } }),
    })
    expect(w.find('button[aria-label="Show on graph"]').exists()).toBe(false)
  })

  it('exposes the row as click-to-open (role=button, tabindex=0) and emits open-library on row click', async () => {
    // §S4-shell followup: the L and S standalone buttons are retired; the
    // whole ``<article>`` body is the affordance now. Clicking anywhere
    // outside the inner action buttons opens the Episode subject panel.
    const w = mountCard({
      hit: hitOf({
        metadata: {
          doc_type: 'segment',
          source_metadata_relative_path: 'feed/ep.metadata.json',
        },
      }),
      libraryOpensEnabled: true,
    })
    // The standalone L and S buttons are gone.
    expect(w.find('button[aria-label="Open episode in subject panel"]').exists()).toBe(false)
    expect(w.find('button[aria-label="Episode summary in right panel"]').exists()).toBe(false)
    // The article itself is the affordance.
    const article = w.get('article')
    expect(article.attributes('role')).toBe('button')
    expect(article.attributes('tabindex')).toBe('0')
    expect(article.attributes('aria-label')).toBe('Open episode in subject panel')
    await article.trigger('click')
    expect(w.emitted('open-library')).toHaveLength(1)
    // Retired: no separate open-episode-summary emit.
    expect(w.emitted('open-episode-summary')).toBeUndefined()
  })

  it('keyboard: Enter / Space on the article activates open-library', async () => {
    const w = mountCard({
      hit: hitOf({
        metadata: {
          doc_type: 'segment',
          source_metadata_relative_path: 'feed/ep.metadata.json',
        },
      }),
      libraryOpensEnabled: true,
    })
    const article = w.get('article')
    await article.trigger('keydown', { key: 'Enter' })
    expect(w.emitted('open-library')).toHaveLength(1)
    await article.trigger('keydown', { key: ' ' })
    expect(w.emitted('open-library')).toHaveLength(2)
    // Other keys do NOT fire.
    await article.trigger('keydown', { key: 'Escape' })
    expect(w.emitted('open-library')).toHaveLength(2)
  })

  it('row is NOT click-to-open (no role/tabindex) when libraryOpensEnabled is false', () => {
    const w = mountCard({
      hit: hitOf({
        metadata: { source_metadata_relative_path: 'feed/ep.metadata.json' },
      }),
      libraryOpensEnabled: false,
    })
    const article = w.get('article')
    expect(article.attributes('role')).toBeUndefined()
    expect(article.attributes('tabindex')).toBeUndefined()
    expect(article.attributes('aria-label')).toBeUndefined()
  })

  it('renders the E (episode id) chip when an episode_id is present, but suppresses it for a merged KG surface', () => {
    const withEp = mountCard({
      hit: hitOf({ metadata: { episode_id: 'ep-123' } }),
    })
    const eBtn = withEp.findAll('button').find((b) => b.text() === 'E')
    expect(eBtn).toBeTruthy()

    // KG multi-episode dedupe row -> chip suppressed.
    const merged = mountCard({
      hit: hitOf({
        metadata: { episode_id: 'ep-123', doc_type: 'kg_topic', kg_surface_match_count: 3 },
      }),
    })
    const mergedE = merged.findAll('button').find((b) => b.text() === 'E')
    expect(mergedE).toBeFalsy()
  })

  it('clicking the E chip does not bubble (no emit) — it is a tooltip-only affordance', async () => {
    const w = mountCard({ hit: hitOf({ metadata: { episode_id: 'ep-123' } }) })
    const eBtn = w.findAll('button').find((b) => b.text() === 'E')!
    await eBtn.trigger('click')
    expect(w.emitted('focus')).toBeUndefined()
    expect(w.emitted('open-library')).toBeUndefined()
  })

  it('hides the whole right-chip row when there are no actions and no episode chip', () => {
    const w = mountCard({ hit: hitOf({ metadata: {} }) })
    expect(w.find('.ml-auto').exists()).toBe(false)
  })

  // --- Row default action for kg_topic / kg_entity (2026-07-22 UX cleanup) ---
  //
  // The standalone "Open Topic panel →" / "Open Person panel →" text link
  // was retired; clicking anywhere on the row body now opens the Topic /
  // Person subject panel directly.

  it('a kg_topic row is clickable + focuses the topic subject on row click', async () => {
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'kg_topic', source_id: 'topic:ai' } }),
    })
    const row = w.get('article')
    expect(row.attributes('role')).toBe('button')
    expect(row.attributes('aria-label')).toBe('Open Topic panel')
    const subject = useSubjectStore()
    await row.trigger('click')
    expect(subject.kind).toBe('graph-node')
    expect(subject.graphNodeCyId).toBe('topic:ai')
  })

  it('a kg_entity row is clickable + focuses the person subject on row click', async () => {
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'kg_entity', source_id: 'person:ada' } }),
    })
    const row = w.get('article')
    expect(row.attributes('role')).toBe('button')
    expect(row.attributes('aria-label')).toBe('Open Person panel')
    const subject = useSubjectStore()
    await row.trigger('click')
    expect(subject.kind).toBe('graph-node')
    expect(subject.graphNodeCyId).toBe('person:ada')
  })

  it('treats an id starting with topic: as a topic entity even without a kg_topic doc_type', async () => {
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'insight', source_id: 'topic:foo' } }),
    })
    const row = w.get('article')
    expect(row.attributes('aria-label')).toBe('Open Topic panel')
    const subject = useSubjectStore()
    await row.trigger('click')
    expect(subject.graphNodeCyId).toBe('topic:foo')
  })

  it('a kg_topic hit without source_id and without library metadata is NOT row-clickable', () => {
    const w = mountCard({ hit: hitOf({ metadata: { doc_type: 'kg_topic' } }) })
    const row = w.get('article')
    expect(row.attributes('role')).toBeUndefined()
    expect(row.attributes('aria-label')).toBeUndefined()
  })

  it('the retired "Open Topic panel →" text link no longer renders', () => {
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'kg_topic', source_id: 'topic:ai' } }),
    })
    expect(w.find('[data-testid="search-result-entity-link"]').exists()).toBe(false)
  })

  // --- GI rows are a downstream build's (E04 step 8b) ---------------------

  it('renders no compound badge, lifted insight or supporting quotes of its own', () => {
    const w = mountCard({
      hit: hitOf({
        source_tier: 'segment',
        lifted: { insight: { id: 'insight:42', text: 'A lifted insight body.' } },
        supporting_quotes: [{ text: 'A quote' }],
      }),
    })
    expect(w.find('[data-testid="search-result-compound"]').exists()).toBe(false)
    expect(w.find('[aria-label="Lifted GI insight"]').exists()).toBe(false)
    expect(w.text()).not.toContain('A lifted insight body.')
    expect(w.findAll('button').some((b) => b.text().includes('supporting'))).toBe(false)
  })

  it('renders registered badges after the row icon with the hit', () => {
    resultCardBadges.register({
      id: 'probe',
      component: defineComponent({
        props: { hit: { type: Object, required: true } },
        setup: (props) => () => h('span', { 'data-testid': 'probe-badge' }, String((props.hit as SearchHit).score)),
      }),
    })
    const w = mountCard({ hit: hitOf() })
    const icon = w.get('[data-testid="search-result-tier"]').element
    const badge = w.get('[data-testid="probe-badge"]')
    expect(badge.text()).toBe('0.1234')
    expect(icon.nextElementSibling).toBe(badge.element)
  })
})
