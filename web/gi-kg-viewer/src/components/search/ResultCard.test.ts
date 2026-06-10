// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { SearchHit } from '../../api/searchApi'
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

  // --- Basic render --------------------------------------------------------

  it('renders the doc_type, the hit text and the formatted score', () => {
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'insight' }, text: 'Hello world.', score: 0.5 }),
    })
    expect(w.text()).toContain('insight')
    expect(w.text()).toContain('Hello world.')
    // score.toFixed(4)
    expect(w.text()).toContain('0.5000')
  })

  it('falls back to "?" doc_type and "(no text)" when text is empty', () => {
    const w = mountCard({ hit: hitOf({ metadata: {}, text: '' }) })
    expect(w.find('.font-mono.text-primary').text()).toBe('?')
    expect(w.text()).toContain('(no text)')
  })

  it('renders the retrieval-tier badge label per source_tier', () => {
    const insight = mountCard({ hit: hitOf({ source_tier: 'insight' }) })
    expect(insight.get('[data-testid="search-result-tier"]').text()).toBe('Insight')

    const segment = mountCard({ hit: hitOf({ source_tier: 'segment' }) })
    expect(segment.get('[data-testid="search-result-tier"]').text()).toBe('Transcript')

    const aux = mountCard({ hit: hitOf({ source_tier: 'aux' }) })
    expect(aux.get('[data-testid="search-result-tier"]').text()).toBe('Reference')

    // Unknown tier -> Reference fallback.
    const unknown = mountCard({ hit: hitOf({ source_tier: 'mystery' }) })
    expect(unknown.get('[data-testid="search-result-tier"]').text()).toBe('Reference')
  })

  it('defaults the tier to Reference when source_tier is absent', () => {
    const w = mountCard({ hit: hitOf({}) })
    expect(w.get('[data-testid="search-result-tier"]').text()).toBe('Reference')
  })

  // --- Compound (segment + lifted) badge -----------------------------------

  it('shows the "+ insight" compound badge only for a segment hit with a lifted object', () => {
    const compound = mountCard({
      hit: hitOf({ source_tier: 'segment', lifted: { insight: { text: 'x' } } }),
    })
    expect(compound.find('[data-testid="search-result-compound"]').exists()).toBe(true)

    // Same lifted, but aux tier -> not compound.
    const notSegment = mountCard({
      hit: hitOf({ source_tier: 'aux', lifted: { insight: { text: 'x' } } }),
    })
    expect(notSegment.find('[data-testid="search-result-compound"]').exists()).toBe(false)

    // Segment with no lifted -> not compound.
    const noLift = mountCard({ hit: hitOf({ source_tier: 'segment', lifted: null }) })
    expect(noLift.find('[data-testid="search-result-compound"]').exists()).toBe(false)
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

  it('renders L + S buttons and emits open-library / open-episode-summary when library opens are enabled', async () => {
    const w = mountCard({
      hit: hitOf({
        metadata: {
          doc_type: 'segment',
          source_metadata_relative_path: 'feed/ep.metadata.json',
        },
      }),
      libraryOpensEnabled: true,
    })
    const l = w.get('button[aria-label="Open episode in subject panel"]')
    await l.trigger('click')
    expect(w.emitted('open-library')).toHaveLength(1)

    const s = w.get('button[aria-label="Episode summary in right panel"]')
    await s.trigger('click')
    expect(w.emitted('open-episode-summary')).toHaveLength(1)
  })

  it('hides L / S when libraryOpensEnabled is false even with a metadata path', () => {
    const w = mountCard({
      hit: hitOf({
        metadata: { source_metadata_relative_path: 'feed/ep.metadata.json' },
      }),
      libraryOpensEnabled: false,
    })
    expect(w.find('button[aria-label="Open episode in subject panel"]').exists()).toBe(false)
    expect(w.find('button[aria-label="Episode summary in right panel"]').exists()).toBe(false)
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

  // --- Entity link (kg_topic / kg_entity) ----------------------------------

  it('renders an "Open Topic panel" link for a kg_topic hit and focuses the topic subject', async () => {
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'kg_topic', source_id: 'topic:ai' } }),
    })
    const link = w.get('[data-testid="search-result-entity-link"]')
    expect(link.text()).toContain('Open Topic panel')
    const subject = useSubjectStore()
    await link.trigger('click')
    expect(subject.kind).toBe('topic')
    expect(subject.topicId).toBe('topic:ai')
  })

  it('renders an "Open Person panel" link for a kg_entity hit and focuses the person subject', async () => {
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'kg_entity', source_id: 'person:ada' } }),
    })
    const link = w.get('[data-testid="search-result-entity-link"]')
    expect(link.text()).toContain('Open Person panel')
    const subject = useSubjectStore()
    await link.trigger('click')
    expect(subject.kind).toBe('person')
    expect(subject.personId).toBe('person:ada')
  })

  it('treats an id starting with topic: as a topic entity even without a kg_topic doc_type', () => {
    const w = mountCard({
      hit: hitOf({ metadata: { doc_type: 'insight', source_id: 'topic:foo' } }),
    })
    expect(w.get('[data-testid="search-result-entity-link"]').text()).toContain('Open Topic panel')
  })

  it('omits the entity link when there is no source_id', () => {
    const w = mountCard({ hit: hitOf({ metadata: { doc_type: 'kg_topic' } }) })
    expect(w.find('[data-testid="search-result-entity-link"]').exists()).toBe(false)
  })

  // --- Lifted GI insight section -------------------------------------------

  it('renders the lifted insight block (open by default) with id + text and toggles closed', async () => {
    const w = mountCard({
      hit: hitOf({
        lifted: {
          insight: { id: 'insight:42', text: 'A lifted insight body.' },
        },
      }),
    })
    expect(w.find('[role="region"][aria-label="Lifted GI insight"]').exists()).toBe(true)
    expect(w.text()).toContain('insight:42')
    expect(w.text()).toContain('A lifted insight body.')

    const toggle = w.findAll('button').find((b) => b.text().includes('linked GI insight'))!
    expect(toggle.text()).toContain('Hide')
    await toggle.trigger('click')
    expect(toggle.text()).toContain('Show')
    expect(w.text()).not.toContain('A lifted insight body.')
  })

  it('renders lifted speaker + topic links and focuses the corresponding subjects', async () => {
    const w = mountCard({
      hit: hitOf({
        lifted: {
          insight: { text: 'body' },
          speaker: { id: 'person:ada', display_name: 'Ada Lovelace' },
          topic: { id: 'topic:ai', display_name: 'AI' },
        },
      }),
    })
    const subject = useSubjectStore()

    const speakerLink = w.get('[data-testid="search-result-lifted-speaker-link"]')
    expect(speakerLink.text()).toBe('Ada Lovelace')
    await speakerLink.trigger('click')
    expect(subject.kind).toBe('person')
    expect(subject.personId).toBe('person:ada')

    const topicLink = w.get('[data-testid="search-result-lifted-topic-link"]')
    expect(topicLink.text()).toBe('AI')
    await topicLink.trigger('click')
    expect(subject.kind).toBe('topic')
    expect(subject.topicId).toBe('topic:ai')
  })

  it('renders the lifted quote time range from start/end timestamps', () => {
    const w = mountCard({
      hit: hitOf({
        lifted: {
          insight: { text: 'body' },
          quote: { timestamp_start_ms: 1500, timestamp_end_ms: 3200 },
        },
      }),
    })
    expect(w.text()).toContain('Quote time: 1.5s – 3.2s')
  })

  it('does not render the lifted section when there is no lifted payload', () => {
    const w = mountCard({ hit: hitOf({ lifted: null }) })
    expect(w.find('[aria-label="Lifted GI insight"]').exists()).toBe(false)
  })

  // --- Supporting quotes ----------------------------------------------------

  it('renders the supporting-quotes toggle (collapsed by default) and expands the list', async () => {
    const w = mountCard({
      hit: hitOf({
        supporting_quotes: [
          { text: 'First quote', speaker_name: 'Ada', speaker_id: 'person:ada' },
          { text: 'Second quote' },
        ],
      }),
    })
    const toggle = w.findAll('button').find((b) => b.text().includes('supporting quotes'))!
    expect(toggle.text()).toContain('Show 2 supporting quotes')
    // Collapsed: quote bodies not rendered yet.
    expect(w.text()).not.toContain('First quote')

    await toggle.trigger('click')
    expect(toggle.text()).toContain('Hide')
    expect(w.text()).toContain('First quote')
    expect(w.text()).toContain('Second quote')
  })

  it('uses singular "quote" wording for a single supporting quote', () => {
    const w = mountCard({
      hit: hitOf({ supporting_quotes: [{ text: 'Only one' }] }),
    })
    const toggle = w.findAll('button').find((b) => b.text().includes('supporting'))!
    expect(toggle.text()).toContain('1 supporting quote')
    expect(toggle.text()).not.toContain('quotes')
  })

  it('renders a speaker link inside a supporting quote and focuses that person', async () => {
    const w = mountCard({
      hit: hitOf({
        supporting_quotes: [
          { text: 'A quote', speaker_name: 'Ada', speaker_id: 'person:ada' },
        ],
      }),
    })
    await w.findAll('button').find((b) => b.text().includes('supporting'))!.trigger('click')
    const link = w.get('[data-testid="search-result-speaker-link"]')
    expect(link.text()).toBe('Ada')
    const subject = useSubjectStore()
    await link.trigger('click')
    expect(subject.kind).toBe('person')
    expect(subject.personId).toBe('person:ada')
  })

  it('shows the "no speaker detected" hint for a quote lacking speaker info', async () => {
    const w = mountCard({
      hit: hitOf({ supporting_quotes: [{ text: 'Anonymous quote' }] }),
    })
    await w.findAll('button').find((b) => b.text().includes('supporting'))!.trigger('click')
    expect(w.find('[data-testid="supporting-quote-speaker-unavailable"]').exists()).toBe(true)
  })

  it('does not render the supporting-quotes section when there are none', () => {
    const w = mountCard({ hit: hitOf({ supporting_quotes: [] }) })
    expect(w.findAll('button').some((b) => b.text().includes('supporting'))).toBe(false)
  })
})
