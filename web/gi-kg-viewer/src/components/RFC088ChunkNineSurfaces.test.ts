import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * RFC-088 chunk-9 surface guards — the four viewer gaps that landed
 * after the chunk-8 sweep:
 *   1. Search ResultCard renders related_topics chips
 *   2. EpisodeDetailPanel mounts EpisodeEnrichmentSection
 *   3. GraphTabPanel mounts EnrichmentEdgesPanel
 *   4. PersonLandingView surfaces contradiction rows
 *
 * Static-source guards: no v-html sinks, data-testid hooks present,
 * imports + bindings wired.
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const SEARCH_RESULT_CARD = resolve(HERE, 'search/ResultCard.vue')
const EPISODE_DETAIL_PANEL = resolve(HERE, 'episode/EpisodeDetailPanel.vue')
const EPISODE_ENRICHMENT_SECTION = resolve(HERE, 'episode/EpisodeEnrichmentSection.vue')
const GRAPH_TAB_PANEL = resolve(HERE, 'graph/GraphTabPanel.vue')
const ENRICHMENT_EDGES_PANEL = resolve(HERE, 'graph/EnrichmentEdgesPanel.vue')
const PERSON_LANDING_VIEW = resolve(HERE, 'subject/PersonLandingView.vue')


describe('Gap 1 — ResultCard renders related_topics chips', () => {
  const src = readFileSync(SEARCH_RESULT_CARD, 'utf-8')

  it('has no v-html sink', () => {
    expect(src).not.toMatch(/v-html\s*=/)
  })

  it('reads metadata.query_enrichments.related_topics', () => {
    expect(src).toContain('query_enrichments')
    expect(src).toContain('related_topics')
  })

  it('renders the chip strip data-testid + click handler', () => {
    expect(src).toContain('data-testid="search-result-related-topics"')
    expect(src).toContain('focusRelatedTopic(r.topic_id)')
  })
})


describe('Gap 2 — EpisodeDetailPanel mounts EpisodeEnrichmentSection', () => {
  const detail = readFileSync(EPISODE_DETAIL_PANEL, 'utf-8')
  const section = readFileSync(EPISODE_ENRICHMENT_SECTION, 'utf-8')

  it('detail panel imports and mounts the section', () => {
    expect(detail).toContain("import EpisodeEnrichmentSection from './EpisodeEnrichmentSection.vue'")
    expect(detail).toContain('<EpisodeEnrichmentSection')
  })

  it('section reads both episode-scope envelopes via getEpisodeEnrichmentEnvelope', () => {
    expect(section).toContain('getEpisodeEnrichmentEnvelope')
    expect(section).toContain("'insight_density'")
    expect(section).toContain("'topic_cooccurrence'")
  })

  it('section has the data-testid hooks for density + co-occurrence', () => {
    expect(section).toContain('data-testid="episode-enrichment-section"')
    expect(section).toContain('data-testid="episode-enrichment-density"')
    expect(section).toContain('data-testid="episode-enrichment-cooccurrence"')
  })

  it('section has no v-html sink', () => {
    expect(section).not.toMatch(/v-html\s*=/)
  })
})


describe('Gap 3 — GraphTabPanel mounts EnrichmentEdgesPanel', () => {
  const tab = readFileSync(GRAPH_TAB_PANEL, 'utf-8')
  const panel = readFileSync(ENRICHMENT_EDGES_PANEL, 'utf-8')

  it('tab panel imports and mounts the panel', () => {
    expect(tab).toContain("import EnrichmentEdgesPanel from './EnrichmentEdgesPanel.vue'")
    expect(tab).toContain('<EnrichmentEdgesPanel')
  })

  it('panel uses the cache composable for both envelopes', () => {
    expect(panel).toContain('fetchCachedCorpusEnvelope')
    expect(panel).toContain("'topic_similarity'")
    expect(panel).toContain("'nli_contradiction'")
  })

  it('panel renders the similarity and contradictions data-testid hooks', () => {
    expect(panel).toContain('data-testid="enrichment-edges-panel"')
    expect(panel).toContain('data-testid="enrichment-edges-similarity"')
    expect(panel).toContain('data-testid="enrichment-edges-contradictions"')
  })

  it('panel rows bind clicks to subject store focus methods', () => {
    expect(panel).toContain('subject.focusTopic')
    expect(panel).toContain('subject.focusPerson')
  })

  it('panel has no v-html sink', () => {
    expect(panel).not.toMatch(/v-html\s*=/)
  })
})


describe('Gap 4 — PersonLandingView surfaces contradiction rows', () => {
  const src = readFileSync(PERSON_LANDING_VIEW, 'utf-8')

  it('declares the ContradictionRow type and contradictionRows ref', () => {
    expect(src).toContain('interface ContradictionRow')
    expect(src).toContain('contradictionRows = ref')
  })

  it('loads nli_contradiction via the cache composable', () => {
    expect(src).toContain("'nli_contradiction'")
    expect(src).toContain('fetchCachedCorpusEnvelope')
  })

  it('template renders the contradictions row data-testid hook', () => {
    expect(src).toContain('data-testid="person-landing-contradictions"')
    expect(src).toContain('person-landing-contra-${row.partner_id}--${row.topic_id}')
  })

  it('row clicks pivot subject focus to partner / topic', () => {
    expect(src).toContain('subject.focusPerson(row.partner_id)')
    expect(src).toContain('subject.focusTopic(row.topic_id)')
  })
})
