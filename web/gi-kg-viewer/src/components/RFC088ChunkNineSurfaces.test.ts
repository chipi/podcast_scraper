import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * RFC-088 chunk-9 surface guard: EpisodeDetailPanel mounts EpisodeEnrichmentSection.
 * (The related-topic chips, the enrichment-edges panel and the consensus rows read
 * enrichers that moved to the private intelligence package, ADR-162.)
 *
 * Static-source guards: no v-html sinks, data-testid hooks present,
 * imports + bindings wired.
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const EPISODE_DETAIL_PANEL = resolve(HERE, 'episode/EpisodeDetailPanel.vue')
const EPISODE_ENRICHMENT_SECTION = resolve(HERE, 'episode/EpisodeEnrichmentSection.vue')


describe('EpisodeDetailPanel mounts EpisodeEnrichmentSection', () => {
  const detail = readFileSync(EPISODE_DETAIL_PANEL, 'utf-8')
  const section = readFileSync(EPISODE_ENRICHMENT_SECTION, 'utf-8')

  it('detail panel imports and mounts the section', () => {
    expect(detail).toContain("import EpisodeEnrichmentSection from './EpisodeEnrichmentSection.vue'")
    expect(detail).toContain('<EpisodeEnrichmentSection')
  })

  it('section reads the insight_density envelope via getEpisodeEnrichmentEnvelope', () => {
    expect(section).toContain('getEpisodeEnrichmentEnvelope')
    expect(section).toContain("'insight_density'")
  })

  it('section has the data-testid hooks for density', () => {
    expect(section).toContain('data-testid="episode-enrichment-section"')
    expect(section).toContain('data-testid="episode-enrichment-density"')
  })

  it('section has no v-html sink', () => {
    expect(section).not.toMatch(/v-html\s*=/)
  })
})

