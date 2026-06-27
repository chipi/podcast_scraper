import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * RFC-088 chunk 6c — guard the enrichment-signal sections on the two
 * subject-rail views (TopicEntityView + PersonLandingView).
 *
 * Static-source guards. The sections render server-sourced enricher
 * payloads (topic_cooccurrence_corpus, temporal_velocity, grounding_rate,
 * guest_coappearance), so we check:
 *   - no v-html sinks (defence-in-depth)
 *   - data-testid hooks for stack-test specs are present
 *   - the source imports the chunk-6a getCorpusEnrichmentEnvelope helper
 *   - co-occurrence / co-guest chips bind to the matching subject focus
 *     methods so clicking pivots the rail
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const TEV = resolve(HERE, 'TopicEntityView.vue')
const PLV = resolve(HERE, 'PersonLandingView.vue')

describe('TopicEntityView.vue — RFC-088 chunk 6c enrichment signals', () => {
  const src = readFileSync(TEV, 'utf-8')

  it('has no v-html sink', () => {
    expect(src).not.toMatch(/v-html\s*=/)
  })

  it('renders the enrichment-signals testid hooks', () => {
    for (const hook of [
      'topic-entity-view-enrichment-signals',
      'topic-entity-view-velocity',
      'topic-entity-view-cooccurrence',
    ]) {
      expect(src).toContain(`data-testid="${hook}"`)
    }
  })

  it('imports getCorpusEnrichmentEnvelope from enrichmentApi', () => {
    expect(src).toContain('getCorpusEnrichmentEnvelope')
    expect(src).toContain("from '../../api/enrichmentApi'")
  })

  it('co-occurrence chip click pivots subject focus to the partner topic', () => {
    expect(src).toContain('@click="subject.focusTopic(r.topic_id)"')
  })
})

describe('PersonLandingView.vue — RFC-088 chunk 6c enrichment signals', () => {
  const src = readFileSync(PLV, 'utf-8')

  it('has no v-html sink', () => {
    expect(src).not.toMatch(/v-html\s*=/)
  })

  it('renders the enrichment-signals testid hooks', () => {
    for (const hook of [
      'person-landing-enrichment-signals',
      'person-landing-grounding-rate',
      'person-landing-coguests',
    ]) {
      expect(src).toContain(`data-testid="${hook}"`)
    }
  })

  it('imports getCorpusEnrichmentEnvelope from enrichmentApi', () => {
    expect(src).toContain('getCorpusEnrichmentEnvelope')
    expect(src).toContain("from '../../api/enrichmentApi'")
  })

  it('co-guest chip click pivots subject focus to the co-guest person', () => {
    expect(src).toContain('@click="subject.focusPerson(g.person_id)"')
  })
})
