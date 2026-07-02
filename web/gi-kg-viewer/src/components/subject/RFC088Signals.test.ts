import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * RFC-088 — guard the enrichment signals on the subject-rail views.
 *
 * TopicEntityView now delegates its enrichment signals to the shared
 * ``NodeEnrichmentSection`` behind a dedicated Enrichment tab (the signal
 * rendering itself — velocity, co-occurrence chips, focus-pivot clicks — is
 * covered by NodeEnrichmentSection.test.ts). PersonLandingView still inlines
 * its own signal section (migration pending), so its guards are unchanged.
 *
 * Static-source guards. We check:
 *   - no v-html sinks (defence-in-depth)
 *   - TEV exposes the Overview | Enrichment tab hooks + delegates to
 *     NodeEnrichmentSection, gated on reported content
 *   - PLV still renders its inline signal section + focus-pivot clicks
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const TEV = resolve(HERE, 'TopicEntityView.vue')
const PLV = resolve(HERE, 'PersonLandingView.vue')

describe('TopicEntityView.vue — flat + embeddable node-view body', () => {
  const src = readFileSync(TEV, 'utf-8')

  it('has no v-html sink', () => {
    expect(src).not.toMatch(/v-html\s*=/)
  })

  it('is a flat, content-only body — no internal tabs, enrichment moved out', () => {
    // Enrichment (velocity / co-occurrence) + the mentions-by-month timeline now
    // live in the node view's Signals tab, not in TEV. TEV is a flat content scroll.
    expect(src).not.toContain('NodeEnrichmentSection')
    expect(src).not.toContain('topic-entity-view-tab-overview')
    expect(src).not.toContain('role="tablist"')
  })

  it('is embeddable — header/footer suppressed via the embedded prop', () => {
    expect(src).toContain('subjectIdOverride')
    expect(src).toContain('v-if="!embedded"')
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

  it('uses the chunk-8 cache composable (not the raw API helper)', () => {
    expect(src).toContain('fetchCachedCorpusEnvelope')
    expect(src).toContain("from '../../composables/useEnrichmentEnvelopeCache'")
  })

  it('co-guest chip click pivots subject focus to the co-guest person', () => {
    expect(src).toContain('@click="subject.focusPerson(g.person_id)"')
  })
})
