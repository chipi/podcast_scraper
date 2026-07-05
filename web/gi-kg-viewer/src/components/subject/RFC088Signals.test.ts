import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * RFC-088 — guard the enrichment signals on the subject-rail views.
 *
 * Both TopicEntityView and PersonLandingView now delegate their enrichment
 * signals to the shared ``NodeEnrichmentSection`` in the node view's Signals
 * tab (the signal rendering itself — velocity, co-occurrence, grounding,
 * contradictions, focus-pivot clicks — is covered by NodeEnrichmentSection.test.ts).
 * Neither view inlines its own enrichment section any more.
 *
 * Static-source guards. We check:
 *   - no v-html sinks (defence-in-depth)
 *   - neither TEV nor PLV inlines enrichment / mentions-by-month; it lives in
 *     the Signals tab (NodeEnrichmentSection + signalsTimeline) exactly once
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

describe('PersonLandingView.vue — enrichment signals moved to the Signals tab', () => {
  const src = readFileSync(PLV, 'utf-8')

  it('has no v-html sink', () => {
    expect(src).not.toMatch(/v-html\s*=/)
  })

  it('no longer inlines the enrichment-signals section (grounding / co-guests / contradictions)', () => {
    for (const hook of [
      'person-landing-enrichment-signals',
      'person-landing-grounding-rate',
      'person-landing-coguests',
      'person-landing-contradictions',
    ]) {
      expect(src).not.toContain(`data-testid="${hook}"`)
    }
  })

  it('no longer fetches the enrichment envelope (the Signals tab does)', () => {
    expect(src).not.toContain('fetchCachedCorpusEnvelope')
    expect(src).not.toContain("from '../../composables/useEnrichmentEnvelopeCache'")
  })

  it('no longer renders the mentions-by-month timeline (lives in the Signals tab)', () => {
    expect(src).not.toContain('SubjectTimelineChart')
  })
})
