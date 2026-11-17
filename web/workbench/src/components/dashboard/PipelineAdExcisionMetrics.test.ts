import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * #656 Stage D — static + XSS invariants for PipelineAdExcisionMetrics.
 *
 * Mirrors PipelineCleanupMetrics.test.ts. The component reads #663
 * ad-excision counters off ``CorpusRunSummaryItem`` and renders via
 * DiagnosticRow — no templating sinks, but the static guard ensures
 * none slip in and that the three counter names stay wired end-to-end
 * through a future backend rename.
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const COMPONENT = resolve(HERE, 'PipelineAdExcisionMetrics.vue')
const source = readFileSync(COMPONENT, 'utf-8')

describe('PipelineAdExcisionMetrics.vue — shape + safety invariants', () => {
  it('has no v-html directive', () => {
    expect(source).not.toMatch(/\sv-html\s*=/)
    expect(source).not.toMatch(/\.innerHTML\s*=/)
  })

  it('references all three #663 ad-excision counters', () => {
    expect(source).toContain('ad_chars_excised_preroll')
    expect(source).toContain('ad_chars_excised_postroll')
    expect(source).toContain('ad_episodes_with_excision_count')
  })

  it('uses DiagnosticRow for each row', () => {
    expect(source).toMatch(/import DiagnosticRow from/)
    expect(source).toMatch(/<DiagnosticRow/)
  })

  it('renders "—" for null values (pre-#663 runs)', () => {
    expect(source).toMatch(/'—'/)
    expect(source).toMatch(/v == null \? '—'/)
  })

  it('suppresses info chip when value is 0 or null (quiet rows)', () => {
    expect(source).toMatch(/v > 0 \? 'info' : 'default'/)
  })

  it('declares an aria-labelledby heading for the section', () => {
    expect(source).toMatch(/aria-labelledby="pipeline-ad-excision-metrics-heading"/)
    expect(source).toMatch(/id="pipeline-ad-excision-metrics-heading"/)
  })

  it('emits stable test-ids per counter', () => {
    expect(source).toMatch(/`pipeline-ad-excision-\$\{row\.badge\}`/)
  })
})
