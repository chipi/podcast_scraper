import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * RFC-088 chunk 6b — surface guards for the Configuration popup Enrichment tab.
 *
 * The panel renders untrusted enricher health / metrics / run-summary payloads
 * pulled from the server. Static guards prove the surface stays wired:
 *   - no v-html sinks (defence-in-depth: enricher fields are server-sourced
 *     strings; v-html would let a forged auto_disabled_reason inject markup)
 *   - the data-testid hooks the stack-test spec depends on
 *   - the action-button wiring (run-now + re-enable) calls the right API helpers
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const PANEL_SFC = resolve(HERE, 'EnrichmentPanel.vue')

describe('EnrichmentPanel.vue — RFC-088 chunk 6b surface', () => {
  const source = readFileSync(PANEL_SFC, 'utf-8')

  it('has no v-html directive (defence-in-depth XSS guard)', () => {
    expect(source).not.toMatch(/v-html\s*=/)
  })

  it('renders the expected data-testid hooks', () => {
    for (const hook of [
      'enrichment-panel',
      'enrichment-refresh-btn',
      'enrichment-run-btn',
      'enrichment-row-count',
      'enrichment-total-runs',
      'enrichment-autodisabled-count',
      'enrichment-table',
      'enrichment-empty-row',
    ]) {
      expect(source).toContain(`data-testid="${hook}"`)
    }
  })

  it('wires action buttons to the enrichmentApi helpers', () => {
    expect(source).toContain('submitEnrichmentJob')
    expect(source).toContain('reEnableEnricher')
    expect(source).toContain('getEnrichmentHealth')
    expect(source).toContain('getEnrichmentMetrics')
    expect(source).toContain('getEnrichmentStatus')
    expect(source).toContain('getEnrichmentRunSummary')
    expect(source).toContain('getCorpusEnrichmentsCatalogue')
  })

  it('renders the re-enable button under :data-testid binding', () => {
    expect(source).toContain(':data-testid="`enrichment-re-enable-${row.enricher_id}`"')
  })
})
