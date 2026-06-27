import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * RFC-088 chunk 6b — surface guards for the Dashboard pipeline-runs strip.
 *
 * The strip now lists pipeline AND enrichment jobs (command_type filter +
 * kind label). These are static guards proving the surface stays wired:
 *   - the kind filter group renders for all three kinds
 *   - the new helper functions exist (jobKindLabel + matchesKindFilter)
 *   - the option label includes the kind marker
 *   - the data-testid hooks the stack-test specs depend on are present
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const STRIP_SFC = resolve(HERE, 'PipelineJobHistoryStrip.vue')

describe('PipelineJobHistoryStrip.vue — RFC-088 kind filter wiring', () => {
  const source = readFileSync(STRIP_SFC, 'utf-8')

  it('renders a kind-filter button group keyed by the three kinds', () => {
    expect(source).toContain("data-testid=\"pipeline-job-kind-filter\"")
    expect(source).toContain("pipeline-job-kind-filter-${kind}")
    expect(source).toContain("['all', 'pipeline', 'enrichment']")
  })

  it('declares the JobKindFilter type', () => {
    expect(source).toMatch(/type\s+JobKindFilter\s*=\s*'all'\s*\|\s*'pipeline'\s*\|\s*'enrichment'/)
  })

  it('defines matchesKindFilter that maps to command_type', () => {
    expect(source).toMatch(/function\s+matchesKindFilter/)
    expect(source).toContain("'corpus_enrichment'")
    expect(source).toContain("'full_incremental_pipeline'")
  })

  it('option label includes the jobKindLabel marker', () => {
    expect(source).toMatch(/function\s+jobKindLabel/)
    expect(source).toContain('${jobKindLabel(j)}')
    expect(source).toContain("'[enrich]'")
    expect(source).toContain("'[pipe]'")
  })

  it('filtered list applies the kind filter before the substring filter', () => {
    expect(source).toMatch(/finishedJobs\.value\.filter\(matchesKindFilter\)/)
  })
})
