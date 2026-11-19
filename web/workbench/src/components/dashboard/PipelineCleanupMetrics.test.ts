import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * #656 Stage B — static + XSS invariants for PipelineCleanupMetrics.
 *
 * The component is a thin pass-through from ``CorpusRunSummaryItem``
 * into a ``DiagnosticRow`` list — the counter values come from the
 * backend's ``metrics.json`` via ``/api/corpus/runs/summary``, which
 * is untrusted pipeline output even though it's numeric. Until the
 * viewer grows a mount-based component-test harness (post-#656), this
 * static guard is the fast way to ensure no raw-HTML sink slips in
 * and the four #652 counters stay referenced.
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const COMPONENT = resolve(HERE, 'PipelineCleanupMetrics.vue')
const source = readFileSync(COMPONENT, 'utf-8')

describe('PipelineCleanupMetrics.vue — shape + safety invariants', () => {
  it('has no v-html directive', () => {
    expect(source).not.toMatch(/\sv-html\s*=/)
    expect(source).not.toMatch(/\.innerHTML\s*=/)
  })

  it('references all four #652 Part B counters', () => {
    // Regression guard: if the backend renames a field or a future
    // refactor drops one, this fires.
    expect(source).toContain('ads_filtered_count')
    expect(source).toContain('dialogue_insights_dropped_count')
    expect(source).toContain('topics_normalized_count')
    expect(source).toContain('entity_kinds_repaired_count')
  })

  it('uses DiagnosticRow for each row', () => {
    expect(source).toMatch(/import DiagnosticRow from/)
    expect(source).toMatch(/<DiagnosticRow/)
  })

  it('renders "—" for null values (pre-#652 runs)', () => {
    // Signals "data absent" without conflating with a real ``0``.
    expect(source).toMatch(/'—'/)
    // ``null`` check precedes any formatting so NaN can't sneak through.
    expect(source).toMatch(/v == null \? '—'/)
  })

  it('suppresses info chip when value is 0 or null (quiet rows)', () => {
    // ``badgeKind`` only returns ``info`` when ``v > 0`` — zero-count
    // rows stay visually calm, a well-tuned run shouldn't scream.
    expect(source).toMatch(/v > 0 \? 'info' : 'default'/)
  })

  it('declares an aria-labelledby heading for the section', () => {
    // Screen readers announce the card as "Pipeline cleanup" — the
    // section doesn't rely on sighted landmarks.
    expect(source).toMatch(/aria-labelledby="pipeline-cleanup-metrics-heading"/)
    expect(source).toMatch(/id="pipeline-cleanup-metrics-heading"/)
  })

  it('emits stable test-ids per counter', () => {
    // Future Playwright coverage can select each row by counter badge.
    expect(source).toMatch(/`pipeline-cleanup-\$\{row\.badge\}`/)
  })
})
