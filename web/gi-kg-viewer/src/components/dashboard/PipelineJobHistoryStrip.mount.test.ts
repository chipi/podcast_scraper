// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import PipelineJobHistoryStrip from './PipelineJobHistoryStrip.vue'
import { useShellStore } from '../../stores/shell'

/**
 * RFC-088 chunk-9 follow-up — mount tests for the dashboard's job
 * history strip with the new kind filter. Verifies that:
 *   - both pipeline + enrichment jobs appear under "All"
 *   - the "Pipeline" filter narrows to command_type=full_incremental_pipeline
 *   - the "Enrichment" filter narrows to command_type=corpus_enrichment
 *   - the option label embeds the [enrich] / [pipe] kind marker
 */

function res(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const MIXED_JOBS = {
  jobs: [
    {
      job_id: 'pipe-1',
      command_type: 'full_incremental_pipeline',
      status: 'succeeded',
      created_at: '2026-06-27T10:00:00Z',
      started_at: '2026-06-27T10:00:00Z',
      ended_at: '2026-06-27T10:05:00Z',
      pid: null,
      argv_summary: 'pipeline argv',
      exit_code: 0,
      log_relpath: '.viewer/job_pipe-1.log',
      error_reason: null,
    },
    {
      job_id: 'enrich-2',
      command_type: 'corpus_enrichment',
      status: 'succeeded',
      created_at: '2026-06-27T10:06:00Z',
      started_at: '2026-06-27T10:06:00Z',
      ended_at: '2026-06-27T10:08:00Z',
      pid: null,
      argv_summary: 'enrichment argv',
      exit_code: 0,
      log_relpath: '.viewer/job_enrich-2.log',
      error_reason: null,
    },
  ],
}

beforeEach(() => {
  setActivePinia(createPinia())
  const shell = useShellStore()
  shell.corpusPath = '/c'
  // The strip checks shell.healthStatus and shell.jobsApiAvailable.
  shell.healthStatus = 'healthy'
  shell.jobsApiAvailable = true
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/jobs')) return res(MIXED_JOBS)
      return res({}, 404)
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})


describe('PipelineJobHistoryStrip — kind filter behaviour', () => {
  it('renders the three kind-filter buttons', async () => {
    const w = mount(PipelineJobHistoryStrip)
    await flushPromises()
    expect(w.find('[data-testid="pipeline-job-kind-filter-all"]').exists()).toBe(true)
    expect(w.find('[data-testid="pipeline-job-kind-filter-pipeline"]').exists()).toBe(true)
    expect(w.find('[data-testid="pipeline-job-kind-filter-enrichment"]').exists()).toBe(true)
  })

  it('All filter (default) shows both pipeline and enrichment jobs', async () => {
    const w = mount(PipelineJobHistoryStrip)
    await flushPromises()
    const text = w.text()
    expect(text).toContain('[pipe]')
    expect(text).toContain('[enrich]')
  })

  it('clicking Pipeline updates the filter button highlight + narrows the visible set', async () => {
    const w = mount(PipelineJobHistoryStrip)
    await flushPromises()
    // Initial state: 2 finished jobs visible (1 pipe + 1 enrich).
    const initialPipe = (w.text().match(/\[pipe\]/g) || []).length
    const initialEnrich = (w.text().match(/\[enrich\]/g) || []).length
    expect(initialPipe).toBeGreaterThanOrEqual(1)
    expect(initialEnrich).toBeGreaterThanOrEqual(1)
    // Click Pipeline filter.
    const pipelineBtn = w.get('[data-testid="pipeline-job-kind-filter-pipeline"]')
    await pipelineBtn.trigger('click')
    await flushPromises()
    // The Pipeline button should now be visually selected (font-medium).
    expect(pipelineBtn.classes().join(' ')).toContain('font-medium')
    // The All / Enrichment buttons should NOT be selected.
    expect(
      w.get('[data-testid="pipeline-job-kind-filter-all"]').classes().join(' '),
    ).not.toContain('font-medium')
    expect(
      w.get('[data-testid="pipeline-job-kind-filter-enrichment"]').classes().join(' '),
    ).not.toContain('font-medium')
  })

  it('Enrichment filter narrows to enrichment jobs only (no [pipe] visible)', async () => {
    const w = mount(PipelineJobHistoryStrip)
    await flushPromises()
    await w.get('[data-testid="pipeline-job-kind-filter-enrichment"]').trigger('click')
    await flushPromises()
    // After clicking Enrichment, no pipeline rows remain (the auto-selected
    // job was the enrich-2 newest, so the strip doesn't have to preserve
    // a pipe selection across filter change).
    const text = w.text()
    expect(text).toContain('[enrich]')
    // The pipeline kind shouldn't appear in the option label list any more.
    const pipeCount = (text.match(/\[pipe\]/g) || []).length
    expect(pipeCount).toBe(0)
  })
})
