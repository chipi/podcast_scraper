// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import PipelineJobsCard from './PipelineJobsCard.vue'
import { useShellStore } from '../../stores/shell'
import { listPipelineJobs } from '../../api/jobsApi'
import type { VueWrapper } from '@vue/test-utils'

/** The next poll, through the card's own Refresh (same `refresh()` the timer runs). */
async function poll(w: VueWrapper): Promise<void> {
  await w.findAll('button').find((b) => b.text() === 'Refresh')!.trigger('click')
  await flushPromises()
}

vi.mock('../../api/jobsApi', () => ({
  listPipelineJobs: vi.fn(),
  submitPipelineJob: vi.fn(),
  reconcilePipelineJobs: vi.fn(),
  cancelPipelineJob: vi.fn(),
  fetchPipelineJobLogTail: vi.fn().mockResolvedValue({ lines: [] }),
}))

const row = (status: string) => ({ job_id: 'j1', status, corpus_path: '/corpus' })
const list = (...jobs: unknown[]) => ({ jobs, running: 0, max_concurrent: 1 }) as never

/**
 * Library and Digest are kept alive and reload only on a path or health change, so a job run from
 * the Dashboard left them showing the corpus as it was before it (2026-10-09). The card is the one
 * place that watches jobs; a job SUCCEEDING bumps `shell.corpusRevision`, which both tabs watch.
 */
describe('PipelineJobsCard — a finished job tells the kept-alive tabs', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    shell.jobsApiAvailable = true
  })
  afterEach(() => vi.clearAllMocks())

  it('running on one poll, succeeded on the next: the corpus revision moves', async () => {
    vi.mocked(listPipelineJobs).mockResolvedValueOnce(list(row('running'))).mockResolvedValue(list(row('succeeded')))
    const w = mount(PipelineJobsCard)
    await flushPromises()
    const shell = useShellStore()
    expect(shell.corpusRevision).toBe(0)
    await poll(w)
    expect(shell.corpusRevision).toBe(1)
  })

  it('a job that had already succeeded before the card opened does not count', async () => {
    vi.mocked(listPipelineJobs).mockResolvedValue(list(row('succeeded')))
    const w = mount(PipelineJobsCard)
    await flushPromises()
    await poll(w)
    expect(useShellStore().corpusRevision).toBe(0)
  })

  it('a FAILED job does not count', async () => {
    vi.mocked(listPipelineJobs).mockResolvedValueOnce(list(row('running'))).mockResolvedValue(list(row('failed')))
    const w = mount(PipelineJobsCard)
    await flushPromises()
    await poll(w)
    expect(useShellStore().corpusRevision).toBe(0)
  })
})
