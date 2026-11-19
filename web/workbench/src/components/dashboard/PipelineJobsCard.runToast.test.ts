// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import PipelineJobsCard from './PipelineJobsCard.vue'
import { useShellStore } from '../../stores/shell'
import { listPipelineJobs, submitPipelineJob } from '../../api/jobsApi'

// The real module's pure helpers, with every network call mocked: a failed job makes the card's
// explore panel fetch its log tail, which an incomplete mock left as an unhandled rejection.
vi.mock('../../api/jobsApi', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/jobsApi')>()),
  listPipelineJobs: vi.fn(),
  submitPipelineJob: vi.fn(),
  reconcilePipelineJobs: vi.fn(),
  cancelPipelineJob: vi.fn(),
  fetchPipelineJobLogTail: vi.fn(async () => ({ text: '', truncated: false })),
}))

describe('PipelineJobsCard — the run notice follows the job', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    shell.jobsApiAvailable = true
  })

  afterEach(() => vi.useRealTimers())

  it('shows a job that failed right after it was queued as failed, not running', async () => {
    vi.mocked(submitPipelineJob).mockResolvedValue({
      job_id: 'e6a9bdfc1234',
      status: 'running',
      corpus_path: '/corpus',
    } as never)
    // The job ended (failed in 0.2 s, review 2026-11-17) before the card's first refresh.
    vi.mocked(listPipelineJobs).mockResolvedValue({
      jobs: [{ job_id: 'e6a9bdfc1234', status: 'failed' }],
      running: 0,
      max_concurrent: 1,
    } as never)

    const w = mount(PipelineJobsCard)
    await flushPromises()
    await w.find('[data-testid="pipeline-jobs-run"]').trigger('click')
    await flushPromises()

    const notice = w.find('[data-testid="pipeline-jobs-run-toast"]')
    expect(notice.exists()).toBe(true)
    expect(notice.text()).toContain('failed')
    expect(notice.text()).not.toContain('running')
  })

  it('keeps the queued status while the job is not in the list yet', async () => {
    vi.mocked(submitPipelineJob).mockResolvedValue({
      job_id: 'abc123456789',
      status: 'queued',
      corpus_path: '/corpus',
    } as never)
    vi.mocked(listPipelineJobs).mockResolvedValue({
      jobs: [],
      running: 0,
      max_concurrent: 1,
    } as never)

    const w = mount(PipelineJobsCard)
    await flushPromises()
    await w.find('[data-testid="pipeline-jobs-run"]').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="pipeline-jobs-run-toast"]').text()).toContain('queued')
  })
})
