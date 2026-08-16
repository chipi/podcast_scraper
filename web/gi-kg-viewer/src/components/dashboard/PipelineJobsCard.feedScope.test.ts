// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import PipelineJobsCard from './PipelineJobsCard.vue'
import { useShellStore } from '../../stores/shell'
import { listPipelineJobs, submitPipelineJob } from '../../api/jobsApi'

vi.mock('../../api/jobsApi', () => ({
  listPipelineJobs: vi.fn(),
  submitPipelineJob: vi.fn(),
  reconcilePipelineJobs: vi.fn(),
  cancelPipelineJob: vi.fn(),
}))

describe('PipelineJobsCard — per-feed scope run (P1.4 UI)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    vi.mocked(listPipelineJobs).mockResolvedValue({ jobs: [], running: 0, max_concurrent: 1 } as never)
    vi.mocked(submitPipelineJob).mockResolvedValue({
      job_id: 'feedjob123',
      status: 'queued',
      corpus_path: '/corpus',
    } as never)
  })

  afterEach(() => vi.useRealTimers())

  it('submits a feed-scoped run with skip_existing + max_episodes from the inputs', async () => {
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    shell.jobsApiAvailable = true

    const w = mount(PipelineJobsCard)
    await flushPromises()

    await w.find('[data-testid="pipeline-jobs-feed-input"]').setValue('https://a.example/1.xml')
    await w.find('[data-testid="pipeline-jobs-feed-max"]').setValue('3')
    await w.find('[data-testid="pipeline-jobs-run-feed"]').trigger('click')
    await flushPromises()

    expect(submitPipelineJob).toHaveBeenCalledWith('/corpus', {
      feed: 'https://a.example/1.xml',
      skipExisting: true,
      maxEpisodes: 3,
      episodeOrder: 'newest',
    })
  })

  it('disables Run feed when the feed input is empty', async () => {
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    shell.jobsApiAvailable = true

    const w = mount(PipelineJobsCard)
    await flushPromises()
    expect(w.find('[data-testid="pipeline-jobs-run-feed"]').attributes('disabled')).toBeDefined()
  })
})
