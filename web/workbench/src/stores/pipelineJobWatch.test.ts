import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { listPipelineJobs } from '../api/jobsApi'
import { JOB_WATCH_INTERVAL_MS, usePipelineJobWatchStore } from './pipelineJobWatch'
import { useShellStore } from './shell'

vi.mock('../api/jobsApi', () => ({ listPipelineJobs: vi.fn() }))
// Setting `corpusPath` makes the shell store probe /api/health, which fails here (no server) and
// resets `jobsApiAvailable`. A probe that never answers leaves the values these tests set alone.
vi.mock('../api/httpClient', () => ({ fetchWithTimeout: () => new Promise(() => {}) }))

const row = (status: string, id = 'j1') => ({ job_id: id, status }) as never
const list = (...jobs: unknown[]) => ({ path: '/corpus', jobs }) as never

describe('pipelineJobWatch — a job finishing is noticed on any tab (2026-10-09)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.jobsApiAvailable = true
  })
  afterEach(() => {
    usePipelineJobWatchStore().reset()
    vi.useRealTimers()
    vi.clearAllMocks()
  })

  it('the card saw it running and then left; the watcher sees it succeed and bumps the revision', async () => {
    vi.mocked(listPipelineJobs).mockResolvedValue(list(row('succeeded')))
    usePipelineJobWatchStore().observe([row('running')])
    expect(useShellStore().corpusRevision).toBe(0)
    await vi.advanceTimersByTimeAsync(JOB_WATCH_INTERVAL_MS)
    expect(listPipelineJobs).toHaveBeenCalledWith('/corpus')
    expect(useShellStore().corpusRevision).toBe(1)
  })

  it('stops looking once nothing is in flight', async () => {
    vi.mocked(listPipelineJobs).mockResolvedValue(list(row('succeeded')))
    usePipelineJobWatchStore().observe([row('running')])
    await vi.advanceTimersByTimeAsync(JOB_WATCH_INTERVAL_MS)
    await vi.advanceTimersByTimeAsync(JOB_WATCH_INTERVAL_MS * 4)
    expect(listPipelineJobs).toHaveBeenCalledTimes(1)
  })

  it('never looks when nothing was in flight', async () => {
    usePipelineJobWatchStore().observe([row('succeeded')])
    await vi.advanceTimersByTimeAsync(JOB_WATCH_INTERVAL_MS * 3)
    expect(listPipelineJobs).not.toHaveBeenCalled()
    expect(useShellStore().corpusRevision).toBe(0)
  })

  it('a failed job does not bump; a network blip keeps watching', async () => {
    vi.mocked(listPipelineJobs)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValue(list(row('failed')))
    usePipelineJobWatchStore().observe([row('running')])
    await vi.advanceTimersByTimeAsync(JOB_WATCH_INTERVAL_MS)
    await vi.advanceTimersByTimeAsync(JOB_WATCH_INTERVAL_MS)
    expect(listPipelineJobs).toHaveBeenCalledTimes(2)
    expect(useShellStore().corpusRevision).toBe(0)
  })
})
