// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const fetchPipelineJobLogTail = vi.fn()
const listPipelineJobs = vi.fn()

vi.mock('../../api/jobsApi', () => ({
  fetchPipelineJobLogTail: (...args: unknown[]) => fetchPipelineJobLogTail(...args),
  listPipelineJobs: (...args: unknown[]) => listPipelineJobs(...args),
  formatJobHttpErrorMessage: (m: string) => m,
  isLivePipelineJobStatus: (s: string) => s === 'running' || s === 'queued',
  pipelineJobLogUrl: (p: string, id: string) => `/api/jobs/subprocess-log?path=${p}&job_id=${id}`,
}))

vi.mock('../../stores/shell', () => ({
  useShellStore: () => ({ jobsApiAvailable: true }),
}))

import PipelineJobLogDialog from './PipelineJobLogDialog.vue'
import { usePipelineJobLogStore } from '../../stores/pipelineJobLog'

// happy-dom <dialog> doesn't flip `open` / emit `close`; patch like AppDialog test.
beforeEach(() => {
  const proto = HTMLDialogElement.prototype as unknown as {
    showModal: () => void
    close: () => void
  }
  proto.showModal = function showModal(this: HTMLDialogElement) {
    this.setAttribute('open', '')
  }
  proto.close = function close(this: HTMLDialogElement) {
    if (!this.open) return
    this.removeAttribute('open')
    this.dispatchEvent(new Event('close'))
  }
  setActivePinia(createPinia())
  fetchPipelineJobLogTail.mockReset()
  listPipelineJobs.mockReset()
  listPipelineJobs.mockResolvedValue({ path: '/mock/corpus', jobs: [] })
})

afterEach(() => {
  vi.restoreAllMocks()
})

function mountDialog() {
  return mount(PipelineJobLogDialog, { attachTo: document.body })
}

async function openTerminalJob(text = 'log line one\nlog line two', truncated = false) {
  fetchPipelineJobLogTail.mockResolvedValue({ text, truncated })
  const w = mountDialog()
  const store = usePipelineJobLogStore()
  store.viewLog({ corpusPath: '/mock/corpus', jobId: 'job-abc-123', status: 'succeeded', logRelpath: 'logs/job.log' })
  await flushPromises()
  return { w, store }
}

describe('PipelineJobLogDialog', () => {
  it('fetches and shows the tail when opened', async () => {
    const { w } = await openTerminalJob()
    expect(fetchPipelineJobLogTail).toHaveBeenCalledTimes(1)
    expect(fetchPipelineJobLogTail).toHaveBeenCalledWith('/mock/corpus', 'job-abc-123', 65_536)
    expect(w.get('[data-testid="pipeline-job-log-body"]').text()).toContain('log line two')
  })

  it('does not auto-refresh a terminal job (no polling thrash)', async () => {
    const { w } = await openTerminalJob()
    expect(w.text()).toContain('terminal state')
  })

  it('Refresh issues a new tail request', async () => {
    const { w } = await openTerminalJob()
    await w.get('[data-testid="pipeline-job-log-refresh"]').trigger('click')
    await flushPromises()
    expect(fetchPipelineJobLogTail).toHaveBeenCalledTimes(2)
  })

  it('changing tail size refetches with the new byte budget', async () => {
    const { w } = await openTerminalJob()
    const select = w.get('[data-testid="pipeline-job-log-tail-size"]')
    // Numeric-valued <select>: Vue tracks the bound number via _value, not the DOM
    // value attribute, so drive selection by index then fire change.
    ;(select.element as HTMLSelectElement).selectedIndex = 2 // 256 KB
    await select.trigger('change')
    await flushPromises()
    expect(fetchPipelineJobLogTail).toHaveBeenLastCalledWith('/mock/corpus', 'job-abc-123', 262_144)
  })

  it('Copy writes the log to the clipboard and confirms', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    const { w } = await openTerminalJob('copy me')
    await w.get('[data-testid="pipeline-job-log-copy"]').trigger('click')
    await flushPromises()
    expect(writeText).toHaveBeenCalledWith('copy me')
    expect(w.get('[data-testid="pipeline-job-log-copy"]').text()).toContain('Copied')
  })

  it('surfaces the truncated-head hint when the tail is truncated', async () => {
    const { w } = await openTerminalJob('tail only', true)
    expect(w.find('[data-testid="pipeline-job-log-truncated-hint"]').exists()).toBe(true)
  })

  it('offers a download-full-log escape hatch', async () => {
    const { w } = await openTerminalJob()
    const dl = w.get('[data-testid="pipeline-job-log-download"]')
    expect(dl.attributes('href')).toContain('/api/jobs/subprocess-log?')
  })

  it('closing the dialog clears the store open flag', async () => {
    const { w, store } = await openTerminalJob()
    await w.get('[data-testid="pipeline-job-log-close"]').trigger('click')
    expect(store.open).toBe(false)
  })

  it('in-log search highlights matches and reports the count', async () => {
    const { w } = await openTerminalJob('log line one\nlog line two')
    await w.get('[data-testid="pipeline-job-log-search"]').setValue('line')
    expect(w.get('[data-testid="pipeline-job-log-search-count"]').text()).toBe('1/2')
    expect(w.findAll('[data-testid="pipeline-job-log-body"] mark').length).toBe(2)
  })

  it('next/prev cycles the active match', async () => {
    const { w } = await openTerminalJob('line line line')
    await w.get('[data-testid="pipeline-job-log-search"]').setValue('line')
    expect(w.get('[data-testid="pipeline-job-log-search-count"]').text()).toBe('1/3')
    await w.get('[data-testid="pipeline-job-log-search-next"]').trigger('click')
    expect(w.get('[data-testid="pipeline-job-log-search-count"]').text()).toBe('2/3')
    await w.get('[data-testid="pipeline-job-log-search-prev"]').trigger('click')
    await w.get('[data-testid="pipeline-job-log-search-prev"]').trigger('click')
    // wrap-around: 2 → 1 → 3
    expect(w.get('[data-testid="pipeline-job-log-search-count"]').text()).toBe('3/3')
  })

  it('shows 0/0 when the query has no matches', async () => {
    const { w } = await openTerminalJob('nothing to see')
    await w.get('[data-testid="pipeline-job-log-search"]').setValue('zzz')
    expect(w.get('[data-testid="pipeline-job-log-search-count"]').text()).toBe('0/0')
  })

  it('stops auto-refreshing when the polled status goes terminal', async () => {
    vi.useFakeTimers()
    try {
      fetchPipelineJobLogTail.mockResolvedValue({ text: 'running…', truncated: false })
      // The status poll reports the job has finished.
      listPipelineJobs.mockResolvedValue({
        path: '/mock/corpus',
        jobs: [{ job_id: 'job-run-1', status: 'succeeded' }],
      })
      const w = mountDialog()
      const store = usePipelineJobLogStore()
      store.viewLog({ corpusPath: '/mock/corpus', jobId: 'job-run-1', status: 'running', logRelpath: 'l' })
      await flushPromises()
      // Snapshot status is 'running' → live (poll timer scheduled, not yet fired).
      expect(w.text()).toContain('Auto-refreshing')
      // Advance to the first poll tick; status comes back terminal.
      await vi.advanceTimersByTimeAsync(3_000)
      await flushPromises()
      expect(w.text()).toContain('terminal state')
    } finally {
      vi.useRealTimers()
    }
  })
})
