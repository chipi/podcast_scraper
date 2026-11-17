// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

const getScheduledJobs = vi.fn()
vi.mock('../../api/scheduledJobsApi', () => ({
  getScheduledJobs: (...a: unknown[]) => getScheduledJobs(...a),
}))

import ScheduledJobsSection from './ScheduledJobsSection.vue'

afterEach(() => {
  getScheduledJobs.mockReset()
})

function mountSection() {
  return mount(ScheduledJobsSection, {
    props: { corpusPath: '/mock/corpus', active: true },
    attachTo: document.body,
  })
}

describe('ScheduledJobsSection', () => {
  it('renders rows; disabled job shows — for next run', async () => {
    getScheduledJobs.mockResolvedValue({
      path: '/mock/corpus',
      scheduler_running: true,
      timezone: 'UTC',
      jobs: [
        {
          name: 'nightly',
          cron: '0 2 * * *',
          enabled: true,
          kind: 'enrichment',
          next_run_at: '2099-01-01T02:00:00Z',
        },
        { name: 'weekly', cron: '0 3 * * 0', enabled: false, kind: 'pipeline', next_run_at: null },
      ],
    })
    const w = mountSection()
    await flushPromises()
    expect(w.get('[data-testid="scheduled-jobs-row-0"]').text()).toContain('nightly')
    expect(w.get('[data-testid="scheduled-jobs-next-0"]').text()).toMatch(/^in /)
    expect(w.get('[data-testid="scheduled-jobs-next-1"]').text()).toBe('—')
    // #1069: the kind is surfaced so enrichment schedules read as peers of ingestion.
    expect(w.get('[data-testid="scheduled-jobs-kind-0"]').text()).toBe('enrichment')
    expect(w.get('[data-testid="scheduled-jobs-kind-1"]').text()).toBe('pipeline')
  })

  it('flags an invalid cron', async () => {
    getScheduledJobs.mockResolvedValue({
      path: '/mock/corpus',
      scheduler_running: true,
      timezone: 'UTC',
      jobs: [
        { name: 'broken', cron: 'not a cron', enabled: true, kind: 'pipeline', next_run_at: null },
      ],
    })
    const w = mountSection()
    await flushPromises()
    expect(w.find('[data-testid="scheduled-jobs-invalid-cron"]').exists()).toBe(true)
    expect(w.get('[data-testid="scheduled-jobs-next-0"]').text()).toContain('invalid cron')
  })

  it('emits toggle with the flipped value', async () => {
    getScheduledJobs.mockResolvedValue({
      path: '/mock/corpus',
      scheduler_running: true,
      timezone: 'UTC',
      jobs: [
        {
          name: 'nightly',
          cron: '0 2 * * *',
          enabled: true,
          kind: 'pipeline',
          next_run_at: '2099-01-01T02:00:00Z',
        },
      ],
    })
    const w = mountSection()
    await flushPromises()
    await w.get('[data-testid="scheduled-jobs-toggle-0"]').trigger('click')
    expect(w.emitted('toggle')?.[0]).toEqual(['nightly', false])
  })

  it('shows the empty state when there are no jobs', async () => {
    getScheduledJobs.mockResolvedValue({
      path: '/mock/corpus',
      scheduler_running: false,
      timezone: 'UTC',
      jobs: [],
    })
    const w = mountSection()
    await flushPromises()
    expect(w.find('[data-testid="scheduled-jobs-empty"]').exists()).toBe(true)
  })
})
