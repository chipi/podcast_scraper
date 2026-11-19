import { afterEach, describe, expect, it, vi } from 'vitest'

import { submitPipelineJob } from './jobsApi'

function mockFetchOk(body: unknown): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      status: 202,
      text: async () => '',
      json: async () => body,
    })) as unknown as typeof fetch,
  )
}

function calledUrl(): string {
  return (fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls[0][0] as string
}

describe('submitPipelineJob', () => {
  const accepted = { job_id: 'j1', status: 'queued', corpus_path: '/c' }

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('POSTs the whole-batch run when no scope is given (default unchanged)', async () => {
    mockFetchOk(accepted)
    await submitPipelineJob('/c')
    const url = calledUrl()
    expect(url).toContain('/api/jobs?')
    expect(url).toContain('path=%2Fc')
    expect(url).not.toContain('feed=')
  })

  it('serializes a per-feed scope + incremental knobs (P1.4)', async () => {
    mockFetchOk(accepted)
    await submitPipelineJob('/c', {
      feed: 'https://a.example/1.xml',
      skipExisting: true,
      maxEpisodes: 5,
      episodeOrder: 'newest',
    })
    const url = calledUrl()
    expect(url).toContain('feed=https%3A%2F%2Fa.example%2F1.xml')
    expect(url).toContain('skip_existing=true')
    expect(url).toContain('max_episodes=5')
    expect(url).toContain('episode_order=newest')
  })

  it('omits unset knobs and drops an empty feed', async () => {
    mockFetchOk(accepted)
    await submitPipelineJob('/c', { feed: '   ', skipExisting: true })
    const url = calledUrl()
    expect(url).not.toContain('feed=')
    expect(url).not.toContain('skip_existing') // knobs apply only with a feed
  })
})
