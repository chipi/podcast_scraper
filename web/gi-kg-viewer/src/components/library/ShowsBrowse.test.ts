// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import ShowsBrowse from './ShowsBrowse.vue'
import { useShellStore } from '../../stores/shell'

/**
 * UXS-015 / RFC-104 — ShowsBrowse ties the grid to the detail: selecting a show
 * replaces the grid in-panel; Back returns to it. Exercises the real ShowsView +
 * ShowDetailView children over stubbed /api/corpus endpoints.
 */

function res(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubApi(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/corpus/feeds'))
        return res({
          path: '/corpus',
          feeds: [{ feed_id: 'alpha', display_title: 'Alpha Show', episode_count: 1 }],
        })
      if (url.includes('/api/corpus/episodes'))
        return res({
          path: '/corpus',
          feed_id: 'alpha',
          items: [
            {
              metadata_relative_path: 'metadata/a1.metadata.json',
              feed_id: 'alpha',
              episode_id: 'a1',
              episode_title: 'Alpha Episode One',
              publish_date: '2026-06-01',
            },
          ],
          next_cursor: null,
        })
      return res({}, 404)
    }),
  )
}

beforeEach(() => {
  setActivePinia(createPinia())
  useShellStore().corpusPath = '/corpus'
  stubApi()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ShowsBrowse — grid ⇄ detail', () => {
  it('starts on the grid', async () => {
    const w = mount(ShowsBrowse)
    await flushPromises()
    expect(w.find('[data-testid="shows-grid"]').exists()).toBe(true)
    expect(w.find('[data-testid="show-detail"]').exists()).toBe(false)
  })

  it('opens a show detail on selection and returns on Back', async () => {
    const w = mount(ShowsBrowse)
    await flushPromises()

    await w.find('[data-testid="shows-card-alpha"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="show-detail"]').exists()).toBe(true)
    expect(w.find('[data-testid="show-detail"]').text()).toContain('Alpha Show')

    await w.find('[data-testid="show-detail-back"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="shows-grid"]').exists()).toBe(true)
    expect(w.find('[data-testid="show-detail"]').exists()).toBe(false)
  })
})
