// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import ShowsBrowse from './ShowsBrowse.vue'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

/**
 * UXS-015 / RFC-104 — ShowsBrowse renders the shows grid and opens a selected show
 * in the RIGHT SUBJECT RAIL (subject.focusShow), not in-panel. Exercises the real
 * ShowsView grid over a stubbed /api/corpus/feeds.
 */

function res(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubFeeds(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/corpus/feeds'))
        return res({
          path: '/corpus',
          feeds: [{ feed_id: 'alpha', display_title: 'Alpha Show', episode_count: 1 }],
        })
      return res({}, 404)
    }),
  )
}

beforeEach(() => {
  setActivePinia(createPinia())
  useShellStore().corpusPath = '/corpus'
  stubFeeds()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ShowsBrowse — grid opens a show in the rail', () => {
  it('renders the shows grid (no in-panel detail)', async () => {
    const w = mount(ShowsBrowse)
    await flushPromises()
    expect(w.find('[data-testid="shows-grid"]').exists()).toBe(true)
    expect(w.find('[data-testid="show-detail"]').exists()).toBe(false)
  })

  it('opens the selected show in the subject rail via focusShow', async () => {
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusShow')
    const w = mount(ShowsBrowse)
    await flushPromises()

    await w.find('[data-testid="shows-card-alpha"]').trigger('click')
    expect(spy).toHaveBeenCalledWith('alpha', { uiTitle: 'Alpha Show' })
    // The grid stays put — the show opens in the rail, not in-panel.
    expect(w.find('[data-testid="shows-grid"]').exists()).toBe(true)
    expect(w.find('[data-testid="show-detail"]').exists()).toBe(false)
  })
})
