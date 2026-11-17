// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { IndexStatsBody, IndexStatsEnvelope } from '../../api/indexStatsApi'
import { useIndexStatsStore } from '../../stores/indexStats'
import IndexStatusCard from './IndexStatusCard.vue'

const NOW = Date.parse('2026-06-15T12:00:00Z')

function stats(over: Partial<IndexStatsBody> = {}): IndexStatsBody {
  return {
    total_vectors: 12345,
    doc_type_counts: {},
    feeds_indexed: ['a', 'b'],
    embedding_model: 'm',
    embedding_dim: 384,
    last_updated: '2026-06-10T12:00:00Z',
    index_size_bytes: 1,
    ...over,
  }
}

function mountCard(env: IndexStatsEnvelope | null) {
  const store = useIndexStatsStore()
  store.indexEnvelope = env
  return { w: mount(IndexStatusCard), store }
}

describe('IndexStatusCard', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('shows dashes when no index envelope is loaded', () => {
    const { w } = mountCard(null)
    expect(w.text()).toContain('Last rebuilt: —')
    expect(w.text()).toContain('—')
    expect(w.text()).not.toContain('Rebuild recommended')
    expect(w.text()).not.toContain('Last rebuild error')
  })

  it('shows dashes when the index is unavailable', () => {
    const { w } = mountCard({ available: false, stats: stats() })
    expect(w.text()).not.toContain('vectors')
  })

  it('renders vector count and pluralised feed count', () => {
    const { w } = mountCard({ available: true, stats: stats() })
    expect(w.text()).toContain(`${(12345).toLocaleString()} vectors · 2 feeds listed`)
    expect(w.text()).toContain('5 days ago')
  })

  it('uses singular feed wording and tolerates a missing feed list', () => {
    const one = mountCard({ available: true, stats: stats({ feeds_indexed: ['x'] }) })
    expect(one.w.text()).toContain('1 feed listed')
    const none = mountCard({
      available: true,
      stats: stats({ feeds_indexed: undefined as unknown as string[] }),
    })
    expect(none.w.text()).toContain('0 feeds listed')
  })

  it('formats the rebuild age as today, 1 day ago and N days ago', () => {
    expect(
      mountCard({ available: true, stats: stats({ last_updated: '2026-06-15T01:00:00Z' }) }).w.text(),
    ).toContain('Last rebuilt: today')
    expect(
      mountCard({ available: true, stats: stats({ last_updated: '2026-06-14T01:00:00Z' }) }).w.text(),
    ).toContain('Last rebuilt: 1 day ago')
    expect(
      mountCard({ available: true, stats: stats({ last_updated: '2026-06-01T12:00:00Z' }) }).w.text(),
    ).toContain('Last rebuilt: 14 days ago')
  })

  it('treats a future timestamp as today', () => {
    const { w } = mountCard({ available: true, stats: stats({ last_updated: '2026-07-01T00:00:00Z' }) })
    expect(w.text()).toContain('Last rebuilt: today')
  })

  it('echoes an unparseable timestamp verbatim', () => {
    const { w } = mountCard({ available: true, stats: stats({ last_updated: 'sometime' }) })
    expect(w.text()).toContain('Last rebuilt: sometime')
  })

  it('treats a blank timestamp as unknown', () => {
    const { w } = mountCard({ available: true, stats: stats({ last_updated: '   ' }) })
    expect(w.text()).toContain('Last rebuilt: —')
  })

  it('flags a recommended rebuild and the last rebuild error', () => {
    const { w } = mountCard({
      available: true,
      stats: stats(),
      reindex_recommended: true,
      rebuild_last_error: 'disk full',
    })
    expect(w.text()).toContain('Rebuild recommended')
    expect(w.text()).toContain('Last rebuild error: disk full')
  })

  it('routes the manage button to the Configuration dialog request', async () => {
    const { w, store } = mountCard({ available: true, stats: stats() })
    const before = store.dialogOpenNonce
    await w.get('[data-testid="index-status-manage"]').trigger('click')
    expect(store.dialogOpenNonce).toBe(before + 1)
  })
})
