/**
 * @vitest-environment happy-dom
 *
 * Derived-state coverage for the index-stats store: the read-model computeds + the dialog nonce.
 * Pure of timers/network — exercises indexRows / indexHealthBanner (+ expandIndexReasonLines) /
 * rebuildActionsDisabled / requestOpenIndexDialog against fixed envelopes.
 */
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { IndexStatsEnvelope } from '../api/indexStatsApi'
import { useIndexStatsStore } from './indexStats'
import { useShellStore } from './shell'

function envelope(overrides: Record<string, unknown> = {}): IndexStatsEnvelope {
  return { available: false, ...overrides } as unknown as IndexStatsEnvelope
}

describe('indexStats store — derived read-model', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('indexRows is empty until an available envelope with stats is present', () => {
    const store = useIndexStatsStore()
    expect(store.indexRows).toEqual([])

    store.indexEnvelope = envelope({
      available: true,
      index_path: '/corpus/search',
      stats: {
        total_vectors: 42,
        embedding_model: 'all-MiniLM-L6-v2',
        embedding_dim: 384,
        last_updated: '2026-08-10',
        index_size_bytes: 2048,
        feeds_indexed: ['feed-a', 'feed-b'],
      },
    })

    const rows = store.indexRows
    expect(rows.find((r) => r.k === 'Total vectors')?.v).toBe('42')
    expect(rows.find((r) => r.k === 'Feeds indexed')?.v).toBe('feed-a, feed-b')
    expect(rows.find((r) => r.k === 'Embedding model')?.v).toBe('all-MiniLM-L6-v2')
  })

  it('indexHealthBanner is null without healthStatus, and warns when reindex is recommended', () => {
    const store = useIndexStatsStore()
    const shell = useShellStore()

    store.indexEnvelope = envelope({ available: true, reindex_reasons: ['embedding_model_changed'] })
    expect(store.indexHealthBanner).toBeNull() // no health signal yet

    shell.healthStatus = 'ok'
    store.indexEnvelope = envelope({
      available: true,
      reindex_recommended: true,
      reindex_reasons: ['embedding_model_changed'],
    })
    const banner = store.indexHealthBanner
    expect(banner?.kind).toBe('warn')
    expect(banner?.lines.length).toBeGreaterThan(0)
  })

  it('rebuildActionsDisabled reflects health + rebuild-in-progress', () => {
    const store = useIndexStatsStore()
    const shell = useShellStore()
    expect(store.rebuildActionsDisabled).toBe(true) // disabled while health is unknown

    shell.healthStatus = 'ok'
    store.indexEnvelope = envelope({ available: true })
    expect(store.rebuildActionsDisabled).toBe(false)

    store.indexEnvelope = envelope({ available: true, rebuild_in_progress: true })
    expect(store.rebuildActionsDisabled).toBe(true)
  })

  it('requestOpenIndexDialog bumps the dialog-open nonce', () => {
    const store = useIndexStatsStore()
    const before = store.dialogOpenNonce
    store.requestOpenIndexDialog()
    expect(store.dialogOpenNonce).toBe(before + 1)
  })
})
