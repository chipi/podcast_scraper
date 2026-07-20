// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

const searchCorpusMock = vi.fn()
vi.mock('../../api/searchApi', () => ({
  searchCorpus: (...args: unknown[]) => searchCorpusMock(...args),
}))

import CommandPalette from './CommandPalette.vue'
import type { SearchHit } from '../../api/searchApi'
import { useShellStore } from '../../stores/shell'
import { useUserPreferencesStore } from '../../stores/userPreferences'

/**
 * CommandPalette is the Cmd-K / `/` shell overlay from Search v3 §S3
 * (RFC-107 §4). It's the palette-shape sibling of the launcher — same
 * store bindings, different affordance. This spec asserts the structural
 * contract + the parent-facing emissions; it does NOT reach into
 * SearchPanel's internals (the palette runs its own fetch).
 */
function mountPalette() {
  return mount(CommandPalette, { attachTo: document.body })
}

function fakeHit(over: Partial<SearchHit> = {}): SearchHit {
  return {
    doc_id: 'hit-1',
    score: 0.5,
    text: 'A representative hit body.',
    metadata: {
      doc_type: 'kg_topic',
      source_id: 'topic:demo',
      topic_label: 'Demo Topic',
    },
    ...over,
  } as unknown as SearchHit
}

describe('CommandPalette (Cmd-K shell overlay)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    searchCorpusMock.mockReset()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('is closed by default and mounts nothing', () => {
    const w = mountPalette()
    expect(document.querySelector('[data-testid="command-palette"]')).toBeNull()
    w.unmount()
  })

  it('open() renders the overlay with input + empty-state placeholders', async () => {
    const w = mountPalette()
    ;(w.vm as unknown as { open: () => void }).open()
    await flushPromises()
    const overlay = document.querySelector('[data-testid="command-palette"]')
    expect(overlay).not.toBeNull()
    expect(
      document.querySelector('[data-testid="command-palette-input"]'),
    ).not.toBeNull()
    // Empty state: Saved placeholder + Recent honest empty (no USERPREFS-1 key set).
    expect(
      document.querySelector('[data-testid="command-palette-saved-empty"]'),
    ).not.toBeNull()
    expect(
      document.querySelector('[data-testid="command-palette-recent-empty"]'),
    ).not.toBeNull()
    w.unmount()
  })

  it('renders recent queries from USERPREFS-1 when the key is set', async () => {
    const prefs = useUserPreferencesStore()
    prefs.set('search.recentQueries', [
      { q: 'llm strategy', ts: 1 },
      { q: 'cell therapy', ts: 2 },
    ])
    const w = mountPalette()
    ;(w.vm as unknown as { open: () => void }).open()
    await flushPromises()
    expect(
      document.querySelector('[data-testid="command-palette-recent-list"]'),
    ).not.toBeNull()
    expect(
      document.querySelector('[data-testid="command-palette-recent-empty"]'),
    ).toBeNull()
    w.unmount()
  })

  it('runs a debounced live search and renders the 3 action buttons per hit', async () => {
    const shell = useShellStore()
    shell.corpusPath = '/tmp/corpus'
    searchCorpusMock.mockResolvedValue({ results: [fakeHit()], error: null })
    const w = mountPalette()
    ;(w.vm as unknown as { open: () => void }).open()
    await flushPromises()

    const input = document.querySelector<HTMLInputElement>(
      '[data-testid="command-palette-input"]',
    )!
    input.value = 'demo'
    input.dispatchEvent(new Event('input'))
    await flushPromises()

    // Debounce: nothing fired yet.
    expect(searchCorpusMock).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(210)
    await flushPromises()

    expect(searchCorpusMock).toHaveBeenCalledTimes(1)
    expect(
      document.querySelectorAll(
        '[data-testid="command-palette-action-open-workspace"]',
      ),
    ).toHaveLength(1)
    expect(
      document.querySelectorAll('[data-testid="command-palette-action-pin-rail"]'),
    ).toHaveLength(1)
    expect(
      document.querySelectorAll(
        '[data-testid="command-palette-action-show-graph"]',
      ),
    ).toHaveLength(1)
    w.unmount()
  })

  it('emits open-in-workspace with the current query and closes on click', async () => {
    const shell = useShellStore()
    shell.corpusPath = '/tmp/corpus'
    searchCorpusMock.mockResolvedValue({ results: [fakeHit()], error: null })
    const w = mountPalette()
    ;(w.vm as unknown as { open: () => void }).open()
    await flushPromises()

    const input = document.querySelector<HTMLInputElement>(
      '[data-testid="command-palette-input"]',
    )!
    input.value = 'demo query'
    input.dispatchEvent(new Event('input'))
    await vi.advanceTimersByTimeAsync(210)
    await flushPromises()

    ;(
      document.querySelector<HTMLButtonElement>(
        '[data-testid="command-palette-action-open-workspace"]',
      )!
    ).click()
    await flushPromises()

    expect(w.emitted('open-in-workspace')?.[0]).toEqual(['demo query'])
    // Closed after emit.
    expect(document.querySelector('[data-testid="command-palette"]')).toBeNull()
    w.unmount()
  })

  it('emits show-on-graph with a resolvable cy id for kg_topic hits', async () => {
    const shell = useShellStore()
    shell.corpusPath = '/tmp/corpus'
    searchCorpusMock.mockResolvedValue({ results: [fakeHit()], error: null })
    const w = mountPalette()
    ;(w.vm as unknown as { open: () => void }).open()
    await flushPromises()

    const input = document.querySelector<HTMLInputElement>(
      '[data-testid="command-palette-input"]',
    )!
    input.value = 'demo'
    input.dispatchEvent(new Event('input'))
    await vi.advanceTimersByTimeAsync(210)
    await flushPromises()

    ;(
      document.querySelector<HTMLButtonElement>(
        '[data-testid="command-palette-action-show-graph"]',
      )!
    ).click()
    await flushPromises()

    const emitted = w.emitted('show-on-graph')?.[0]
    expect(emitted).toBeDefined()
    expect(typeof emitted![0]).toBe('string')
    expect((emitted![0] as string).length).toBeGreaterThan(0)
    w.unmount()
  })

  it('close() clears the query and unmounts the overlay', async () => {
    const w = mountPalette()
    const vm = w.vm as unknown as { open: () => void; close: () => void }
    vm.open()
    await flushPromises()
    expect(document.querySelector('[data-testid="command-palette"]')).not.toBeNull()
    vm.close()
    await flushPromises()
    expect(document.querySelector('[data-testid="command-palette"]')).toBeNull()
    w.unmount()
  })
})
