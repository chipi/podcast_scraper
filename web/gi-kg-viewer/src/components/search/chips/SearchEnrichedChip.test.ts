// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import SearchEnrichedChip from './SearchEnrichedChip.vue'
import { useSearchStore } from '../../../stores/search'
import { useShellStore } from '../../../stores/shell'

/**
 * Enriched-answer chip — Search v3 §S5. Tri-state store field
 * (``null`` = auto, ``true`` / ``false`` = explicit) with the effective
 * on/off tied to the server's capability signal for null, else the
 * explicit value.
 */
function mountChip() {
  return mount(SearchEnrichedChip, { attachTo: document.body })
}

describe('SearchEnrichedChip (Search v3 §S5)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders disabled with off-state when the server does not advertise enrichment', () => {
    const w = mountChip()
    const btn = w.get('[data-testid="search-chip-enriched"]')
    expect((btn.element as HTMLButtonElement).disabled).toBe(true)
    expect(btn.attributes('aria-pressed')).toBe('false')
    expect(btn.text()).toContain('Enriched')
    expect(btn.attributes('title')).toContain('not configured')
    w.unmount()
  })

  it('auto-adopts capability-on when the store filter is null (default)', () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const w = mountChip()
    const btn = w.get('[data-testid="search-chip-enriched"]')
    expect((btn.element as HTMLButtonElement).disabled).toBe(false)
    expect(btn.attributes('aria-pressed')).toBe('true')
    expect(btn.text()).toContain('Enriched ✓')
    w.unmount()
  })

  it('click while auto-on flips the store filter to explicit false', async () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const search = useSearchStore()
    expect(search.filters.enrichResults).toBeNull()
    const w = mountChip()
    await w.get('[data-testid="search-chip-enriched"]').trigger('click')
    expect(search.filters.enrichResults).toBe(false)
    // Chip now reflects off-state.
    expect(w.get('[data-testid="search-chip-enriched"]').attributes('aria-pressed')).toBe('false')
    w.unmount()
  })

  it('click while off flips to true (explicit opt-in even without capability)', async () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const search = useSearchStore()
    search.filters.enrichResults = false
    const w = mountChip()
    await w.get('[data-testid="search-chip-enriched"]').trigger('click')
    expect(search.filters.enrichResults).toBe(true)
    w.unmount()
  })

  it('click is a no-op when the server capability is off (disabled)', async () => {
    const search = useSearchStore()
    const w = mountChip()
    await w.get('[data-testid="search-chip-enriched"]').trigger('click')
    // Disabled buttons still receive click events in JSDOM/happy-dom, but our
    // handler bails on !capabilityOn — the filter must remain null (default).
    expect(search.filters.enrichResults).toBeNull()
    w.unmount()
  })
})
