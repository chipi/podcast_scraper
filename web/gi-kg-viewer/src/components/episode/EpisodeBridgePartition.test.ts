// @vitest-environment happy-dom
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { BridgePartitionSummary } from '../../api/corpusLibraryApi'
import EpisodeBridgePartition from './EpisodeBridgePartition.vue'

/**
 * #656 Stage B — invariants on the per-episode bridge partition indicator.
 *
 * Partition counts flow from ``bridge.json`` through the backend-
 * computed ``BridgePartitionSummary`` to the component. The shape is
 * simple (three integers + total) but the visual guarantees matter:
 *   - the row self-hides when the partition is absent so legacy
 *     episodes don't render an empty frame
 *   - counts render via safe interpolation (no v-html)
 *   - all three partition keys (``gi_only`` / ``both`` / ``kg_only``)
 *     are referenced so a future backend rename surfaces here
 *   - screen-reader wrapper is announced as a partition group
 *
 * Like the other Stage B / foundation guards, this is a source-level
 * check while the viewer waits on a broader component-test harness
 * (tracked as POST-#656 work).
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const COMPONENT = resolve(HERE, 'EpisodeBridgePartition.vue')
const source = readFileSync(COMPONENT, 'utf-8')

describe('EpisodeBridgePartition.vue — shape + safety invariants', () => {
  it('has no raw-HTML sinks', () => {
    expect(source).not.toMatch(/\sv-html\s*=/)
    expect(source).not.toMatch(/\.innerHTML\s*=/)
  })

  it('references all three partition keys', () => {
    // If a backend rename drops a key, this fires.
    expect(source).toContain('gi_only')
    expect(source).toContain('kg_only')
    expect(source).toContain("'both'")
  })

  it('self-hides when partition is absent', () => {
    // ``hasData`` guards the outer element so the "Bridge partition"
    // label doesn't render an empty row for legacy episodes.
    expect(source).toMatch(/v-if="hasData"/)
    expect(source).toMatch(/hasData\s*=\s*computed/)
  })

  it('self-hides on empty partition (total = 0)', () => {
    // total=0 means "bridge file is present but the builder emitted
    // no identities" — we treat that as "no data" visually.
    expect(source).toMatch(/partition\.total\s*>\s*0/)
  })

  it('renders integer counts via safe interpolation', () => {
    expect(source).toMatch(/\{\{\s*cell\.count\s*\}\}/)
    expect(source).toMatch(/\{\{\s*partition!\.total\s*\}\}/)
  })

  it('emphasises the "Both" cell (overlap signal #654 tunes for)', () => {
    // The "Both" column is what the bridge threshold was rewritten to
    // surface meaningfully; it gets the bolder accent.
    expect(source).toMatch(/font-semibold/)
  })

  it('announces cells via aria-label so screen readers get all three counts', () => {
    expect(source).toMatch(/:aria-label="`\$\{cell\.label\}: \$\{cell\.count\} identities`"/)
  })

  it('exposes stable test-ids per partition cell', () => {
    expect(source).toMatch(/`bridge-partition-\$\{cell\.key\}`/)
    expect(source).toMatch(/'episode-bridge-partition'/)
  })
})

// ── @vue/test-utils mount tests (render states + props + a11y) ────────────────

function partitionOf(over: Partial<BridgePartitionSummary> = {}): BridgePartitionSummary {
  return { gi_only: 3, kg_only: 4, both: 5, total: 12, ...over }
}

const ROW = '[data-testid="episode-bridge-partition"]'

describe('EpisodeBridgePartition.vue — mount behaviour', () => {
  it('renders nothing when partition is null', () => {
    const w = mount(EpisodeBridgePartition, { props: { partition: null } })
    expect(w.find(ROW).exists()).toBe(false)
  })

  it('renders nothing when partition is undefined', () => {
    const w = mount(EpisodeBridgePartition, { props: { partition: undefined } })
    expect(w.find(ROW).exists()).toBe(false)
  })

  it('renders nothing when partition total is 0 (present but empty)', () => {
    const w = mount(EpisodeBridgePartition, {
      props: { partition: partitionOf({ gi_only: 0, kg_only: 0, both: 0, total: 0 }) },
    })
    expect(w.find(ROW).exists()).toBe(false)
  })

  it('renders the three partition cells with their counts when populated', () => {
    const w = mount(EpisodeBridgePartition, { props: { partition: partitionOf() } })
    expect(w.find(ROW).exists()).toBe(true)
    expect(w.get('[data-testid="bridge-partition-gi_only"]').text()).toContain('3')
    expect(w.get('[data-testid="bridge-partition-both"]').text()).toContain('5')
    expect(w.get('[data-testid="bridge-partition-kg_only"]').text()).toContain('4')
  })

  it('shows the total identities in the header', () => {
    const w = mount(EpisodeBridgePartition, { props: { partition: partitionOf({ total: 99 }) } })
    expect(w.get(ROW).text()).toContain('99 identities')
  })

  it('labels each cell and exposes a screen-reader aria-label with the count', () => {
    const w = mount(EpisodeBridgePartition, { props: { partition: partitionOf() } })
    const both = w.get('[data-testid="bridge-partition-both"]')
    expect(both.text()).toContain('Both')
    expect(both.attributes('aria-label')).toBe('Both: 5 identities')
    expect(both.attributes('title')).toContain('overlap signal')
  })

  it('wraps the cells in a screen-reader partition group', () => {
    const w = mount(EpisodeBridgePartition, { props: { partition: partitionOf() } })
    const row = w.get(ROW)
    expect(row.attributes('role')).toBe('group')
    expect(row.attributes('aria-label')).toBe('Bridge partition summary')
  })

  it('honours a custom dataTestid for the row wrapper', () => {
    const w = mount(EpisodeBridgePartition, {
      props: { partition: partitionOf(), dataTestid: 'custom-bridge' },
    })
    expect(w.find('[data-testid="custom-bridge"]').exists()).toBe(true)
    expect(w.find(ROW).exists()).toBe(false)
    // Cell testids are independent of the row testid override.
    expect(w.find('[data-testid="bridge-partition-both"]').exists()).toBe(true)
  })
})
