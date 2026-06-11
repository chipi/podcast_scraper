// @vitest-environment happy-dom
import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useArtifactsStore } from '../../../stores/artifacts'
import { useGraphFilterStore } from '../../../stores/graphFilters'
import GraphSourcesChip from './GraphSourcesChip.vue'

const CHIP = '[data-testid="graph-chip-sources"]'
const POPOVER = '[data-testid="graph-popover-sources"]'

function fileOf(name: string, data: unknown): File {
  return new File([JSON.stringify(data)], name, { type: 'application/json' })
}

/** A FileList-like wrapper acceptable to ``loadFromLocalFiles``. */
function fakeFileList(files: File[]): FileList {
  return {
    length: files.length,
    item: (i: number) => files[i] ?? null,
    ...files,
    [Symbol.iterator]: function* () {
      yield* files
    },
  } as unknown as FileList
}

const GI = fileOf('ep1.gi.json', {
  episode_id: 'ep1',
  nodes: [
    { id: 'episode:ep1', type: 'Episode' },
    { id: 'i1', type: 'Insight', properties: { grounded: true } },
  ],
  edges: [],
})
const KG = fileOf('ep1.kg.json', {
  episode_id: 'ep1',
  nodes: [
    { id: 'episode:ep1', type: 'Episode' },
    { id: 't1', type: 'Topic' },
  ],
  edges: [],
})

/**
 * Seed a real GI+KG load so ``displayArtifact.kind === 'both'`` and the
 * graphFilters watcher populates ``gf.state``. The Sources chip only
 * renders for the ``both`` view, so this is the load shape under test.
 */
async function seedBothLoad(): Promise<void> {
  const artifacts = useArtifactsStore()
  await artifacts.loadFromLocalFiles(fakeFileList([GI, KG]))
  await flushPromises()
}

/** Seed a pure-GI load so ``displayArtifact.kind === 'gi'``. */
async function seedGiOnlyLoad(): Promise<void> {
  const artifacts = useArtifactsStore()
  await artifacts.loadFromLocalFiles(fakeFileList([GI]))
  await flushPromises()
}

const mountChip = () => mount(GraphSourcesChip, { attachTo: document.body })

describe('GraphSourcesChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders nothing when the view is not the merged GI+KG kind', async () => {
    await seedGiOnlyLoad()
    const w = mountChip()
    expect(w.find(CHIP).exists()).toBe(false)
  })

  it('renders nothing before any artifact is loaded', () => {
    const w = mountChip()
    expect(w.find(CHIP).exists()).toBe(false)
  })

  it('renders the inactive label for a both-view with all sources on', async () => {
    await seedBothLoad()
    const w = mountChip()
    expect(w.get(CHIP).text()).toContain('Sources ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('keeps the popover hidden until the chip is clicked, and toggles aria-expanded', async () => {
    await seedBothLoad()
    const w = mountChip()
    expect(w.get(POPOVER).isVisible()).toBe(false)
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(false)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('renders three checkboxes reflecting the current store state', async () => {
    await seedBothLoad()
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    const boxes = w.findAll<HTMLInputElement>(`${POPOVER} input[type="checkbox"]`)
    expect(boxes).toHaveLength(3)
    // GI on, KG on, hide-ungrounded off by default.
    expect(boxes[0].element.checked).toBe(true)
    expect(boxes[1].element.checked).toBe(true)
    expect(boxes[2].element.checked).toBe(false)
  })

  it('unchecking GI updates the store + flips the label to "GI only" inverse (KG only)', async () => {
    await seedBothLoad()
    const gf = useGraphFilterStore()
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    const boxes = w.findAll(`${POPOVER} input[type="checkbox"]`)

    // Toggle GI off → showGiLayer false, showKgLayer still true → "KG only".
    await boxes[0].trigger('change')
    expect(gf.state!.showGiLayer).toBe(false)
    expect(w.get(CHIP).text()).toContain('Sources: KG only ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
  })

  it('unchecking KG yields the "GI only" label', async () => {
    await seedBothLoad()
    const gf = useGraphFilterStore()
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    const boxes = w.findAll(`${POPOVER} input[type="checkbox"]`)
    await boxes[1].trigger('change')
    expect(gf.state!.showKgLayer).toBe(false)
    expect(w.get(CHIP).text()).toContain('Sources: GI only ▾')
  })

  it('disabling both GI and KG yields the "none" label', async () => {
    await seedBothLoad()
    const gf = useGraphFilterStore()
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    const boxes = w.findAll(`${POPOVER} input[type="checkbox"]`)
    await boxes[0].trigger('change')
    await boxes[1].trigger('change')
    expect(gf.state!.showGiLayer).toBe(false)
    expect(gf.state!.showKgLayer).toBe(false)
    expect(w.get(CHIP).text()).toContain('Sources: none ▾')
  })

  it('checking "Hide ungrounded" alone yields the "grounded" label', async () => {
    await seedBothLoad()
    const gf = useGraphFilterStore()
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    const boxes = w.findAll(`${POPOVER} input[type="checkbox"]`)
    await boxes[2].trigger('change')
    expect(gf.state!.hideUngroundedInsights).toBe(true)
    expect(w.get(CHIP).text()).toContain('Sources: grounded ▾')
  })

  it('combines a layer filter with hide-ungrounded into one label ("+grounded")', async () => {
    await seedBothLoad()
    const gf = useGraphFilterStore()
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    const boxes = w.findAll(`${POPOVER} input[type="checkbox"]`)
    // KG off (→ "GI only") + hide ungrounded (→ "+grounded").
    await boxes[1].trigger('change')
    await boxes[2].trigger('change')
    expect(gf.state!.showKgLayer).toBe(false)
    expect(gf.state!.hideUngroundedInsights).toBe(true)
    expect(w.get(CHIP).text()).toContain('Sources: GI only +grounded ▾')
  })

  it('re-checking a previously disabled layer returns to the inactive label', async () => {
    await seedBothLoad()
    const gf = useGraphFilterStore()
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    const boxes = w.findAll(`${POPOVER} input[type="checkbox"]`)
    await boxes[0].trigger('change')
    expect(w.get(CHIP).text()).toContain('Sources: KG only ▾')
    // Toggle GI back on → both layers on, nothing hidden → inactive.
    await boxes[0].trigger('change')
    expect(gf.state!.showGiLayer).toBe(true)
    expect(w.get(CHIP).text()).toContain('Sources ▾')
    expect(w.get(CHIP).text()).not.toContain('Sources:')
  })
})
