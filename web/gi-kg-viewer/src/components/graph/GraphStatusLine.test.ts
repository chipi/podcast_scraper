// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphExpansionStore } from '../../stores/graphExpansion'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import type { ArtifactData } from '../../types/artifact'
import { localYmdDaysAgo } from '../../utils/localCalendarDate'
import { parseArtifact } from '../../utils/parsing'
import GraphStatusLine from './GraphStatusLine.vue'

const SUMMARY = '[data-testid="graph-status-line"]'
const CONTROLS = '[data-testid="graph-status-line-controls"]'

/**
 * Push a single ``.gi.json`` artifact into the artifacts store so the
 * graphFilters ``filteredArtifact`` computed (and therefore the summary
 * counts) reflect real store state. A lone GI artifact passes through
 * ``buildDisplayArtifact`` unmodified (no id prefixing / dedup), so the
 * node set is predictable. Quote / Speaker types are hidden by the graph
 * default visibility; Episode / Topic survive.
 */
function loadGiArtifact(data: ArtifactData): void {
  const artifacts = useArtifactsStore()
  artifacts.corpusPath = '/corpus'
  artifacts.parsedList = [parseArtifact('ep1.gi.json', data)]
}

const mountSummary = (props: Record<string, unknown> = {}) =>
  mount(GraphStatusLine, {
    props: { variant: 'summary', ...props },
    attachTo: document.body,
  })

const mountControls = (props: Record<string, unknown> = {}) =>
  mount(GraphStatusLine, {
    props: { variant: 'controls', ...props },
    attachTo: document.body,
  })

describe('GraphStatusLine', () => {
  beforeEach(() => setActivePinia(createPinia()))

  describe('variant rendering', () => {
    it('summary variant renders the counts strip, not the controls', () => {
      const w = mountSummary()
      expect(w.find(SUMMARY).exists()).toBe(true)
      expect(w.find(CONTROLS).exists()).toBe(false)
      // Distinct summary content: lens label + the three count labels.
      expect(w.text()).toContain('Showing')
      expect(w.text()).toContain('episodes')
      expect(w.text()).toContain('nodes')
      expect(w.text()).toContain('components')
    })

    it('controls variant renders preset buttons + Since input, not the counts strip', () => {
      const w = mountControls()
      expect(w.find(CONTROLS).exists()).toBe(true)
      expect(w.find(SUMMARY).exists()).toBe(false)
      const labels = w.findAll('button').map((b) => b.text())
      expect(labels).toContain('7d')
      expect(labels).toContain('30d')
      expect(labels).toContain('90d')
      expect(labels).toContain('All')
      expect(w.find('[data-testid="graph-status-since-input"]').exists()).toBe(true)
    })
  })

  describe('summary counts reflect store state', () => {
    it('shows zeros when no artifact is loaded', () => {
      const w = mountSummary()
      expect(w.get('[data-testid="graph-status-episode-count"]').text()).toBe('0')
      expect(w.get('[data-testid="graph-status-node-count"]').text()).toBe('0')
      expect(w.get('[data-testid="graph-status-component-count"]').text()).toBe('0')
    })

    it('reflects episode / node / component counts from the filtered artifact', () => {
      loadGiArtifact({
        model_version: 'm',
        prompt_version: 'p',
        nodes: [
          { id: 'e1', type: 'Episode' },
          { id: 'e2', type: 'Episode' },
          { id: 't1', type: 'Topic' },
        ],
        // e1<->t1 connected; e2 isolated → 2 weak components.
        edges: [{ from: 'e1', to: 't1', type: 'mentions' }],
      })
      const w = mountSummary()
      expect(w.get('[data-testid="graph-status-episode-count"]').text()).toBe('2')
      expect(w.get('[data-testid="graph-status-node-count"]').text()).toBe('3')
      expect(w.get('[data-testid="graph-status-component-count"]').text()).toBe('2')
    })

    it('hidden-by-default Quote nodes are excluded from the node count', () => {
      loadGiArtifact({
        model_version: 'm',
        prompt_version: 'p',
        nodes: [
          { id: 'e1', type: 'Episode' },
          { id: 'q1', type: 'Quote' },
        ],
        edges: [],
      })
      const w = mountSummary()
      // Quote is off by default → only the Episode survives.
      expect(w.get('[data-testid="graph-status-node-count"]').text()).toBe('1')
      expect(w.get('[data-testid="graph-status-episode-count"]').text()).toBe('1')
    })

    it('formats large node counts with a k suffix', () => {
      const nodes = Array.from({ length: 1500 }, (_, i) => ({
        id: `t${i}`,
        type: 'Topic',
      }))
      loadGiArtifact({ model_version: 'm', prompt_version: 'p', nodes, edges: [] })
      const w = mountSummary()
      expect(w.get('[data-testid="graph-status-node-count"]').text()).toBe('1.5k')
    })
  })

  describe('lens label + capped flag', () => {
    it('shows "all time" by default and reacts to a preset since-date', async () => {
      const w = mountSummary()
      expect(w.get('[data-testid="graph-status-lens-label"]').text()).toContain('all time')

      const ge = useGraphExplorerStore()
      ge.setPresetDays(7)
      await w.vm.$nextTick()
      expect(w.get('[data-testid="graph-status-lens-label"]').text()).toContain(
        'last 7 days',
      )
    })

    it('renders a custom "since" label for a non-preset date', async () => {
      const w = mountSummary()
      const ge = useGraphExplorerStore()
      ge.setSinceYmd('2020-01-15')
      await w.vm.$nextTick()
      expect(w.get('[data-testid="graph-status-lens-label"]').text()).toContain(
        'since 2020-01-15',
      )
    })

    it('renders the "(capped)" marker only when the store flags it', async () => {
      const w = mountSummary()
      expect(w.find('[data-testid="graph-status-capped"]').exists()).toBe(false)
      const ge = useGraphExplorerStore()
      ge.setLastAutoLoadCapped(true)
      await w.vm.$nextTick()
      expect(w.find('[data-testid="graph-status-capped"]').exists()).toBe(true)
    })
  })

  describe('embedded / bare flag branches', () => {
    it('summary "bare" omits the outer border/padding chrome class', () => {
      const plain = mountSummary({ bare: false })
      expect(plain.get(SUMMARY).classes()).toContain('border-b')

      const bare = mountSummary({ bare: true })
      expect(bare.get(SUMMARY).classes()).not.toContain('border-b')
      expect(bare.get(SUMMARY).classes()).toContain('text-muted')
    })

    it('controls "embedded" uses the smaller preset button class', () => {
      const normal = mountControls({ embedded: false })
      const normalBtn = normal.findAll('button')[0]!
      expect(normalBtn.classes()).toContain('text-[10px]')

      const embedded = mountControls({ embedded: true })
      const embeddedBtn = embedded.findAll('button')[0]!
      expect(embeddedBtn.classes()).toContain('text-[9px]')
    })
  })

  describe('controls preset buttons drive the store + emit request-reload', () => {
    const clickPreset = async (label: string) => {
      const w = mountControls()
      const btn = w.findAll('button').find((b) => b.text() === label)!
      await btn.trigger('click')
      return w
    }

    it('7d sets a 7-day since lower bound and emits request-reload', async () => {
      const w = await clickPreset('7d')
      const ge = useGraphExplorerStore()
      expect(ge.sinceYmd).toBe(localYmdDaysAgo(7))
      expect(w.emitted('request-reload')).toHaveLength(1)
    })

    it('30d sets a 30-day since lower bound', async () => {
      await clickPreset('30d')
      expect(useGraphExplorerStore().sinceYmd).toBe(localYmdDaysAgo(30))
    })

    it('90d sets a 90-day since lower bound', async () => {
      await clickPreset('90d')
      expect(useGraphExplorerStore().sinceYmd).toBe(localYmdDaysAgo(90))
    })

    it('All clears the since lower bound (all time) and emits request-reload', async () => {
      const ge = useGraphExplorerStore()
      ge.setPresetDays(30)
      const w = mountControls()
      const all = w.findAll('button').find((b) => b.text() === 'All')!
      await all.trigger('click')
      expect(ge.sinceYmd).toBe('')
      expect(w.emitted('request-reload')).toHaveLength(1)
    })

    it('the active preset gets a ring-2 highlight', async () => {
      const ge = useGraphExplorerStore()
      ge.setPresetDays(7)
      const w = mountControls()
      await w.vm.$nextTick()
      const btn7 = w.findAll('button').find((b) => b.text() === '7d')!
      expect(btn7.classes()).toContain('ring-2')
    })
  })

  describe('Since date input', () => {
    it('applies a valid YYYY-MM-DD value to the store and emits request-reload', async () => {
      const w = mountControls()
      const input = w.get('[data-testid="graph-status-since-input"]')
      // setValue writes the v-model and dispatches the change event the
      // template binds ``applySinceInput`` to.
      await input.setValue('2021-06-01')
      expect(useGraphExplorerStore().sinceYmd).toBe('2021-06-01')
      expect(w.emitted('request-reload')).toHaveLength(1)
    })

    it('ignores a malformed date (no store write, no emit)', async () => {
      const w = mountControls()
      const input = w.get('[data-testid="graph-status-since-input"]')
      await input.setValue('not-a-date')
      await input.trigger('change')
      expect(useGraphExplorerStore().sinceYmd).toBe('')
      expect(w.emitted('request-reload')).toBeUndefined()
    })
  })

  describe('graph-full Reset button', () => {
    it('is hidden until a cross-episode expansion is recorded', async () => {
      const w = mountControls()
      expect(w.find('[data-testid="graph-status-reset"]').exists()).toBe(false)

      const expansion = useGraphExpansionStore()
      expansion.recordExpand('seed-1', ['a.gi.json'])
      await w.vm.$nextTick()
      expect(w.find('[data-testid="graph-status-reset"]').exists()).toBe(true)
    })

    it('emits request-graph-full-reset on click when not loading', async () => {
      const expansion = useGraphExpansionStore()
      expansion.recordExpand('seed-1', ['a.gi.json'])
      const w = mountControls()
      await w.get('[data-testid="graph-status-reset"]').trigger('click')
      expect(w.emitted('request-graph-full-reset')).toHaveLength(1)
    })

    it('does not emit while artifacts are loading', async () => {
      const expansion = useGraphExpansionStore()
      expansion.recordExpand('seed-1', ['a.gi.json'])
      const artifacts = useArtifactsStore()
      artifacts.loading = true
      const w = mountControls()
      await w.get('[data-testid="graph-status-reset"]').trigger('click')
      expect(w.emitted('request-graph-full-reset')).toBeUndefined()
    })
  })
})
