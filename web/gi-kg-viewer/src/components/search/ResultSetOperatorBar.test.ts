// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import ResultSetOperatorBar from './ResultSetOperatorBar.vue'
import type { SearchHit } from '../../api/searchApi'

/**
 * ResultSetOperatorBar is the Search v3 §S4a operator surface. This spec
 * covers the S4a client-only scope: bar shell + Timeline + On-graph.
 * Cluster + Consensus are S4b — asserted disabled here.
 */
function makeHit(over: Partial<SearchHit> & { metadata?: Record<string, unknown> } = {}): SearchHit {
  return {
    doc_id: 'hit-x',
    score: 0.5,
    text: 'body',
    metadata: {
      doc_type: 'insight',
      source_id: 'insight:x',
      episode_id: 'ep-x',
      publish_date: '2026-04-15',
      ...(over.metadata ?? {}),
    },
    ...over,
  } as unknown as SearchHit
}

function mountBar(hits: SearchHit[] = []) {
  return mount(ResultSetOperatorBar, {
    props: { visibleHits: hits },
    attachTo: document.body,
    // Stub Chart.js — the Timeline panel wraps SubjectTimelineChart which
    // pulls chart.js at mount time.
    global: {
      stubs: {
        SubjectTimelineChart: {
          name: 'SubjectTimelineChart',
          props: ['timeline', 'variant', 'valueLabel', 'emptyText', 'ariaLabel'],
          template:
            '<div data-stub="timeline-chart" :data-total="timeline.total" :data-months="timeline.months.length" :data-undated="timeline.undated"></div>',
        },
      },
    },
  })
}

describe('ResultSetOperatorBar (Search v3 §S4a)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the 4 operator chips, S4b chips (Cluster / Consensus) disabled', () => {
    const w = mountBar([makeHit()])
    expect(w.find('[data-testid="operator-chip-cluster"]').exists()).toBe(true)
    expect(w.find('[data-testid="operator-chip-timeline"]').exists()).toBe(true)
    expect(w.find('[data-testid="operator-chip-graph"]').exists()).toBe(true)
    expect(w.find('[data-testid="operator-chip-consensus"]').exists()).toBe(true)
    // S4b operators disabled until server aggregation lands.
    expect(
      (w.get('[data-testid="operator-chip-cluster"]').element as HTMLButtonElement).disabled,
    ).toBe(true)
    expect(
      (w.get('[data-testid="operator-chip-consensus"]').element as HTMLButtonElement).disabled,
    ).toBe(true)
  })

  it('Timeline panel is NOT rendered until the chip is toggled active', async () => {
    const w = mountBar([makeHit()])
    expect(w.find('[data-testid="operator-timeline-panel"]').exists()).toBe(false)
    await w.get('[data-testid="operator-chip-timeline"]').trigger('click')
    expect(w.find('[data-testid="operator-timeline-panel"]').exists()).toBe(true)
    // aria-pressed reflects the toggle.
    expect(
      w.get('[data-testid="operator-chip-timeline"]').attributes('aria-pressed'),
    ).toBe('true')
  })

  it('Timeline panel toggles OFF on a second click', async () => {
    const w = mountBar([makeHit()])
    await w.get('[data-testid="operator-chip-timeline"]').trigger('click')
    expect(w.find('[data-testid="operator-timeline-panel"]').exists()).toBe(true)
    await w.get('[data-testid="operator-chip-timeline"]').trigger('click')
    expect(w.find('[data-testid="operator-timeline-panel"]').exists()).toBe(false)
  })

  it('Timeline buckets hits by YYYY-MM and counts undated separately', async () => {
    const hits = [
      makeHit({ metadata: { publish_date: '2026-04-15', episode_id: 'e1' } }),
      makeHit({ metadata: { publish_date: '2026-04-30', episode_id: 'e2' } }),
      makeHit({ metadata: { publish_date: '2026-05-01', episode_id: 'e3' } }),
      makeHit({ metadata: { publish_date: '', episode_id: 'e4' } }),
      makeHit({ metadata: { publish_date: 'not-a-date', episode_id: 'e5' } }),
    ]
    const w = mountBar(hits)
    await w.get('[data-testid="operator-chip-timeline"]').trigger('click')
    const stub = w.get('[data-stub="timeline-chart"]')
    // 2 buckets (2026-04 count=2; 2026-05 count=1); undated=2.
    expect(stub.attributes('data-months')).toBe('2')
    expect(stub.attributes('data-total')).toBe('3')
    expect(stub.attributes('data-undated')).toBe('2')
    // Undated notice visible when > 0.
    const undated = w.get('[data-testid="operator-timeline-undated"]')
    expect(undated.text()).toContain('2')
  })

  it('On-graph chip is DISABLED when no hit resolves to a graph id', () => {
    // Hits with neither source_id (topic/entity) nor episode_id → no ids.
    const hits = [
      makeHit({
        metadata: { doc_type: 'aux_note', source_id: '', episode_id: '', publish_date: '' },
      }),
    ]
    const w = mountBar(hits)
    const chip = w.get('[data-testid="operator-chip-graph"]')
    expect((chip.element as HTMLButtonElement).disabled).toBe(true)
    expect(chip.text()).toContain('no ids')
  })

  it('On-graph emits focus-set with de-duped graph ids and switches active to graph', async () => {
    const hits = [
      makeHit({ metadata: { doc_type: 'kg_topic', source_id: 'topic:a', episode_id: 'e1' } }),
      makeHit({
        metadata: { doc_type: 'kg_entity', source_id: 'person:b', episode_id: 'e2' },
      }),
      // Insight hits fall through to episode_id.
      makeHit({ metadata: { doc_type: 'insight', source_id: 'insight:x', episode_id: 'e3' } }),
      // Duplicate should collapse.
      makeHit({ metadata: { doc_type: 'insight', source_id: 'insight:y', episode_id: 'e3' } }),
    ]
    const w = mountBar(hits)
    const chip = w.get('[data-testid="operator-chip-graph"]')
    expect((chip.element as HTMLButtonElement).disabled).toBe(false)
    expect(chip.text()).toContain('On graph (3)')
    await chip.trigger('click')
    const emitted = w.emitted('focus-set')
    expect(emitted).toHaveLength(1)
    // Topic + Person source_ids; Insight resolves via episode_id.
    expect(emitted![0][0]).toEqual(['topic:a', 'person:b', 'e3'])
  })

  it('the active v-model reflects timeline / graph selection', async () => {
    const w = mountBar([makeHit()])
    // v-model is a defineModel; the emit key is 'update:active'.
    await w.get('[data-testid="operator-chip-timeline"]').trigger('click')
    const events = w.emitted('update:active')
    expect(events).toBeDefined()
    expect(events![events!.length - 1]).toEqual(['timeline'])
    await w.get('[data-testid="operator-chip-graph"]').trigger('click')
    const events2 = w.emitted('update:active')
    expect(events2![events2!.length - 1]).toEqual(['graph'])
  })
})
