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

  it('renders all 4 operator chips (Cluster + Consensus now enabled, S4b landed)', () => {
    const w = mountBar([makeHit()])
    expect(w.find('[data-testid="operator-chip-cluster"]').exists()).toBe(true)
    expect(w.find('[data-testid="operator-chip-timeline"]').exists()).toBe(true)
    expect(w.find('[data-testid="operator-chip-graph"]').exists()).toBe(true)
    expect(w.find('[data-testid="operator-chip-consensus"]').exists()).toBe(true)
    // S4b landed — Cluster + Consensus enabled (they emit run-* to the parent).
    expect(
      (w.get('[data-testid="operator-chip-cluster"]').element as HTMLButtonElement).disabled,
    ).toBe(false)
    expect(
      (w.get('[data-testid="operator-chip-consensus"]').element as HTMLButtonElement).disabled,
    ).toBe(false)
  })

  it('Cluster chip: click emits run-cluster and sets active=cluster; second click toggles OFF', async () => {
    const w = mountBar([makeHit()])
    await w.get('[data-testid="operator-chip-cluster"]').trigger('click')
    expect(w.emitted('run-cluster')).toHaveLength(1)
    const events = w.emitted('update:active')
    expect(events![events!.length - 1]).toEqual(['cluster'])
    // Toggle off — no second run-cluster (only opens the panel first time).
    await w.get('[data-testid="operator-chip-cluster"]').trigger('click')
    expect(w.emitted('run-cluster')).toHaveLength(1)
    const events2 = w.emitted('update:active')
    expect(events2![events2!.length - 1]).toEqual([null])
  })

  it('Consensus chip: click emits run-consensus and sets active=consensus', async () => {
    const w = mountBar([makeHit()])
    await w.get('[data-testid="operator-chip-consensus"]').trigger('click')
    expect(w.emitted('run-consensus')).toHaveLength(1)
    const events = w.emitted('update:active')
    expect(events![events!.length - 1]).toEqual(['consensus'])
  })

  it('Cluster panel: empty state when clusters=null after activation', async () => {
    const w = mountBar([makeHit()])
    await w.get('[data-testid="operator-chip-cluster"]').trigger('click')
    // No clusters prop passed → empty-state shown.
    expect(w.find('[data-testid="operator-cluster-panel"]').exists()).toBe(true)
    expect(w.find('[data-testid="operator-cluster-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="operator-cluster-list"]').exists()).toBe(false)
  })

  it('Cluster panel: renders group rows with label + size when clusters passed', async () => {
    const clusters = [
      {
        cluster_id: 'tc:env',
        cluster_kind: 'topic_cluster',
        label: 'Environment',
        size: 3,
        hit_indices: [0, 1, 2],
      },
      {
        cluster_id: null,
        cluster_kind: 'ungrouped',
        label: 'Ungrouped',
        size: 1,
        hit_indices: [3],
      },
    ]
    const w = mount(ResultSetOperatorBar, {
      props: { visibleHits: [makeHit()], clusters },
      attachTo: document.body,
      global: {
        stubs: {
          SubjectTimelineChart: { template: '<div />' },
        },
      },
    })
    await w.get('[data-testid="operator-chip-cluster"]').trigger('click')
    const rows = w.findAll('[data-testid="operator-cluster-list"] li')
    expect(rows).toHaveLength(2)
    expect(rows[0].text()).toContain('Environment')
    expect(rows[0].text()).toContain('3 hits')
    expect(rows[1].text()).toContain('Other')
    expect(rows[1].text()).toContain('1 hit')
    w.unmount()
  })

  it('Consensus panel: renders pair rows with speaker labels + contradiction score', async () => {
    const consensusPairs = [
      {
        topic_id: 'topic:climate',
        topic_label: 'Climate',
        person_a_id: 'person:alice',
        person_a_label: 'Alice',
        person_b_id: 'person:bob',
        person_b_label: 'Bob',
        insight_a_id: 'i:a',
        insight_b_id: 'i:b',
        insight_a_text: 'Alice says X',
        insight_b_text: 'Bob agrees Y',
        contradiction_score: 0.08,
        cosine_similarity: 0.87,
      },
    ]
    const w = mount(ResultSetOperatorBar, {
      props: { visibleHits: [makeHit()], consensusPairs },
      attachTo: document.body,
      global: {
        stubs: {
          SubjectTimelineChart: { template: '<div />' },
        },
      },
    })
    await w.get('[data-testid="operator-chip-consensus"]').trigger('click')
    const list = w.get('[data-testid="operator-consensus-list"]')
    expect(list.text()).toContain('Climate')
    expect(list.text()).toContain('Alice')
    expect(list.text()).toContain('Alice says X')
    expect(list.text()).toContain('Bob agrees Y')
    expect(list.text()).toContain('0.08')
    expect(list.text()).toContain('0.87')
    w.unmount()
  })

  it('operatorError string renders in the bar shell', () => {
    const w = mount(ResultSetOperatorBar, {
      props: {
        visibleHits: [makeHit()],
        operatorError: 'no_index: rebuild the index',
      },
      attachTo: document.body,
      global: { stubs: { SubjectTimelineChart: { template: '<div />' } } },
    })
    expect(w.get('[data-testid="operator-error"]').text()).toContain('no_index')
    w.unmount()
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
