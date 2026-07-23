// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import CompareOperatorPanel from './CompareOperatorPanel.vue'
import type { SearchCompareResponse, SearchHit } from '../../api/searchApi'

/**
 * Search v3 §S8 — CompareOperatorPanel unit specs. Covers subject
 * discovery from hit metadata, picker defaults, disabled states, and
 * the 2-column render + judge summary mute rule.
 */

function makeHit(overrides: Partial<SearchHit> & { metadata?: Record<string, unknown> } = {}): SearchHit {
  return {
    doc_id: overrides.doc_id ?? 'd:x',
    score: overrides.score ?? 0.5,
    text: overrides.text ?? '',
    metadata: overrides.metadata ?? {},
  } as SearchHit
}

function makePack(overrides: Partial<SearchCompareResponse['pack_a']> = {}): SearchCompareResponse['pack_a'] {
  return {
    subject: overrides.subject ?? { kind: 'person', id: 'x', label: null },
    query: overrides.query ?? 'x',
    query_type: overrides.query_type ?? 'semantic',
    rendered: overrides.rendered ?? '',
    token_count: overrides.token_count ?? 0,
    max_tokens: overrides.max_tokens ?? 2000,
    top_insight_id: overrides.top_insight_id ?? null,
    top_insight_text: overrides.top_insight_text ?? '',
    supporting_segment_ids: overrides.supporting_segment_ids ?? [],
    supporting_segment_texts: overrides.supporting_segment_texts ?? [],
    coverage_summary: overrides.coverage_summary ?? { episode_count: 0 },
    confidence_p50: overrides.confidence_p50 ?? 0,
    result_count: overrides.result_count ?? 0,
    grounded: overrides.grounded ?? false,
  }
}

function mountPanel(props: Partial<{
  visibleHits: SearchHit[]
  compareResult: SearchCompareResponse | null
  compareLoading: boolean
  compareError: string | null
}> = {}) {
  return mount(CompareOperatorPanel, {
    attachTo: document.body,
    props: {
      visibleHits: props.visibleHits ?? [],
      compareResult: props.compareResult ?? null,
      compareLoading: props.compareLoading ?? false,
      compareError: props.compareError ?? null,
    },
  })
}

describe('CompareOperatorPanel (Search v3 §S8)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the empty-state notice when fewer than 2 distinct subjects appear in the hits', () => {
    const w = mountPanel({
      visibleHits: [makeHit({ metadata: { speaker_name: 'Alice' } })],
    })
    // 1 candidate → below the threshold. Empty-state notice must render.
    expect(w.get('[data-testid="operator-compare-empty"]').text()).toContain(
      'Fewer than 2 comparable subjects',
    )
    w.unmount()
  })

  it('discovers persons, topics, episodes, feeds and seeds both picker slots', async () => {
    const w = mountPanel({
      visibleHits: [
        makeHit({
          metadata: {
            speaker_name: 'Alice',
            episode_id: 'ep1',
            episode_title: 'Ep One',
            topic_label: 'Compute',
          },
        }),
        makeHit({
          metadata: {
            speaker_name: 'Bob',
            episode_id: 'ep2',
            feed_id: 'sha256:zzz',
          },
        }),
      ],
    })
    const slotA = w.get('[data-testid="operator-compare-slot-a"]')
    const slotB = w.get('[data-testid="operator-compare-slot-b"]')
    // Both defaults are set to the top-2 discovered subjects.
    expect((slotA.element as HTMLSelectElement).value).not.toBe('')
    expect((slotB.element as HTMLSelectElement).value).not.toBe('')
    expect((slotA.element as HTMLSelectElement).value).not.toBe(
      (slotB.element as HTMLSelectElement).value,
    )
    // Run button is enabled by default when both slots hold distinct values.
    const run = w.get('[data-testid="operator-compare-run"]')
    expect((run.element as HTMLButtonElement).disabled).toBe(false)
    w.unmount()
  })

  it('emits run-compare with the resolved SubjectRef payloads on click', async () => {
    const w = mountPanel({
      visibleHits: [
        makeHit({ metadata: { speaker_name: 'Alice' } }),
        makeHit({ metadata: { speaker_name: 'Bob' } }),
      ],
    })
    await w.get('[data-testid="operator-compare-run"]').trigger('click')
    const emitted = w.emitted('run-compare')
    expect(emitted).toBeTruthy()
    const [payload] = emitted![0] as [{ subjectA: { id: string }; subjectB: { id: string } }]
    expect(payload.subjectA.id).toBe('Alice')
    expect(payload.subjectB.id).toBe('Bob')
    w.unmount()
  })

  it('disables Run when both slots resolve to the same subject', async () => {
    const w = mountPanel({
      visibleHits: [
        makeHit({ metadata: { speaker_name: 'Alice' } }),
        makeHit({ metadata: { speaker_name: 'Bob' } }),
      ],
    })
    // Force both slots to slot A's value.
    const slotA = w.get('[data-testid="operator-compare-slot-a"]')
    const slotB = w.get('[data-testid="operator-compare-slot-b"]')
    const chosen = (slotA.element as HTMLSelectElement).value
    await (slotB.element as HTMLSelectElement).dispatchEvent(new Event('change'))
    ;(slotB.element as HTMLSelectElement).value = chosen
    await slotB.trigger('change')
    const run = w.get('[data-testid="operator-compare-run"]')
    expect((run.element as HTMLButtonElement).disabled).toBe(true)
    w.unmount()
  })

  it('renders 2-column packs + judge summary when compareResult is grounded', () => {
    const w = mountPanel({
      visibleHits: [
        makeHit({ metadata: { speaker_name: 'Alice' } }),
        makeHit({ metadata: { speaker_name: 'Bob' } }),
      ],
      compareResult: {
        pack_a: makePack({
          subject: { kind: 'person', id: 'person:alice', label: 'Alice' },
          top_insight_text: 'A insight',
          grounded: true,
          confidence_p50: 0.9,
          result_count: 3,
        }),
        pack_b: makePack({
          subject: { kind: 'person', id: 'person:bob', label: 'Bob' },
          top_insight_text: 'B insight',
          grounded: true,
          confidence_p50: 0.5,
          result_count: 2,
        }),
        judge_summary: 'Alice shows higher confidence',
      },
    })
    expect(w.find('[data-testid="operator-compare-columns"]').exists()).toBe(true)
    expect(w.get('[data-testid="operator-compare-pack-a"]').text()).toContain('Alice')
    expect(w.get('[data-testid="operator-compare-pack-a"]').text()).toContain('A insight')
    expect(w.get('[data-testid="operator-compare-pack-b"]').text()).toContain('B insight')
    expect(w.get('[data-testid="operator-compare-judge"]').text()).toContain(
      'Alice shows higher confidence',
    )
    // Neither pack has the ungrounded badge when both are grounded.
    expect(w.find('[data-testid="operator-compare-pack-a-ungrounded"]').exists()).toBe(false)
    expect(w.find('[data-testid="operator-compare-pack-b-ungrounded"]').exists()).toBe(false)
    w.unmount()
  })

  it('does not render the judge summary when one side is ungrounded', () => {
    const w = mountPanel({
      visibleHits: [
        makeHit({ metadata: { speaker_name: 'Alice' } }),
        makeHit({ metadata: { speaker_name: 'Bob' } }),
      ],
      compareResult: {
        pack_a: makePack({
          subject: { kind: 'person', id: 'A', label: 'Alice' },
          grounded: true,
          top_insight_text: 'A insight',
        }),
        pack_b: makePack({
          subject: { kind: 'person', id: 'B', label: 'Bob' },
          grounded: false,
        }),
        judge_summary: null,
      },
    })
    expect(w.find('[data-testid="operator-compare-judge"]').exists()).toBe(false)
    expect(w.get('[data-testid="operator-compare-pack-b-ungrounded"]').exists()).toBe(true)
    w.unmount()
  })

  it('surfaces the compareError state', () => {
    const w = mountPanel({
      visibleHits: [
        makeHit({ metadata: { speaker_name: 'Alice' } }),
        makeHit({ metadata: { speaker_name: 'Bob' } }),
      ],
      compareError: 'boom',
    })
    expect(w.get('[data-testid="operator-compare-error"]').text()).toBe('boom')
    w.unmount()
  })

  it('shows the Clear button only after a compareResult renders and emits clear-compare on click', async () => {
    const w = mountPanel({
      visibleHits: [
        makeHit({ metadata: { speaker_name: 'Alice' } }),
        makeHit({ metadata: { speaker_name: 'Bob' } }),
      ],
    })
    expect(w.find('[data-testid="operator-compare-clear"]').exists()).toBe(false)
    await w.setProps({
      compareResult: {
        pack_a: makePack({
          subject: { kind: 'person', id: 'A', label: 'Alice' },
          grounded: true,
          top_insight_text: 'A',
        }),
        pack_b: makePack({
          subject: { kind: 'person', id: 'B', label: 'Bob' },
          grounded: true,
          top_insight_text: 'B',
        }),
        judge_summary: null,
      },
    })
    await w.get('[data-testid="operator-compare-clear"]').trigger('click')
    expect(w.emitted('clear-compare')).toBeTruthy()
    w.unmount()
  })

  it('flips Run button label to "Comparing…" while compareLoading is true', () => {
    const w = mountPanel({
      visibleHits: [
        makeHit({ metadata: { speaker_name: 'Alice' } }),
        makeHit({ metadata: { speaker_name: 'Bob' } }),
      ],
      compareLoading: true,
    })
    const run = w.get('[data-testid="operator-compare-run"]')
    expect(run.text()).toBe('Comparing…')
    expect((run.element as HTMLButtonElement).disabled).toBe(true)
    w.unmount()
  })
})
