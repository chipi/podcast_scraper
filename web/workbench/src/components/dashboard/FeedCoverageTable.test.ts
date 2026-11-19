// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { CoverageFeedItem } from '../../api/corpusCoverageApi'
import { feedCoverageColumns } from '../../extensions'
import FeedCoverageTable from './FeedCoverageTable.vue'

// Server order (by another layer) puts Alpha first; the table orders by KG share: Beta (0%) first.
const rows: CoverageFeedItem[] = [
  { feed_id: 'f1', display_title: 'Alpha Show', total: 4, with_gi: 1, with_kg: 3 },
  { feed_id: 'f2', display_title: 'Beta Show', total: 3, with_gi: 3, with_kg: 0 },
]

describe('FeedCoverageTable', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })
  afterEach(() => feedCoverageColumns.reset())

  it('shows the empty state when there are no feeds', () => {
    const w = mount(FeedCoverageTable, { props: { rows: [], feedsIndexed: [] } })
    expect(w.text()).toContain('No feeds in catalog.')
    expect(w.find('table').exists()).toBe(false)
    expect(w.text()).not.toContain('lowest KG coverage')
  })

  it('orders feeds by lowest KG share, with KG percentages and no GI column of its own', () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [] } })
    expect(w.get('h3').text()).toBe('Feed coverage (lowest KG first)')
    const heads = w.findAll('th').map((t) => t.text())
    expect(heads).toEqual(['Feed', 'Episodes', 'KG', 'In index'])
    const trs = w.findAll('[data-testid="feed-coverage-row"]')
    expect(trs).toHaveLength(2)
    expect(trs[0].text()).toContain('Beta Show')
    expect(trs[0].text()).toContain('0%')
    expect(trs[1].text()).toContain('Alpha Show')
    expect(trs[1].text()).toContain('75%')
    expect(w.find('.bg-gi').exists()).toBe(false)
  })

  it('breaks a KG-share tie by feed id', () => {
    const tie: CoverageFeedItem[] = [
      { feed_id: 'b', display_title: 'B', total: 2, with_gi: 0, with_kg: 1 },
      { feed_id: 'a', display_title: 'A', total: 4, with_gi: 0, with_kg: 2 },
    ]
    const w = mount(FeedCoverageTable, { props: { rows: tie, feedsIndexed: [] } })
    expect(w.findAll('[data-testid="feed-coverage-row"]').map((r) => r.find('td').text())).toEqual(['A', 'B'])
  })

  it('sizes the KG bar from the percentage', () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [] } })
    expect(w.findAll('.bg-kg').map((b) => b.attributes('style'))).toEqual(['width: 0%;', 'width: 75%;'])
  })

  it('reports 0% (not NaN) for a feed with no episodes', () => {
    const w = mount(FeedCoverageTable, {
      props: {
        rows: [{ feed_id: 'z', display_title: 'Zero', total: 0, with_gi: 0, with_kg: 0 }],
        feedsIndexed: [],
      },
    })
    expect(w.text()).not.toContain('NaN')
    expect(w.text()).toContain('0%')
    expect(w.text()).not.toContain('lowest KG coverage')
  })

  it('marks indexed feeds Yes using trimmed ids and ignores blank entries', () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [' f2 ', '', '  '] } })
    const trs = w.findAll('[data-testid="feed-coverage-row"]')
    expect(trs[0].findAll('td').at(-1)!.text()).toBe('Yes')
    expect(trs[1].findAll('td').at(-1)!.text()).toBe('—')
  })

  it('writes an insight about the lowest-KG feed with plural missing count', () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [] } })
    expect(w.text()).toContain('Feed Beta Show has lowest KG coverage at 0% — 3 episodes without KG artifacts.')
  })

  it('uses singular wording when one episode lacks KG, and writes none when every feed is covered', () => {
    const one = mount(FeedCoverageTable, {
      props: { rows: [{ feed_id: 'f', display_title: 'Solo', total: 2, with_gi: 2, with_kg: 1 }], feedsIndexed: [] },
    })
    expect(one.text()).toContain('1 episode without KG artifacts.')
    const full = mount(FeedCoverageTable, {
      props: { rows: [{ feed_id: 'f', display_title: 'Full', total: 4, with_gi: 0, with_kg: 4 }], feedsIndexed: [] },
    })
    expect(full.text()).not.toContain('lowest KG coverage')
  })

  it('emits select-feed with the feed id from every clickable cell', async () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [] } })
    const cells = w.findAll('[data-testid="feed-coverage-row"]')[0].findAll('td')
    expect(cells).toHaveLength(4)
    for (const c of cells) {
      await c.trigger('click')
    }
    expect(w.emitted('select-feed')).toEqual(Array(4).fill(['f2']))
  })

  it('adds a registered column before KG, with its share bar, title and click', async () => {
    feedCoverageColumns.register({ id: 'probe', label: 'PR', noun: 'Probe coverage', barClass: 'bg-probe', count: (r) => r.with_gi })
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [] } })
    expect(w.findAll('th').map((t) => t.text())).toEqual(['Feed', 'Episodes', 'PR', 'KG', 'In index'])
    const cell = w.findAll('[data-testid="feed-coverage-cell-probe"]')[1]
    expect(cell.text()).toBe('25%')
    expect(cell.attributes('title')).toBe('Probe coverage for this feed — click to open Library with this feed (25%)')
    expect(cell.find('.bg-probe').attributes('style')).toContain('width: 25%')
    await cell.trigger('click')
    expect(w.emitted('select-feed')).toEqual([['f1']])
  })
})
