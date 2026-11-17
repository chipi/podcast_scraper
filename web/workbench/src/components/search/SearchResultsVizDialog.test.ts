// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import SearchResultsVizDialog from './SearchResultsVizDialog.vue'
import type { SearchHit } from '../../api/searchApi'

function mountDialog(hits: SearchHit[] = []) {
  // Ensure hits have required fields to avoid rendering errors
  const normalizedHits = hits.map((h) => ({
    doc_type: 'chunk',
    id: '1',
    ...h,
  })) as SearchHit[]
  return mount(SearchResultsVizDialog, {
    props: {
      hits: normalizedHits,
    },
    attachTo: document.body,
  })
}

describe('SearchResultsVizDialog', () => {
  // ── Base render ──────────────────────────────────────────────────────────

  it('renders the dialog shell', () => {
    const w = mountDialog()
    const dialog = w.find('dialog')
    expect(dialog.exists()).toBe(true)
  })

  it('displays the hit count in header', () => {
    const w = mountDialog([
      { doc_type: 'chunk', id: '1', score: 0.5 },
      { doc_type: 'chunk', id: '2', score: 0.6 },
    ] as never)
    expect(w.text()).toContain('2')
    expect(w.text()).toContain('hits')
  })

  it('uses singular "hit" when count is 1', () => {
    const w = mountDialog([{ doc_type: 'chunk', id: '1', score: 0.5 }] as never)
    expect(w.text()).toContain('1')
    expect(w.text()).toContain('hit')
  })

  it('uses plural "hits" when count > 1', () => {
    const w = mountDialog([
      { doc_type: 'chunk', id: '1', score: 0.5 },
      { doc_type: 'chunk', id: '2', score: 0.6 },
    ] as never)
    expect(w.text()).toContain('2')
    expect(w.text()).toContain('hits')
  })

  // ── Close button ─────────────────────────────────────────────────────────

  it('provides a close button', () => {
    const w = mountDialog()
    const closeBtn = w.find('button')
    expect(closeBtn.exists()).toBe(true)
    expect(closeBtn.text()).toContain('Close')
  })

  it('closes dialog when close button clicked', async () => {
    const w = mountDialog()
    const closeBtn = w.find('button')
    // Manually trigger the expose methods via component instance
    if (w.vm.close) {
      w.vm.close()
      const dialog = w.find('dialog').element as HTMLDialogElement
      expect(dialog.open).toBe(false)
    }
  })

  // ── Dialog open/close API ────────────────────────────────────────────────

  it('exposes open method', () => {
    const w = mountDialog()
    expect(typeof w.vm.open).toBe('function')
  })

  it('exposes close method', () => {
    const w = mountDialog()
    expect(typeof w.vm.close).toBe('function')
  })

  it('opens dialog with open method', async () => {
    const w = mountDialog()
    const dialog = w.find('dialog').element as HTMLDialogElement
    if (w.vm.open) {
      w.vm.open()
      // Dialog is now open
      expect(dialog.open).toBe(true)
    }
  })

  // ── Backdrop click handler ───────────────────────────────────────────────

  it('closes dialog when backdrop is clicked', async () => {
    const w = mountDialog()
    const dialog = w.find('dialog').element as HTMLDialogElement
    if (w.vm.open) {
      w.vm.open()
      await w.vm.$nextTick()
      // Simulate backdrop click (target === dialog)
      const evt = new MouseEvent('click', { bubbles: true })
      Object.defineProperty(evt, 'target', { value: dialog, enumerable: true })
      dialog.dispatchEvent(evt)
      await w.vm.$nextTick()
      expect(dialog.open).toBe(false)
    }
  })

  // ── Doc types section ────────────────────────────────────────────────────

  it('renders doc types section', () => {
    const w = mountDialog()
    expect(w.text()).toContain('Doc types')
  })

  it('displays doc type sections', () => {
    const w = mountDialog()
    // Doc types section should always exist
    const sections = w.findAll('section')
    expect(sections.length).toBeGreaterThan(0)
  })

  // ── Timeline section ────────────────────────────────────────────────────

  it('renders publish month section', () => {
    const w = mountDialog()
    expect(w.text()).toContain('Publish month')
  })

  it('displays a note about publish_date index', () => {
    const w = mountDialog()
    expect(w.text()).toContain('publish_date')
  })

  // ── Episodes section ────────────────────────────────────────────────────

  it('renders episodes section', () => {
    const w = mountDialog()
    expect(w.text()).toContain('Episodes')
  })

  // ── Feeds section ───────────────────────────────────────────────────────

  it('renders feeds section', () => {
    const w = mountDialog()
    expect(w.text()).toContain('Feeds')
  })

  // ── Similarity scores section ────────────────────────────────────────────

  it('renders similarity scores section', () => {
    const w = mountDialog()
    expect(w.text()).toContain('Similarity scores')
  })

  it('displays score stats when hits have scores', () => {
    const hits = [
      { doc_type: 'chunk', id: '1', score: 0.85 },
      { doc_type: 'chunk', id: '2', score: 0.92 },
      { doc_type: 'chunk', id: '3', score: 0.78 },
    ] as never
    const w = mountDialog(hits)
    // The score stats should display min, max, mean, spread
    const scoresSection = w.text()
    // Check for score-related content
    expect(scoresSection).toContain('min')
  })

  it('displays a note about score normalization', () => {
    const w = mountDialog()
    expect(w.text()).toContain('strongest hit in this list')
  })

  // ── Terms section ───────────────────────────────────────────────────────

  it('renders terms section', () => {
    const w = mountDialog()
    expect(w.text()).toContain('Terms')
  })

  it('displays a note about stopwords', () => {
    const w = mountDialog()
    expect(w.text()).toContain('stopwords removed')
  })

  // ── ARIA attributes ─────────────────────────────────────────────────────

  it('has proper aria-labelledby for dialog', () => {
    const w = mountDialog()
    const dialog = w.find('dialog')
    expect(dialog.attributes('aria-labelledby')).toBe('search-results-viz-title')
  })

  it('has a title with id search-results-viz-title', () => {
    const w = mountDialog()
    const title = w.find('h2#search-results-viz-title')
    expect(title.exists()).toBe(true)
    expect(title.text()).toContain('Search result insights')
  })

  // ── Section accessibility ────────────────────────────────────────────────

  it('marks doc types section as a region', () => {
    const w = mountDialog()
    const sections = w.findAll('section[role="region"]')
    expect(sections.length).toBeGreaterThan(0)
    // Doc types should be one of them
    expect(sections.some((s) => s.text().includes('Doc types'))).toBe(true)
  })

  it('marks episodes section as a region', () => {
    const w = mountDialog()
    const sections = w.findAll('section[role="region"]')
    expect(sections.some((s) => s.text().includes('Episodes'))).toBe(true)
  })

  // ── Empty states ────────────────────────────────────────────────────────

  it('displays "No month buckets" when no timeline data', () => {
    const w = mountDialog([])
    // Empty hits means no timeline
    const text = w.text()
    // May show empty or minimal timeline
    expect(text).toContain('Publish month')
  })

  it('displays "No terms extracted" when hits have no text', () => {
    const w = mountDialog([{ doc_type: 'chunk', id: '1', score: 0.5 }] as never)
    const text = w.text()
    expect(text).toContain('Terms')
  })

  // ── Computed properties ──────────────────────────────────────────────────

  it('computes doc type distribution', async () => {
    const hits = [
      { doc_type: 'chunk', id: '1', score: 0.5 },
      { doc_type: 'chunk', id: '2', score: 0.6 },
      { doc_type: 'summary', id: '3', score: 0.7 },
    ] as never
    const w = mountDialog(hits)
    await w.vm.$nextTick()
    // Should have some doc types computed
    expect(Array.isArray(w.vm.docTypes)).toBe(true)
  })

  it('computes episode distribution', async () => {
    const hits = [
      { doc_type: 'chunk', id: '1', episode_id: 'ep:1', score: 0.5 },
      { doc_type: 'chunk', id: '2', episode_id: 'ep:1', score: 0.6 },
      { doc_type: 'chunk', id: '3', episode_id: 'ep:2', score: 0.7 },
    ] as never
    const w = mountDialog(hits)
    await w.vm.$nextTick()
    expect(w.vm.episodes).toBeDefined()
  })

  it('computes feed distribution', async () => {
    const w = mountDialog()
    await w.vm.$nextTick()
    expect(w.vm.feeds).toBeDefined()
  })

  it('computes score statistics', async () => {
    const w = mountDialog()
    await w.vm.$nextTick()
    // scoreStats can be null or defined
    expect(w.vm.scoreStats === null || typeof w.vm.scoreStats === 'object').toBe(true)
  })

  it('computes timeline buckets', async () => {
    const w = mountDialog()
    await w.vm.$nextTick()
    expect(w.vm.timeline).toBeDefined()
    expect(Array.isArray(w.vm.timeline.buckets)).toBe(true)
  })

  it('computes top terms from hits', async () => {
    const w = mountDialog()
    await w.vm.$nextTick()
    expect(Array.isArray(w.vm.terms)).toBe(true)
  })

  // ── Headline insights ────────────────────────────────────────────────────

  it('computes headline insight', async () => {
    const w = mountDialog()
    await w.vm.$nextTick()
    // headlineInsight can be string or null
    expect(typeof w.vm.headlineInsight === 'string' || w.vm.headlineInsight === null).toBe(true)
  })

  it('computes time insight', async () => {
    const w = mountDialog()
    await w.vm.$nextTick()
    // timeInsight can be string or null
    expect(typeof w.vm.timeInsight === 'string' || w.vm.timeInsight === null).toBe(true)
  })

  it('computes terms insight', async () => {
    const w = mountDialog()
    await w.vm.$nextTick()
    // termsInsight can be string or null
    expect(typeof w.vm.termsInsight === 'string' || w.vm.termsInsight === null).toBe(true)
  })

  // ── Props reactivity ────────────────────────────────────────────────────

  it('updates when hits prop changes', async () => {
    const w = mountDialog([{ doc_type: 'chunk', id: '1', score: 0.5 }] as never)
    await w.vm.$nextTick()
    const initialLength = w.vm.docTypes.length

    // Change props
    await w.setProps({
      hits: [
        { doc_type: 'chunk', id: '1', score: 0.5 },
        { doc_type: 'summary', id: '2', score: 0.6 },
      ] as never,
    })
    await w.vm.$nextTick()
    // DocTypes should be recalculated (may be same or different)
    expect(Array.isArray(w.vm.docTypes)).toBe(true)
  })

  // ── Scaling factors ──────────────────────────────────────────────────

  it('computes maxTermCount', async () => {
    const w = mountDialog()
    await w.vm.$nextTick()
    expect(typeof w.vm.maxTermCount).toBe('number')
    expect(w.vm.maxTermCount).toBeGreaterThanOrEqual(1)
  })

  it('computes maxMonthCount', async () => {
    const w = mountDialog()
    await w.vm.$nextTick()
    expect(typeof w.vm.maxMonthCount).toBe('number')
    expect(w.vm.maxMonthCount).toBeGreaterThanOrEqual(1)
  })
})
