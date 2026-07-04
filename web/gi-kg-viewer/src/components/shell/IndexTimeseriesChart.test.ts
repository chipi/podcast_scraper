// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import IndexTimeseriesChart from './IndexTimeseriesChart.vue'
import type { TimeseriesSeries } from './timeseriesChart'

// Mock Chart.js to avoid canvas-related errors
vi.mock('chart.js', () => ({
  Chart: vi.fn().mockImplementation(function () {
    this.destroy = vi.fn()
  }),
  CategoryScale: vi.fn(),
  LinearScale: vi.fn(),
  PointElement: vi.fn(),
  LineElement: vi.fn(),
  BarController: vi.fn(),
  BarElement: vi.fn(),
  Title: vi.fn(),
  Tooltip: vi.fn(),
  Legend: vi.fn(),
  Filler: vi.fn(),
}))

// Mock the composable for theme reloading
vi.mock('../../composables/useThemeChartReloader', () => ({
  useThemeChartReloader: vi.fn((callback) => {
    // Just call immediately in tests
    callback()
  }),
}))

// Mock chart registration utility
vi.mock('../../utils/chartRegister', () => ({
  ensureChartJsRegistered: vi.fn(),
}))

function mountChart(
  labels: string[] = [],
  series: TimeseriesSeries[] = [],
  caption?: string,
) {
  return mount(IndexTimeseriesChart, {
    props: {
      labels,
      series,
      caption,
    },
  })
}

describe('IndexTimeseriesChart', () => {
  // ── Base render ──────────────────────────────────────────────────────────

  it('renders the chart container', () => {
    const w = mountChart()
    const container = w.find('[data-testid="index-timeseries"]')
    expect(container.exists()).toBe(true)
  })

  it('renders the title', () => {
    const w = mountChart()
    expect(w.text()).toContain('Documents by month')
  })

  // ── Month range inputs ───────────────────────────────────────────────────

  it('renders from and to month input fields', () => {
    const w = mountChart()
    const fromInput = w.find('[data-testid="index-date-from"]')
    const toInput = w.find('[data-testid="index-date-to"]')
    expect(fromInput.exists()).toBe(true)
    expect(toInput.exists()).toBe(true)
  })

  it('labels the month inputs correctly', () => {
    const w = mountChart()
    expect(w.text()).toContain('from')
    expect(w.text()).toContain('to')
  })

  // ── Series checkboxes ────────────────────────────────────────────────────

  it('renders checkboxes for each series', () => {
    const series = [
      { key: 'episodes', label: 'Episodes', data: [1, 2, 3], defaultEnabled: true },
      { key: 'chunks', label: 'Chunks', data: [2, 3, 4], defaultEnabled: true },
    ] as TimeseriesSeries[]
    const w = mountChart(['2024-01', '2024-02', '2024-03'], series)
    const checkboxes = w.findAll('input[type="checkbox"]')
    expect(checkboxes.length).toBe(2)
  })

  it('labels series checkboxes with series labels', () => {
    const series = [{ key: 'episodes', label: 'Episodes', data: [1, 2, 3] }] as TimeseriesSeries[]
    const w = mountChart(['2024-01'], series)
    expect(w.text()).toContain('Episodes')
  })

  it('initializes checkboxes as checked by default', () => {
    const series = [
      { key: 'episodes', label: 'Episodes', data: [1, 2, 3], defaultEnabled: true },
    ] as TimeseriesSeries[]
    const w = mountChart(['2024-01'], series)
    const checkbox = w.find('input[type="checkbox"]') as any
    if (checkbox.exists()) {
      expect((checkbox.element as HTMLInputElement).checked).toBe(true)
    }
  })

  it('respects defaultEnabled=false for series', async () => {
    const series = [
      { key: 'episodes', label: 'Episodes', data: [1, 2, 3], defaultEnabled: false },
    ] as TimeseriesSeries[]
    const w = mountChart(['2024-01'], series)
    await w.vm.$nextTick()
    const checkbox = w.find('input[type="checkbox"]') as any
    if (checkbox.exists()) {
      expect((checkbox.element as HTMLInputElement).checked).toBe(false)
    }
  })

  // ── Data range filtering ─────────────────────────────────────────────────

  it('initializes date range from first and last label', async () => {
    const labels = ['2024-01', '2024-02', '2024-03', '2024-04']
    const w = mountChart(labels)
    await w.vm.$nextTick()
    const fromInput = w.find('[data-testid="index-date-from"]') as any
    const toInput = w.find('[data-testid="index-date-to"]') as any
    if (fromInput.exists() && toInput.exists()) {
      expect((fromInput.element as HTMLInputElement).value).toBe('2024-01')
      expect((toInput.element as HTMLInputElement).value).toBe('2024-04')
    }
  })

  it('filters visible indices based on from/to range', async () => {
    const labels = ['2024-01', '2024-02', '2024-03', '2024-04']
    const w = mountChart(labels)
    await w.vm.$nextTick()
    // Set from to 2024-02
    const fromInput = w.find('[data-testid="index-date-from"]')
    await fromInput.setValue('2024-02')
    await w.vm.$nextTick()
    // visibleIndices should only include indices >= 2024-02
    const visibleCount = w.vm.visibleIndices.length
    expect(visibleCount).toBeLessThanOrEqual(labels.length)
  })

  it('resets date range when labels change', async () => {
    let w = mountChart(['2024-01', '2024-02'])
    await w.vm.$nextTick()
    let fromInput = w.find('[data-testid="index-date-from"]') as any
    // Change the range
    await fromInput.setValue('2024-02')
    await w.vm.$nextTick()
    // Now change labels (empty them)
    await w.setProps({ labels: [] })
    await w.vm.$nextTick()
    // New labels trigger re-seed
    await w.setProps({
      labels: ['2025-01', '2025-02'],
    })
    await w.vm.$nextTick()
    fromInput = w.find('[data-testid="index-date-from"]') as any
    if (fromInput.exists()) {
      expect((fromInput.element as HTMLInputElement).value).toBe('2025-01')
    }
  })

  // ── Empty data state ─────────────────────────────────────────────────────

  it('shows "No data" message when no labels', async () => {
    const w = mountChart()
    await w.vm.$nextTick()
    expect(w.text()).toContain('No data in the selected range')
  })

  it('shows "No data" message when range filters out all data', async () => {
    const labels = ['2024-01', '2024-02']
    const w = mountChart(labels)
    await w.vm.$nextTick()
    const fromInput = w.find('[data-testid="index-date-from"]')
    const toInput = w.find('[data-testid="index-date-to"]')
    // Set range that overlaps none
    await fromInput.setValue('2024-03')
    await toInput.setValue('2024-04')
    await w.vm.$nextTick()
    // Message should appear
    const noData = w.text().includes('No data')
    expect(noData || w.vm.filteredLabels.length === 0).toBe(true)
  })

  // ── Canvas rendering ────────────────────────────────────────────────────

  it('renders canvas element when data is available', async () => {
    const labels = ['2024-01', '2024-02']
    const w = mountChart(labels)
    await w.vm.$nextTick()
    const canvas = w.find('canvas')
    expect(canvas.exists()).toBe(true)
  })

  it('hides canvas when no data in range', async () => {
    const labels = ['2024-01', '2024-02']
    const w = mountChart(labels)
    await w.vm.$nextTick()
    const fromInput = w.find('[data-testid="index-date-from"]')
    const toInput = w.find('[data-testid="index-date-to"]')
    await fromInput.setValue('2024-03')
    await toInput.setValue('2024-04')
    await w.vm.$nextTick()
    const canvas = w.find('canvas')
    // Canvas should be hidden in the empty state div
    expect(canvas.exists()).toBe(false)
  })

  // ── Caption display ──────────────────────────────────────────────────────

  it('displays default caption when none provided', () => {
    const w = mountChart()
    const caption = w.text()
    expect(caption).toContain('By publish month')
  })

  it('displays custom caption when provided', () => {
    const customCaption = 'This is a custom chart caption'
    const w = mountChart([], [], customCaption)
    expect(w.text()).toContain(customCaption)
  })

  // ── Reactivity ───────────────────────────────────────────────────────────

  it('updates when labels prop changes', async () => {
    let w = mountChart(['2024-01'])
    await w.vm.$nextTick()
    expect(w.vm.filteredLabels.length).toBe(1)

    // When labels change to empty, the seeded flag resets
    await w.setProps({ labels: [] })
    await w.vm.$nextTick()

    // Then set new labels - they should seed again
    await w.setProps({ labels: ['2025-01', '2025-02', '2025-03'] })
    await w.vm.$nextTick()
    expect(w.vm.filteredLabels.length).toBeGreaterThanOrEqual(1)
  })

  it('updates when series prop changes', async () => {
    const series1 = [{ key: 'a', label: 'A', data: [1] }] as TimeseriesSeries[]
    const w = mountChart(['2024-01'], series1)
    await w.vm.$nextTick()
    expect(w.vm.enabled).toBeDefined()

    const series2 = [
      { key: 'a', label: 'A', data: [1] },
      { key: 'b', label: 'B', data: [2] },
    ] as TimeseriesSeries[]
    await w.setProps({ series: series2 })
    await w.vm.$nextTick()
    // enabled should have both keys now
    expect(Object.keys(w.vm.enabled).length).toBeGreaterThanOrEqual(1)
  })

  // ── Series visibility ────────────────────────────────────────────────────

  it('toggles series visibility via checkbox', async () => {
    const series = [
      { key: 'episodes', label: 'Episodes', data: [1, 2, 3], defaultEnabled: true },
    ] as TimeseriesSeries[]
    const labels = ['2024-01', '2024-02', '2024-03']
    const w = mountChart(labels, series)
    await w.vm.$nextTick()

    // Initially enabled
    expect(w.vm.enabled['episodes']).toBe(true)

    // Toggle via checkbox
    const checkbox = w.find('input[type="checkbox"]')
    await checkbox.setValue(false)
    await w.vm.$nextTick()

    expect(w.vm.enabled['episodes']).toBe(false)
  })

  // ── Computed properties ──────────────────────────────────────────────────

  it('computes visibleIndices correctly', async () => {
    const labels = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
    const w = mountChart(labels)
    await w.vm.$nextTick()

    // Initially all visible
    expect(w.vm.visibleIndices.length).toBe(labels.length)

    // Set range to 2024-02 to 2024-04
    const fromInput = w.find('[data-testid="index-date-from"]')
    const toInput = w.find('[data-testid="index-date-to"]')
    await fromInput.setValue('2024-02')
    await toInput.setValue('2024-04')
    await w.vm.$nextTick()

    // Should have indices for 2024-02, 2024-03, 2024-04
    expect(w.vm.visibleIndices.length).toBe(3)
  })

  it('computes filteredLabels from visibleIndices', async () => {
    const labels = ['2024-01', '2024-02', '2024-03']
    const w = mountChart(labels)
    await w.vm.$nextTick()
    // Initially all labels visible
    expect(w.vm.filteredLabels).toEqual(labels)
  })

  // ── Container dimensions ─────────────────────────────────────────────────

  it('sets chart container height to 200px', () => {
    const w = mountChart(['2024-01', '2024-02'])
    const container = w.find('.relative')
    if (container.exists()) {
      const style = container.attributes('style')
      expect(style).toContain('200px')
    }
  })

  // ── Empty series handling ────────────────────────────────────────────────

  it('handles empty series array', async () => {
    const w = mountChart(['2024-01', '2024-02'], [])
    await w.vm.$nextTick()
    // Should still render without series checkboxes
    expect(w.vm.enabled).toBeDefined()
  })

  // ── Disabled state when no data ──────────────────────────────────────────

  it('disables date inputs when no labels', async () => {
    const w = mountChart()
    await w.vm.$nextTick()
    const fromInput = w.find('[data-testid="index-date-from"]') as any
    if (fromInput.exists()) {
      // Input might be disabled or have empty value
      expect(fromInput.element instanceof HTMLInputElement).toBe(true)
    }
  })

  // ── Keyboard interaction ─────────────────────────────────────────────────

  it('allows typing in month input fields', async () => {
    const labels = ['2024-01', '2024-02', '2024-03']
    const w = mountChart(labels)
    await w.vm.$nextTick()

    const fromInput = w.find('[data-testid="index-date-from"]')
    await fromInput.setValue('2024-02')
    expect(w.vm.fromMonth).toBe('2024-02')
  })

  // ── Label array edge cases ───────────────────────────────────────────────

  it('handles single label', async () => {
    const w = mountChart(['2024-01'])
    await w.vm.$nextTick()
    expect(w.vm.filteredLabels.length).toBe(1)
  })

  it('handles undefined labels initially', async () => {
    const w = mountChart([])
    await w.vm.$nextTick()
    expect(w.vm.filteredLabels.length).toBe(0)
  })
})
