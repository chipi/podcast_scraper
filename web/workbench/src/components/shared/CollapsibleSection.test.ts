// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import CollapsibleSection from './CollapsibleSection.vue'

const TOGGLE = 'button[aria-expanded]'

const mountSection = (props: Record<string, unknown> = {}, slots: Record<string, string> = {}) =>
  mount(CollapsibleSection, {
    props: { title: 'My Section', ...props },
    slots: { default: '<p data-testid="body">Body content</p>', ...slots },
    attachTo: document.body,
  })

describe('CollapsibleSection', () => {
  it('renders the title text', () => {
    const w = mountSection()
    expect(w.text()).toContain('My Section')
  })

  it('is open by default and shows the slot content', () => {
    const w = mountSection()
    expect(w.get(TOGGLE).attributes('aria-expanded')).toBe('true')
    expect(w.get('[data-testid="body"]').isVisible()).toBe(true)
  })

  it('respects defaultOpen=false (collapsed on mount)', () => {
    const w = mountSection({ defaultOpen: false })
    expect(w.get(TOGGLE).attributes('aria-expanded')).toBe('false')
    // Body uses v-show, so it exists but is hidden.
    expect(w.get('[data-testid="body"]').isVisible()).toBe(false)
  })

  it('collapses when the toggle is clicked from the open state', async () => {
    const w = mountSection()
    await w.get(TOGGLE).trigger('click')
    expect(w.get(TOGGLE).attributes('aria-expanded')).toBe('false')
    expect(w.get('[data-testid="body"]').isVisible()).toBe(false)
  })

  it('expands again on a second toggle click', async () => {
    const w = mountSection({ defaultOpen: false })
    await w.get(TOGGLE).trigger('click')
    expect(w.get(TOGGLE).attributes('aria-expanded')).toBe('true')
    expect(w.get('[data-testid="body"]').isVisible()).toBe(true)
  })

  it('rotates the chevron only when open', async () => {
    const w = mountSection({ defaultOpen: false })
    const svg = w.get('svg')
    expect(svg.classes()).not.toContain('rotate-90')
    await w.get(TOGGLE).trigger('click')
    expect(svg.classes()).toContain('rotate-90')
  })

  it('hides the summary affordance while open', () => {
    const w = mountSection({ summary: 'short preview' })
    // Only the chevron toggle button should be present while open.
    expect(w.findAll('button').length).toBe(1)
  })

  it('shows the summary button only when collapsed AND summary is set', async () => {
    const w = mountSection({ summary: 'short preview', defaultOpen: false })
    const summaryBtn = w
      .findAll('button')
      .find((b) => b.text() === 'short preview')
    expect(summaryBtn).toBeTruthy()
    expect(summaryBtn!.attributes('aria-label')).toBe('Expand section: short preview')
  })

  it('does not show a summary button when collapsed with no summary', () => {
    const w = mountSection({ defaultOpen: false })
    expect(w.findAll('button').length).toBe(1)
  })

  it('clicking the collapsed summary re-opens the section', async () => {
    const w = mountSection({ summary: 'short preview', defaultOpen: false })
    const summaryBtn = w
      .findAll('button')
      .find((b) => b.text() === 'short preview')!
    await summaryBtn.trigger('click')
    expect(w.get(TOGGLE).attributes('aria-expanded')).toBe('true')
    expect(w.get('[data-testid="body"]').isVisible()).toBe(true)
  })

  it('renders the actions slot when provided', () => {
    const w = mountSection({}, { actions: '<button data-testid="action">Do</button>' })
    expect(w.get('[data-testid="action"]').exists()).toBe(true)
  })

  it('does not render the actions wrapper when no actions slot', () => {
    const w = mountSection()
    expect(w.find('[data-testid="action"]').exists()).toBe(false)
  })

  it('stops propagation so action clicks do not toggle the section', async () => {
    const w = mountSection(
      { defaultOpen: true },
      { actions: '<button data-testid="action">Do</button>' },
    )
    await w.get('[data-testid="action"]').trigger('click')
    // Section must remain open — the @click.stop wrapper prevents toggling.
    expect(w.get(TOGGLE).attributes('aria-expanded')).toBe('true')
  })

  it('renders the subtitle slot when provided', () => {
    const w = mountSection({}, { subtitle: '<span data-testid="sub">extra</span>' })
    expect(w.get('[data-testid="sub"]').exists()).toBe(true)
  })

  it('does not render the subtitle region without a subtitle slot', () => {
    const w = mountSection()
    expect(w.find('[data-testid="sub"]').exists()).toBe(false)
  })
})
