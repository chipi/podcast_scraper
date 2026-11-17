// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { searchDocTypes } from '../../extensions'
import SearchResultRowIcon from './SearchResultRowIcon.vue'

const ICON = '[data-testid="search-result-row-icon"]'

describe('SearchResultRowIcon', () => {
  beforeEach(() => setActivePinia(createPinia()))
  afterEach(() => searchDocTypes.reset())

  it('uses a registered doc type\'s letter, classes and title', () => {
    searchDocTypes.register({
      id: 'insight',
      label: 'Insights',
      icon: { letter: 'I', classes: 'bg-primary/15 text-primary', title: 'Insight' },
    })
    const w = mount(SearchResultRowIcon, { props: { docType: 'insight', sourceTier: 'insight' } })
    expect(w.get(ICON).attributes('title')).toBe('Insight · Insight')
    expect(w.get(ICON).text()).toBe('I')
    expect(w.get(ICON).find('span').classes()).toContain('text-primary')
  })

  it('an unregistered insight or quote hit gets the generic letter icon', () => {
    const w = mount(SearchResultRowIcon, { props: { docType: 'quote' } })
    expect(w.get(ICON).text()).toBe('Q')
    expect(w.get(ICON).attributes('title')).toBe('quote')
    expect(w.get(ICON).find('span').classes()).toContain('text-muted')
  })

  it('keeps the platform\'s own icons', () => {
    const w = mount(SearchResultRowIcon, { props: { docType: 'kg_topic', sourceTier: 'aux' } })
    expect(w.get(ICON).text()).toBe('T')
    expect(w.get(ICON).attributes('title')).toBe('Reference · Topic')
  })
})
