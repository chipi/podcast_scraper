// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import SearchTypeChip from './SearchTypeChip.vue'
import { useSearchStore } from '../../../stores/search'
import { useShellStore } from '../../../stores/shell'

describe('SearchTypeChip (UXS-019)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('is absent when the corpus holds no news', () => {
    const w = mount(SearchTypeChip)
    expect(w.find('[data-testid="search-chip-type"]').exists()).toBe(false)
  })

  it('labels the default and the chosen type, and re-runs the query', async () => {
    const shell = useShellStore()
    shell.contentTypes = ['podcast_episode', 'news_article']
    shell.corpusPath = '/c'
    const search = useSearchStore()
    search.query = 'risk'
    const run = vi.spyOn(search, 'runSearch').mockResolvedValue()
    const w = mount(SearchTypeChip)
    expect(w.get('[data-testid="search-chip-type"]').text()).toBe('Type ▾')
    await w.get('[data-testid="search-type-option-news"]').trigger('click')
    await flushPromises()
    expect(search.filters.contentTypes).toEqual(['news_article'])
    expect(w.get('[data-testid="search-chip-type"]').text()).toBe('Type: News ▾')
    expect(run).toHaveBeenCalledWith('/c')
    await w.get('[data-testid="search-type-option-both"]').trigger('click')
    expect(search.filters.contentTypes).toEqual(['podcast_episode', 'news_article'])
    expect(w.get('[data-testid="search-chip-type"]').text()).toBe('Type: Podcasts and news ▾')
  })

  it('sends content_types only when news is asked', () => {
    // The chip shows only on a corpus that holds news; on one without, no news filter is sent.
    useShellStore().contentTypes = ['podcast_episode', 'news_article']
    const search = useSearchStore()
    expect(search.requestOptions('/c').contentTypes).toBeUndefined()
    search.filters.contentTypes = ['news_article']
    expect(search.requestOptions('/c').contentTypes).toEqual(['news_article'])
  })
})
