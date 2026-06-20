// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import FeedOverrideEditor from './FeedOverrideEditor.vue'

function mountEditor(entry: unknown, globalMaxEpisodes: number | null = null) {
  return mount(FeedOverrideEditor, {
    props: { entry: entry as never, globalMaxEpisodes },
    attachTo: document.body,
  })
}

describe('FeedOverrideEditor', () => {
  it('populates inputs from an entry with overrides + extras', () => {
    const w = mountEditor({
      url: 'https://a.example/rss',
      max_episodes: 5,
      episode_order: 'oldest',
      delay_ms: 250,
    })
    expect(w.get('[data-testid="feed-override-url"]').text()).toBe('https://a.example/rss')
    expect((w.get('[data-testid="feed-override-max-episodes"]').element as HTMLInputElement).value).toBe('5')
    expect((w.get('[data-testid="feed-override-order"]').element as HTMLSelectElement).value).toBe('oldest')
    // Known advanced key now renders as a structured input.
    expect((w.get('[data-testid="feed-override-adv-delay_ms"]').element as HTMLInputElement).value).toBe('250')
  })

  it('emits a structured advanced field with the right type', async () => {
    const w = mountEditor('https://a.example/rss')
    await w.get('[data-testid="feed-override-adv-delay_ms"]').setValue('250')
    await w.get('[data-testid="feed-override-adv-circuit_breaker_enabled"]').setValue('true')
    await w.get('[data-testid="feed-override-save"]').trigger('click')
    expect(w.emitted('save')?.[0]?.[0]).toEqual({
      url: 'https://a.example/rss',
      delay_ms: 250,
      circuit_breaker_enabled: true,
    })
  })

  it('preserves an unknown key via the raw-JSON box', async () => {
    const w = mountEditor({ url: 'https://a.example/rss', future_key: 'keep' })
    expect((w.get('[data-testid="feed-override-extras"]').element as HTMLTextAreaElement).value).toContain('future_key')
    await w.get('[data-testid="feed-override-save"]').trigger('click')
    expect(w.emitted('save')?.[0]?.[0]).toEqual({ url: 'https://a.example/rss', future_key: 'keep' })
  })

  it('emits a structured entry with only the set must-fields', async () => {
    const w = mountEditor('https://a.example/rss')
    await w.get('[data-testid="feed-override-max-episodes"]').setValue('3')
    await w.get('[data-testid="feed-override-order"]').setValue('newest')
    await w.get('[data-testid="feed-override-save"]').trigger('click')
    expect(w.emitted('save')?.[0]?.[0]).toEqual({
      url: 'https://a.example/rss',
      max_episodes: 3,
      episode_order: 'newest',
    })
  })

  it('collapses back to a bare URL string when all overrides are cleared', async () => {
    const w = mountEditor({ url: 'https://a.example/rss', max_episodes: 5 })
    await w.get('[data-testid="feed-override-max-episodes"]').setValue('')
    await w.get('[data-testid="feed-override-save"]').trigger('click')
    expect(w.emitted('save')?.[0]?.[0]).toBe('https://a.example/rss')
  })

  it('rejects invalid max_episodes without emitting save', async () => {
    const w = mountEditor('https://a.example/rss')
    await w.get('[data-testid="feed-override-max-episodes"]').setValue('0')
    await w.get('[data-testid="feed-override-save"]').trigger('click')
    expect(w.find('[data-testid="feed-override-error"]').exists()).toBe(true)
    expect(w.emitted('save')).toBeUndefined()
  })

  it('rejects malformed advanced JSON', async () => {
    const w = mountEditor('https://a.example/rss')
    await w.get('[data-testid="feed-override-extras"]').setValue('{ not json ')
    await w.get('[data-testid="feed-override-save"]').trigger('click')
    expect(w.get('[data-testid="feed-override-error"]').text()).toContain('valid JSON')
    expect(w.emitted('save')).toBeUndefined()
  })

  it('shows the overriding-default hint with the global value', async () => {
    const w = mountEditor('https://a.example/rss', 5)
    await w.get('[data-testid="feed-override-max-episodes"]').setValue('20')
    expect(w.get('[data-testid="feed-override-maxep-hint"]').text()).toContain('= 5')
  })

  it('emits back', async () => {
    const w = mountEditor('https://a.example/rss')
    await w.get('[data-testid="feed-override-back"]').trigger('click')
    expect(w.emitted('back')).toBeTruthy()
  })
})
