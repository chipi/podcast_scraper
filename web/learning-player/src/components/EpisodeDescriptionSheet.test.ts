import { flushPromises, mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import en from '../i18n/locales/en.json'

const openExternal = vi.fn(async (_url: string) => {})
vi.mock('../services/native', () => ({ openExternal: (url: string) => openExternal(url) }))

import EpisodeDescriptionSheet from './EpisodeDescriptionSheet.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

// jsdom has <dialog> without showModal()/close() — same stub as ConfirmDialog.test.ts.
beforeAll(() => {
  if (!('showModal' in HTMLDialogElement.prototype)) {
    Object.assign(HTMLDialogElement.prototype, {
      showModal(this: HTMLDialogElement) {
        this.open = true
      },
      close(this: HTMLDialogElement) {
        this.open = false
        this.dispatchEvent(new Event('close'))
      },
    })
  }
})
beforeEach(() => openExternal.mockClear())

async function make(description: string, open = true) {
  const w = mount(EpisodeDescriptionSheet, {
    props: { open, title: 'Ep 12 — Rates', description, showTitle: 'Long Horizon Notes' },
    global: { plugins: [i18n] },
    attachTo: document.body,
  })
  await flushPromises()
  return w
}

describe('EpisodeDescriptionSheet', () => {
  it('shows the whole description, and opens as a modal when asked', async () => {
    const long = 'A long publisher description. '.repeat(200).trim()
    const w = await make(long)
    expect((w.get('[data-testid="episode-description"]').element as HTMLDialogElement).open).toBe(true)
    expect(w.get('[data-testid="episode-description-text"]').text()).toBe(long)
    expect(w.get('[data-testid="episode-description-episode"]').text()).toBe('Ep 12 — Rates')
    w.unmount()
  })

  it('makes written-out links tappable and opens them outside the app', async () => {
    const w = await make('Sponsors: https://example.com/offer. Notes www.example.org')
    const links = w.findAll('[data-testid="episode-description-link"]')
    expect(links.map((a) => a.attributes('href'))).toEqual([
      'https://example.com/offer',
      'https://www.example.org',
    ])
    expect(links[0].attributes('rel')).toBe('noopener noreferrer')
    await links[0].trigger('click')
    expect(openExternal).toHaveBeenCalledWith('https://example.com/offer')
    w.unmount()
  })

  it('renders markup in a description as text, never as HTML', async () => {
    const w = await make('<img src=x onerror="alert(1)"> <a href="javascript:alert(1)">tap</a>')
    const body = w.get('[data-testid="episode-description-text"]')
    expect(body.find('img').exists()).toBe(false)
    expect(body.findAll('a')).toHaveLength(0)
    expect(body.text()).toContain('<img src=x onerror="alert(1)">')
    w.unmount()
  })

  it('closes from the ✕ and from Escape (the dialog\'s own close)', async () => {
    const w = await make('Text')
    await w.get('[data-testid="episode-description-close"]').trigger('click')
    expect(w.emitted('close')).toHaveLength(1)
    ;(w.get('[data-testid="episode-description"]').element as HTMLDialogElement).close()
    expect(w.emitted('close')).toHaveLength(2)
    w.unmount()
  })

  it('stays shut until opened', async () => {
    const w = await make('Text', false)
    expect((w.get('[data-testid="episode-description"]').element as HTMLDialogElement).open).toBe(false)
    await w.setProps({ open: true })
    await flushPromises()
    expect((w.get('[data-testid="episode-description"]').element as HTMLDialogElement).open).toBe(true)
    w.unmount()
  })

  it('is laid out like the Brief: named About as its door, the show and episode opening it', async () => {
    const w = await make('Some text.')
    expect(w.get('header').text()).toContain('About')
    expect(w.get('[data-testid="episode-description"]').attributes('aria-label')).toBe('About')
    const body = w.get('[data-testid="episode-description-episode"]').element.parentElement!
    expect(body.textContent).toContain('Long Horizon Notes')
    w.unmount()
  })

  it('pulling the grab handle down closes it, as on the Brief', async () => {
    vi.useFakeTimers()
    try {
      const w = await make('Some text.')
      const handle = w.get('[data-testid="episode-description-handle"]').element
      for (const [type, y] of [['pointerdown', 100], ['pointermove', 320], ['pointerup', 320]] as const) {
        const e = new MouseEvent(type, { clientY: y, bubbles: true })
        Object.defineProperty(e, 'pointerType', { value: 'touch' })
        handle.dispatchEvent(e)
      }
      vi.advanceTimersByTime(400)
      expect(w.emitted('close')).toHaveLength(1)
      w.unmount()
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('EpisodeDescriptionSheet — copyright / licence line (#2325)', () => {
  async function withLicence(description: string, licenceSummary: string | null) {
    const { createMemoryHistory, createRouter } = await import('vue-router')
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: '/', component: { template: '<div/>' } },
        { path: '/podcast/:feedId/licence', name: 'podcast-licence', component: { template: '<div/>' } },
      ],
    })
    const w = mount(EpisodeDescriptionSheet, {
      props: { open: true, title: 'Ep', description, showTitle: 'Show', licenceSummary, feedId: 'p05', slug: 'ep-1' },
      global: { plugins: [i18n, router] },
      attachTo: document.body,
    })
    await flushPromises()
    return w
  }

  it('ends About with the show licence line, linking to its licence page, and closes on tap', async () => {
    const w = await withLicence('An episode.', '© 2020-2021 THE NEW YORK TIMES COMPANY; The New…')
    const link = w.get('[data-testid="episode-description-licence"]')
    expect(link.text()).toBe('© 2020-2021 THE NEW YORK TIMES COMPANY; The New…')
    // The same page as the show's, with this episode attached.
    expect(link.attributes('href')).toBe('/podcast/p05/licence?episode=ep-1')
    await link.trigger('click')
    expect(w.emitted('close')).toBeTruthy()
  })

  it('shows the line alone when the episode has no description', async () => {
    const w = await withLicence('', 'No explicit licence found')
    expect(w.find('[data-testid="episode-description-text"]').exists()).toBe(false)
    expect(w.get('[data-testid="episode-description-licence"]').text()).toBe('No explicit licence found')
  })

  it('has no line when the feed was not checked', async () => {
    const w = await withLicence('An episode.', null)
    expect(w.find('[data-testid="episode-description-licence"]').exists()).toBe(false)
  })
})
