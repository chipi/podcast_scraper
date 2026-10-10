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
    props: { open, title: 'Ep 12 — Rates', description },
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
})
