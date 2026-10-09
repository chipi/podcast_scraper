import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { primaryLanguage, resetCorpusLanguagesForTests } from '../composables/useCorpusLanguages'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { Podcast } from '../services/types'
import LanguageBadge from './LanguageBadge.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function shows(...languages: (string | null)[]): Podcast[] {
  return languages.map((language, i) => ({
    feed_id: `f${i}`,
    title: `Show ${i}`,
    artwork_url: null,
    image_url: null,
    description: null,
    episode_count: 1,
    language,
  }))
}

function corpus(...languages: (string | null)[]) {
  return vi.spyOn(api, 'getPodcasts').mockResolvedValue(shows(...languages))
}

async function mountBadge(props: { lang?: string | null; overlay?: boolean }) {
  const wrapper = mount(LanguageBadge, { props, global: { plugins: [i18n] } })
  await flushPromises()
  return wrapper
}

beforeEach(() => resetCorpusLanguagesForTests())
afterEach(() => vi.restoreAllMocks())

describe('LanguageBadge', () => {
  it('shows the code when the corpus holds more than one language (FR7.2)', async () => {
    corpus('en', 'es')
    const badge = (await mountBadge({ lang: 'es' })).get('[data-testid="language-badge"]')
    expect(badge.text()).toBe('es')
    expect(badge.classes()).toContain('uppercase')
    expect(badge.attributes('data-lang')).toBe('es')
  })

  it('renders nothing in a single-language corpus — the constant #2115 removed (FR7.5)', async () => {
    corpus('en', 'en-US', 'EN')
    expect((await mountBadge({ lang: 'en' })).find('[data-testid="language-badge"]').exists()).toBe(
      false,
    )
  })

  it('omits rather than guesses when the item has no language', async () => {
    corpus('en', 'es')
    for (const lang of [null, undefined, '', '  ']) {
      const wrapper = await mountBadge({ lang })
      expect(wrapper.find('[data-testid="language-badge"]').exists()).toBe(false)
    }
  })

  it('counts regional spellings of one language as one language', async () => {
    // en-US and en-GB are both English: a corpus of only those is monolingual, so no badges.
    corpus('en-US', 'en-GB', 'en')
    expect((await mountBadge({ lang: 'en-GB' })).find('[data-testid="language-badge"]').exists()).toBe(
      false,
    )
  })

  it('normalizes the item tag it is given', async () => {
    corpus('en', 'pt-BR')
    const badge = (await mountBadge({ lang: 'pt-BR' })).get('[data-testid="language-badge"]')
    expect(badge.text()).toBe('pt')
  })

  it('names the language for a screen reader instead of spelling the code', async () => {
    corpus('en', 'es')
    const badge = (await mountBadge({ lang: 'es' })).get('[data-testid="language-badge"]')
    expect(badge.attributes('role')).toBe('img')
    expect(badge.attributes('aria-label')).toBe('Language: Spanish')
    expect(badge.attributes('title')).toBe('Spanish')
  })

  it('uses the dark plate over artwork and the muted outline elsewhere', async () => {
    corpus('en', 'de')
    const overlay = (await mountBadge({ lang: 'de', overlay: true })).get('[data-testid="language-badge"]')
    expect(overlay.classes()).toContain('bg-black/55')
    const plain = (await mountBadge({ lang: 'de' })).get('[data-testid="language-badge"]')
    expect(plain.classes()).toContain('text-muted')
    expect(plain.classes()).not.toContain('bg-black/55')
  })

  it('asks the catalogue once for every badge on the page', async () => {
    const spy = corpus('en', 'fr')
    await Promise.all([mountBadge({ lang: 'fr' }), mountBadge({ lang: 'en' }), mountBadge({ lang: 'fr' })])
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('shows nothing while the catalogue is unreachable, and retries on the next mount', async () => {
    const spy = vi.spyOn(api, 'getPodcasts').mockRejectedValueOnce(new Error('offline'))
    expect((await mountBadge({ lang: 'es' })).find('[data-testid="language-badge"]').exists()).toBe(
      false,
    )
    spy.mockResolvedValue(shows('en', 'es'))
    expect((await mountBadge({ lang: 'es' })).find('[data-testid="language-badge"]').exists()).toBe(
      true,
    )
    expect(spy).toHaveBeenCalledTimes(2)
  })
})

describe('primaryLanguage', () => {
  it.each([
    ['en-US', 'en'],
    ['EN', 'en'],
    [' es ', 'es'],
    ['pt_BR', 'pt'],
    ['', null],
    [null, null],
    [undefined, null],
  ])('%s → %s', (tag, expected) => {
    expect(primaryLanguage(tag)).toBe(expected)
  })
})
