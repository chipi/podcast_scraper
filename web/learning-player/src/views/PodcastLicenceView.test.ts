import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeLicence, LicenceDeclaration, PodcastLicence } from '../services/types'
import PodcastLicenceView from './PodcastLicenceView.vue'

// The full text of a show's licence declarations (#2325). Every declaration verbatim, what it
// is, what it applies to, a recognised licence's conditions, conflicts, documents, history.

const FEED = 'feed-1'
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/podcast/:feedId/licence', name: 'podcast-licence', component: PodcastLicenceView },
  ],
})

const NYT =
  '© 2020-2021 THE NEW YORK TIMES COMPANY; The New York Times encourages the use of RSS feeds ' +
  'for personal use in a news reader or as part of a non-commercial blog, subject to your ' +
  'agreement to our Terms of Service.'

function decl(over: Partial<LicenceDeclaration>): LicenceDeclaration {
  return {
    level: 'channel', mechanism: 'rss_copyright', raw: NYT, url: null,
    asset_scope: 'unspecified', known: null, notes: [], ...over,
  }
}

function record(over: Partial<PodcastLicence> = {}): PodcastLicence {
  const declarations = [
    decl({}),
    decl({
      mechanism: 'podcast_license', raw: 'cc-by-nc-4.0', asset_scope: 'audio_video',
      url: 'https://creativecommons.org/licenses/by-nc/4.0/',
      known: {
        id: 'CC-BY-NC-4.0', name: 'CC BY-NC 4.0', url: 'https://creativecommons.org/licenses/by-nc/4.0/',
        commercial_use: false, adaptations: true, attribution: true, share_alike: false,
      },
    }),
  ]
  return {
    feed_id: FEED, title: 'The Daily', summary: 'CC BY-NC 4.0', status: 'declared',
    licence: {
      status: 'declared', declarations,
      conflicts: ["'All rights reserved' alongside a licence: CC-BY-NC-4.0"],
      documents_for_review: ['https://example.com/terms'],
      source_feed_url: 'https://feeds.example/daily', retrieved_at: '2026-10-11T13:30:50+00:00',
    },
    history: [
      { status: 'declared', declarations, first_read: '2026-09-01T00:00:00+00:00', last_read: '2026-10-11T00:00:00+00:00' },
      { status: 'declared', declarations: [decl({ raw: '© 2020 NYT' })], first_read: '2026-01-01T00:00:00+00:00', last_read: '2026-08-01T00:00:00+00:00' },
    ],
    ...over,
  }
}

async function mountWith(body: PodcastLicence | Error, episode: string | null = null) {
  if (body instanceof Error) vi.spyOn(api, 'getPodcastLicence').mockRejectedValue(body)
  else vi.spyOn(api, 'getPodcastLicence').mockResolvedValue(body)
  const w = mount(PodcastLicenceView, {
    props: { feedId: FEED, episode },
    global: { plugins: [i18n, router] },
  })
  await flushPromises()
  return w
}

afterEach(() => vi.restoreAllMocks())

describe('PodcastLicenceView', () => {
  it('shows every declaration verbatim, with what it is and what it applies to', async () => {
    const w = await mountWith(record())
    const raws = w.findAll('[data-testid="podcast-licence-raw"]').map((r) => r.text())
    expect(raws).toEqual([NYT, 'cc-by-nc-4.0'])
    const cards = w.findAll('[data-testid="podcast-licence-declaration"]')
    expect(cards[0].text()).toContain('Copyright notice (RSS <copyright>)')
    expect(cards[0].text()).toContain('Applies to: not specified')
    expect(cards[1].text()).toContain('Applies to: audio and video')
    expect(cards[1].text()).toContain('Recognised as CC BY-NC 4.0')
    expect(cards[1].text()).toContain('Commercial use: no')
    expect(w.text()).toContain('not a licence grant')
  })

  it('lists conflicts, documents to review, the read date and earlier versions', async () => {
    const w = await mountWith(record())
    expect(w.get('[data-testid="podcast-licence-conflicts"]').text()).toContain('All rights reserved')
    expect(w.get('[data-testid="podcast-licence-documents"] a').attributes('href')).toBe('https://example.com/terms')
    expect(w.get('[data-testid="podcast-licence-read"]').text()).toContain('https://feeds.example/daily')
    expect(w.get('[data-testid="podcast-licence-history"]').text()).toContain('© 2020 NYT')
  })

  it('says plainly when the feed declares nothing, and when it was never read', async () => {
    const none = await mountWith(record({ status: 'no_explicit_licence_found', summary: 'No explicit licence found', licence: { status: 'no_explicit_licence_found', declarations: [], conflicts: [], documents_for_review: [], source_feed_url: null, retrieved_at: null }, history: [] }))
    expect(none.get('[data-testid="podcast-licence-none"]').text()).toContain('not the same as unrestricted')
    vi.restoreAllMocks()
    const unchecked = await mountWith(record({ status: 'not_checked', summary: null, licence: null, history: [] }))
    expect(unchecked.find('[data-testid="podcast-licence-not-checked"]').exists()).toBe(true)
    expect(unchecked.find('[data-testid="podcast-licence-declaration"]').exists()).toBe(false)
  })

  it('reports a failed load instead of an empty page', async () => {
    const w = await mountWith(new Error('500'))
    expect(w.find('[data-testid="podcast-licence-error"]').exists()).toBe(true)
  })
})

describe('PodcastLicenceView — opened from an episode (#2325)', () => {
  function episodeRecord(declared: boolean): EpisodeLicence {
    const own = declared
      ? [decl({ level: 'episode', mechanism: 'podcast_license', raw: 'cc-by-4.0', asset_scope: 'audio_video' })]
      : []
    return {
      slug: 'ep-1', title: 'Ep 1', feed_id: FEED, show_title: 'The Daily',
      summary: declared ? 'CC BY 4.0' : 'x', scope: declared ? 'episode' : 'show',
      episode: { status: declared ? 'declared' : 'no_explicit_licence_found', declarations: own, conflicts: [], documents_for_review: [], source_feed_url: null, retrieved_at: null },
      show: null,
    }
  }

  it("shows the episode's own declaration above the show's, on the same page", async () => {
    vi.spyOn(api, 'getEpisodeLicence').mockResolvedValue(episodeRecord(true))
    const w = await mountWith(record(), 'ep-1')
    const own = w.get('[data-testid="podcast-licence-episode"]')
    expect(own.text()).toContain('This episode: Ep 1')
    expect(own.text()).toContain('cc-by-4.0')
    // The show's declarations are still all there below it.
    expect(w.findAll('[data-testid="podcast-licence-raw"]').map((r) => r.text())).toEqual([
      'cc-by-4.0',
      NYT,
      'cc-by-nc-4.0',
    ])
  })

  it("is the show's page unchanged when the episode declares nothing of its own", async () => {
    vi.spyOn(api, 'getEpisodeLicence').mockResolvedValue(episodeRecord(false))
    const w = await mountWith(record(), 'ep-1')
    expect(w.find('[data-testid="podcast-licence-episode"]').exists()).toBe(false)
  })
})
