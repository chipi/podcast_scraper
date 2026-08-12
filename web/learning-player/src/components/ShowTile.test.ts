import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createMemoryHistory, createRouter } from 'vue-router'
import type { Podcast } from '../services/types'
import ShowTile from './ShowTile.vue'

const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
  ],
})

function show(title: string | null, feedId = 'f1'): Podcast {
  return {
    feed_id: feedId,
    title,
    artwork_url: null,
    image_url: null,
    description: null,
    episode_count: 3,
  }
}

function mountTile(p: Podcast, lines?: 1 | 2) {
  return mount(ShowTile, {
    props: lines ? { show: p, lines } : { show: p },
    global: { plugins: [router] },
  })
}

describe('ShowTile', () => {
  it('reserves the label box height so grid rows cannot go ragged (#1584)', () => {
    // The bug this component exists to prevent: in a CSS grid the row is as tall as its tallest
    // cell, so an unclamped label makes row height a function of title length. Clamping alone is
    // NOT enough — a 1-line title beside a 2-line one still differs by a line — so the label box
    // must also reserve its full height. Assert both halves.
    const short = mountTile(show('Acquired'))
    const long = mountTile(show('How I Built This with Guy Raz and Friends'))

    for (const w of [short, long]) {
      const label = w.get('div.mt-1')
      expect(label.classes()).toContain('line-clamp-2')
      expect(label.classes()).toContain('min-h-[2.25rem]')
    }

    // Same reserved box regardless of title length — the property that keeps rows uniform.
    expect(short.get('div.mt-1').classes().sort()).toEqual(long.get('div.mt-1').classes().sort())
  })

  it('exposes the full title on hover, since the visible label may be clipped', () => {
    const title = 'How I Built This with Guy Raz and Friends'
    expect(mountTile(show(title)).get('div.mt-1').attributes('title')).toBe(title)
  })

  it('falls back to the feed id when a show has no title', () => {
    const w = mountTile(show(null, 'p09'))
    expect(w.text()).toContain('p09')
  })

  it('single-line variant truncates instead of clamping', () => {
    const label = mountTile(show('Conversations with Tyler'), 1).get('div.mt-1')
    expect(label.classes()).toContain('truncate')
    expect(label.classes()).not.toContain('line-clamp-2')
  })

  it('links to the show page', () => {
    expect(mountTile(show('Acquired', 'p03')).get('a').attributes('href')).toBe('/podcast/p03')
  })
})
