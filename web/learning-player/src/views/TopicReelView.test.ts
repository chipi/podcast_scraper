/**
 * A short listen (`/listen/topic/:id`, operator 2026-10-11): no start page — it starts on clip 1 at
 * once, and where the browser blocks the sound clip 1 waits with ▶; one bar of time heard of total;
 * the clips to come as a list, each a jump; the title is the subject with its real length.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import { usePlayerStore } from '../stores/player'

const api = vi.hoisted(() => ({ getTopicReel: vi.fn() }))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: () => undefined,
}))

import TopicReelView from './TopicReelView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/listen/topic/:id', name: 'topic-reel', component: TopicReelView },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/browse', name: 'browse', component: { template: '<div/>' } },
  ],
})
const clip = (n: number, slug: string) => ({
  episode: { slug, title: `Episode ${slug}`, podcast_title: 'Show', artwork_url: `https://img.test/${slug}.jpg` },
  insight_id: `i${n}`,
  text: `Claim ${n}`,
  quote_text: `Said ${n}`,
  speaker: `Speaker ${n}`,
  start_ms: 10_000,
  end_ms: 22_000, // 12 s each
})
const REEL = {
  subject_id: 'thc:x',
  subject_kind: 'storyline',
  subject_label: 'Second-order effects',
  minutes: 15,
  total_seconds: 36,
  clips: [clip(1, 'a'), clip(2, 'b'), clip(3, 'a')],
}

async function mountView(at?: number) {
  setActivePinia(createPinia())
  const player = usePlayerStore()
  const started = vi.spyOn(player, 'startTopicReel').mockImplementation(async (topic, moments, _r, at = 0) => {
    player.reel = { slug: moments[at].slug!, topic, moments, index: at, returnTo: 0, done: false }
    return true
  })
  const jumped = vi.spyOn(player, 'reelGo').mockImplementation(async (i: number) => {
    if (player.reel) player.reel = { ...player.reel, index: i }
  })
  api.getTopicReel.mockResolvedValue(REEL)
  await router.push('/listen/topic/thc:x')
  await router.isReady()
  const w = mount(TopicReelView, { props: { id: 'thc:x', at }, global: { plugins: [i18n, router] } })
  await flushPromises()
  return { w, player, started, jumped }
}

afterEach(() => vi.restoreAllMocks())

describe('TopicReelView', () => {
  it('titles the subject with what it really runs, starts on clip 1, and ▶ resumes a blocked start', async () => {
    const { w, player, started } = await mountView()
    expect(w.get('[data-testid="topic-reel-title"]').text()).toBe('Second-order effects')
    expect(w.get('[data-testid="topic-reel-length"]').text().replace(/\s+/g, ' ')).toBe('0:36 · 3 clips · 2 episodes')
    expect(started).toHaveBeenCalledOnce()
    expect(w.get('[data-testid="topic-reel-current"]').text()).toContain('Clip 1 of 3')
    expect(w.get('[data-testid="topic-reel-quote"]').text()).toBe('“Said 1”')
    expect(w.findAll('[data-testid="topic-reel-up-next"]')).toHaveLength(2)
    // Each clip to come carries its episode's cover (operator 2026-10-15).
    const art = w.findAll('[data-testid="topic-reel-up-next-art"]').map((i) => i.attributes('src'))
    expect(art).toHaveLength(2)
    expect(art[0]).toContain('b.jpg')
    expect(art[1]).toContain('a.jpg')
    // The browser blocked the sound (no tap behind the open): ▶, and a tap resumes, not restarts.
    expect(player.playing).toBe(false)
    const resumed = vi.spyOn(player, 'resumeAfterInterruption').mockImplementation(() => undefined)
    await w.get('[data-testid="topic-reel-play"]').trigger('click')
    expect(resumed).toHaveBeenCalledOnce()
    expect(started).toHaveBeenCalledOnce()
    player.playing = true
    await flushPromises()
    expect(w.find('[data-testid="topic-reel-pause"]').exists()).toBe(true)
  })

  it('the bar counts the clips heard; a listed clip is a jump', async () => {
    const { w, player, started, jumped } = await mountView()
    expect(started).toHaveBeenCalledOnce()
    const bar = w.get('[data-testid="topic-reel-progress"] [role="progressbar"]')
    expect(bar.attributes('aria-valuenow')).toBe('0')
    // 6 s into clip 2 of three 12-second clips: 18 of 36 seconds heard.
    player.reel = { ...player.reel!, index: 1 }
    player.currentTime = 16
    await flushPromises()
    expect(bar.attributes('aria-valuenow')).toBe('50')
    expect(w.get('[data-testid="topic-reel-progress"]').text()).toContain('0:18')
    expect(w.findAll('[data-testid="topic-reel-up-next"]')).toHaveLength(1)
    await w.get('[data-testid="topic-reel-up-next"]').trigger('click')
    expect(jumped).toHaveBeenCalledWith(2)
    await flushPromises()
    expect(w.get('[data-testid="topic-reel-current"]').text()).toContain('Clip 3 of 3')
  })

  it("leaves for the clip's episode on from the clip, or from its start (operator 2026-10-15)", async () => {
    for (const [id, seeks] of [['topic-reel-keep', 0], ['topic-reel-play-start', 1]] as const) {
      const { w, player } = await mountView()
      const seek = vi.spyOn(player, 'seek').mockImplementation(() => undefined)
      const exit = vi.spyOn(player, 'exitReel').mockImplementation(() => {
        player.reel = null
      })
      expect(w.get('[data-testid="topic-reel-listen"]').text()).toContain('Listen to the full episode')
      await w.get(`[data-testid="${id}"]`).trigger('click')
      await flushPromises()
      expect(exit, id).toHaveBeenCalledWith(true)
      expect(seek, id).toHaveBeenCalledTimes(seeks)
      if (seeks) expect(seek).toHaveBeenCalledWith(0)
      expect(router.currentRoute.value.fullPath, id).toBe('/episode/a')
      w.unmount()
      vi.restoreAllMocks()
    }
  })

  it('resumes at the clip Home asks for (`?at=`, 1-based), clamped to the reel', async () => {
    const { w, started } = await mountView(3)
    expect(started.mock.calls[0][3]).toBe(2)
    expect(w.get('[data-testid="topic-reel-current"]').text()).toContain('Clip 3 of 3')
    const far = await mountView(99)
    expect(far.started.mock.calls[0][3]).toBe(2)
  })
})
