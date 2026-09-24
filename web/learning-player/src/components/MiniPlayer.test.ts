import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import { createRouter, createWebHistory } from 'vue-router'
import en from '../i18n/locales/en.json'
import { usePlayerStore } from '../stores/player'
import MiniPlayer from './MiniPlayer.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div />' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div />' } },
  ],
})

async function mountMini() {
  await router.push('/')
  await router.isReady()
  return mount(MiniPlayer, { global: { plugins: [i18n, router] } })
}

/** The store's element is never constructed here — only its reactive state matters to this bar. */
function nowPlaying(slug = 'ep-1') {
  const player = usePlayerStore()
  player.currentSlug = slug
  player.currentTitle = 'An Episode'
  return player
}

describe('MiniPlayer audio failure (Player #3)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('says nothing about errors while playback is healthy', async () => {
    nowPlaying()
    const w = await mountMini()
    expect(w.find('[data-testid="mini-player"]').exists()).toBe(true)
    expect(w.find('[data-testid="mini-player-error"]').exists()).toBe(false)
    expect(w.find('[data-testid="mini-player-toggle"]').attributes('disabled')).toBeUndefined()
  })

  it('surfaces a dead source instead of offering a button that does nothing', async () => {
    // Audio can die while the listener is anywhere in the app — most sharply when auto-advance
    // moves to a broken episode with no view mounted. This bar was the only thing on screen
    // claiming to know about playback, and it kept showing a normal play button: pressing it called
    // play(), whose rejection was swallowed, so nothing happened and nothing said why.
    const player = nowPlaying()
    player.audioError = true
    const w = await mountMini()

    const notice = w.find('[data-testid="mini-player-error"]')
    expect(notice.exists()).toBe(true)
    expect(notice.text()).toBe(en.player.audioErrorShort)
    // role=status so it is announced; the disabled icon alone is silent to a screen reader.
    expect(notice.attributes('role')).toBe('status')

    const toggle = w.find('[data-testid="mini-player-toggle"]')
    expect(toggle.attributes('disabled')).toBeDefined()
    expect(toggle.attributes('aria-label')).toBe(en.player.audioErrorShort)
  })

  it('still offers the way back to the episode', async () => {
    // An error must not strand the listener: the title stays tappable so they can reach the full
    // player, which explains the failure properly.
    const player = nowPlaying()
    player.audioError = true
    const w = await mountMini()
    expect(w.find('[data-testid="mini-player-open"]').exists()).toBe(true)
  })
})


describe('MiniPlayer line composition (operator 2026-09-23)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('no longer carries a queue button — the masthead does, at every width', async () => {
    // Asserted as an ABSENCE. Two routes to one place, on the most space-constrained strip in the
    // app, on a screen already showing the other one.
    nowPlaying()
    const w = await mountMini()
    expect(w.find('[data-testid="mini-player-queue"]').exists()).toBe(false)
  })

  it('carries save and add-to-collection for the playing episode', async () => {
    // The slots the queue button gave up. Asserted by IDENTITY, not by label: signed out — which is
    // what a bare mount is — both controls correctly render the same sign-in gate label, so a
    // label-matching test would be unable to tell one from the other or from nothing.
    nowPlaying('ep-42')
    const w = await mountMini()
    expect(w.find('.lp-fav').exists()).toBe(true)
    expect(w.find('[data-testid="add-to-collection"]').exists()).toBe(true)
    // Transport still last, and still the only thing that acts on playback.
    expect(w.find('[data-testid="mini-player-toggle"]').exists()).toBe(true)
  })

  it('reads as a compact ROW — show above, episode below', async () => {
    const player = nowPlaying()
    player.currentShowTitle = 'The Show'
    const w = await mountMini()
    expect(w.text()).toContain('The Show')
    expect(w.text()).toContain('An Episode')
    // The kicker is the show, not the episode — the ordering is the point, not mere presence.
    expect(w.find('.lp-kicker').text()).toBe('The Show')
  })

  it('omits the kicker rather than faking one when the show is unknown', async () => {
    // The offline path rebuilds from the download registry and a caller that does not know the
    // show omits it. A blank kicker line would leave a gap that reads as a loading failure.
    nowPlaying()
    const w = await mountMini()
    expect(w.find('.lp-kicker').exists()).toBe(false)
  })
})

