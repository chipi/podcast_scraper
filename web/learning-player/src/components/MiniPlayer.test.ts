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
    { path: '/player/:slug', name: 'player', component: { template: '<div />' } },
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
