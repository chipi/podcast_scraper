import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import { usePlayerStore } from '../stores/player'
import RouteButton from './RouteButton.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountBtn = () => mount(RouteButton, { global: { plugins: [i18n] } })

/**
 * The output-route control (operator 2026-09-23).
 *
 * The behaviour that matters is when it is ABSENT and what it delegates to — not its markup. It
 * opens the PLATFORM's device sheet; neither iOS nor Android lets a page enumerate AirPlay / Cast /
 * Bluetooth targets, so there is no in-app device list to test and there never will be.
 */
describe('RouteButton', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('does not render when the platform reports nowhere to send audio', () => {
    // The default, and the common case on a desktop browser. A speaker icon that opens an empty
    // sheet offers a capability the room cannot provide.
    expect(mountBtn().find('[data-testid="route-picker"]').exists()).toBe(false)
  })

  it('appears once a route becomes available', () => {
    usePlayerStore().routeAvailable = true
    expect(mountBtn().find('[data-testid="route-picker"]').exists()).toBe(true)
  })

  it('hands off to the system picker — it does not open anything of its own', () => {
    const player = usePlayerStore()
    player.routeAvailable = true
    const spy = vi.spyOn(player, 'showRoutePicker').mockImplementation(() => {})
    const w = mountBtn()
    w.get('[data-testid="route-picker"]').trigger('click')
    expect(spy).toHaveBeenCalled()
  })

  it('says where the audio IS, not just that it could move', () => {
    // Audio leaving the phone is the one player state you cannot see by looking at the screen. A
    // listener who does not know where the sound went concludes the app is broken.
    const player = usePlayerStore()
    player.routeAvailable = true
    player.playingRemotely = true
    const btn = mountBtn().get('[data-testid="route-picker"]')
    expect(btn.attributes('aria-label')).toBe('Playing on another device')
    // Accent, the same "this is on" language the queue and download toggles use.
    expect(btn.classes().join(' ')).toContain('text-accent')
  })

  it('offers the move when audio is still local', () => {
    usePlayerStore().routeAvailable = true
    const btn = mountBtn().get('[data-testid="route-picker"]')
    expect(btn.attributes('aria-label')).toBe('Play on another device')
    expect(btn.classes().join(' ')).not.toContain('text-accent')
  })
})
