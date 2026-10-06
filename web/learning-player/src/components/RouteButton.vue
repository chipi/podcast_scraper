<script setup lang="ts">
/**
 * "Play this somewhere else" — the system route picker, from the transport (operator 2026-09-23).
 *
 * ## What it does, and what it deliberately does not
 *
 * Tapping opens the PLATFORM's device sheet: AirPlay on iOS, the Cast/output picker on Android.
 * That sheet is where This iPhone, the Bluetooth speaker and the MacBook are listed.
 *
 * It does NOT render its own device list, and that is not a shortcut. Neither platform exposes an
 * API for a page — or even a native app — to enumerate AirPlay / Cast / Bluetooth audio targets.
 * Spotify's in-app list works because those are Spotify Connect devices: their own protocol, their
 * own servers, their own device registry. Its AirPlay entry still hands off to the system sheet,
 * exactly as this does. Anyone tempted to "finish" this component by listing devices should stop
 * here: the data does not exist on this side of the boundary.
 *
 * ## Why it can be absent
 *
 * Rendered only when the platform reports a route is AVAILABLE, which is the one thing both APIs do
 * tell us (`webkitplaybacktargetavailabilitychanged` / `remote.watchAvailability`). A speaker icon
 * that opens an empty sheet is worse than no icon: it offers a capability the room cannot provide.
 * ANDROID carries neither API in its WebView, so on Android 14+ the button opens the system output
 * switcher natively instead (BackgroundAudioPlugin → MediaRouter2, operator 2026-10-05). Below 14 it
 * stays hidden; the switcher is still on the media notification, which the native session
 * (NowPlaying.java) provides — the WebView's MediaSession never did.
 *
 * ## The active state
 *
 * When audio IS going somewhere else, the icon takes the accent — the same language the queue and
 * download toggles use for "this is on". It matters more here than there: audio leaving the phone
 * is the one player state you cannot see by looking at the screen, and a listener who does not know
 * where the sound went will think the app is broken.
 */
import { storeToRefs } from 'pinia'
import { useI18n } from 'vue-i18n'
import { usePlayerStore } from '../stores/player'

const { t } = useI18n()
const player = usePlayerStore()
const { routeAvailable, playingRemotely } = storeToRefs(player)

// Size is the CALLER's: 32px in the mini-player, the shared transport size on the full player. A
// size passed as a plain `class` lost to the `h-8 w-8` written here, which is how the full player
// showed a 32px speaker between 44px buttons.
withDefaults(defineProps<{ sizeClass?: string }>(), { sizeClass: 'lp-tap h-8 w-8' })
</script>

<template>
  <button
    v-if="routeAvailable"
    type="button"
    data-testid="route-picker"
    class="z-30 flex shrink-0 items-center justify-center rounded-full border transition"
    :class="[
      sizeClass,
      playingRemotely
        ? 'border-accent text-accent'
        : 'border-border text-muted hover:text-canvas-foreground',
    ]"
    :aria-label="playingRemotely ? t('route.playingElsewhere') : t('route.choose')"
    :title="playingRemotely ? t('route.playingElsewhere') : t('route.choose')"
    @click.stop.prevent="player.showRoutePicker()"
  >
    <!-- A speaker with waves — the shape both platforms use for "output", so it needs no learning. -->
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      class="h-4 w-4"
      aria-hidden="true"
    >
      <path d="M4 9v6h4l5 4V5L8 9z" />
      <template v-if="playingRemotely">
        <path d="M16.5 8.5a5 5 0 0 1 0 7" /><path d="M19.5 5.5a9 9 0 0 1 0 13" />
      </template>
      <template v-else><path d="M16.5 8.5a5 5 0 0 1 0 7" /></template>
    </svg>
  </button>
</template>
