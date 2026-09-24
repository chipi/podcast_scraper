<script setup lang="ts">
/**
 * The ONE mark-for-offline control (#1905). Same behaviour wherever it appears.
 *
 * Native only — hidden rather than disabled on web, because the web build has no offline audio
 * story at all (Capacitor's web Filesystem would put third-party audio in IndexedDB, which is the
 * thing PRD-035 Principle 4 exists to prevent).
 *
 * `queued` renders as its own visible state ("Waiting for Wi-Fi"). Under the L1 design a flagged
 * episode legitimately sits idle until the app is open on an allowed connection, and without a
 * distinct affordance that is indistinguishable from a broken button.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDownloadsStore } from '../stores/downloads'
import { getNetworkPolicy, markForOffline } from '../services/downloadScheduler'
import { deleteEpisode } from '../services/downloads'
import { isNative } from '../services/native'
import { useSignInGate } from '../composables/useSignInGate'

const props = withDefaults(
  defineProps<{
    slug: string
    /** `icon` is the ghost-circle in an action row; `menuitem` is a full-width row inside the
     *  player's ⋯ overflow (operator 2026-09-13 — download is a secondary action there, not a
     *  primary transport control). */
    variant?: 'icon' | 'menuitem'
  }>(),
  { variant: 'icon' },
)
const emit = defineEmits<{ activated: [] }>()
const { t } = useI18n()
const downloads = useDownloadsStore()

/**
 * Downloading requires a session (#1912). Two reasons, and the second is the one users feel:
 *  - the episode routes become auth-gated in #1063/#1066, at which point a signed-out download
 *    would 401 and leave a silent `failed` row — and `__checks__/auth-gate.test.ts` cannot catch
 *    that, because it scans store actions rather than service functions;
 *  - the registry is namespaced per account, so anything downloaded while signed out lands under
 *    `anon` and VANISHES the moment the user signs in. That reads as data loss.
 */
const { isGated, gated } = useSignInGate()

const wifiOnly = ref(true)

onMounted(async () => {
  // The button only renders on native (`v-if="native"`). It is now mounted on every episode
  // surface via the shared EpisodeActions row, so on web it must do NO download/device work —
  // otherwise every card touches the downloads store + Capacitor Preferences for a control that
  // never appears.
  if (!isNative()) return
  void downloads.ensureLoaded()
  // "Waiting for Wi-Fi" is a lie for someone who allowed cellular — they are just offline.
  wifiOnly.value = (await getNetworkPolicy()) === 'wifi-only'
})

const native = isNative()
const state = computed(() => downloads.stateOf(props.slug))
const pct = computed(() => Math.round(downloads.progressOf(props.slug) * 100))
const errorKind = computed(() => downloads.entry(props.slug)?.errorKind)
const permanentlyGone = computed(() => errorKind.value === 'permanent')

const label = computed(() => {
  if (isGated.value) return t('auth.signInToDownload')
  switch (state.value) {
    case 'queued':
      return wifiOnly.value ? t('downloads.waitingWifi') : t('downloads.waitingConnection')
    case 'downloading':
      // A host that sends no Content-Length leaves progress at 0 forever, so an
      // indeterminate label is the honest one rather than "0%".
      return pct.value > 0 ? t('downloads.downloadingPct', { pct: pct.value }) : t('downloads.downloading')
    case 'downloaded':
      return t('downloads.downloaded')
    case 'failed':
      if (permanentlyGone.value) return t('downloads.unavailable')
      // A space refusal is not a transient failure: tapping again cannot help until room is made.
      if (errorKind.value === 'needs-space') return t('downloads.needsSpace')
      return t('downloads.retry')
    default:
      return t('downloads.download')
  }
})

const onClick = gated(async () => {
  const s = state.value
  // A permanently gone episode (corpus removal) must not offer a retry that can only fail again —
  // but it must still be REMOVABLE. Disabling the button left an undeletable row in the Downloaded
  // list forever, since the drain skips permanent failures and nothing else clears them (#1905).
  if (s === 'downloaded' || s === 'queued' || s === 'downloading' || permanentlyGone.value) {
    await deleteEpisode(props.slug)
    return
  }
  await markForOffline(props.slug)
})

// As a ⋯ menu item, tapping should also dismiss the overflow (like every other item there); the
// caller listens for `activated` and calls the menu's `close`.
function onActivate(): void {
  void onClick()
  emit('activated')
}
</script>

<template>
  <!-- Downloaded takes the ACCENT border+glyph, the same language QueueButton uses for "already
       queued" (operator 2026-09-23). A muted glyph on a muted border said "downloaded" only to
       someone already looking for it — and in Up next, where this control is now inline, the whole
       point is to see which episodes are on the device without opening the ⋯. -->
  <button
    v-if="native"
    type="button"
    data-testid="download-button"
    :data-state="state ?? 'none'"
    :data-menuitem="variant === 'menuitem' ? '' : undefined"
    :role="variant === 'menuitem' ? 'menuitem' : undefined"
    :class="
      variant === 'menuitem'
        ? 'flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-canvas-foreground transition hover:bg-overlay'
        : [
            'lp-tap z-30 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border',
            state === 'downloaded'
              ? 'border-accent text-accent'
              : state === 'failed'
                ? 'border-border text-muted'
                : 'border-border text-muted hover:text-canvas-foreground',
          ]
    "
    :aria-label="label"
    :title="variant === 'menuitem' ? undefined : label"
    :aria-busy="state === 'downloading' ? 'true' : undefined"
    @click.stop.prevent="onActivate"
  >
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      class="h-4 w-4 shrink-0"
      aria-hidden="true"
    >
      <!-- downloading: a partial ring, so motion is implied without animating a spinner -->
      <template v-if="state === 'downloading'">
        <path d="M12 3a9 9 0 1 1-9 9" />
      </template>
      <!-- queued: a clock, distinct from "nothing is happening" -->
      <template v-else-if="state === 'queued'">
        <circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />
      </template>
      <!-- failed: an alert -->
      <template v-else-if="state === 'failed'">
        <circle cx="12" cy="12" r="9" /><path d="M12 8v5" /><path d="M12 16h.01" />
      </template>
      <!-- Default AND downloaded: the same download arrow, with colour carrying the state
           (operator 2026-09-23). It used to swap to a bare check when downloaded, which made the
           control change IDENTITY — scanning a list, a tick reads as "selected" or "done", not as
           "this one is on your device", and it no longer matched the thing beside it. The glyph now
           says what the control IS and the accent says whether it is on, exactly as the queue
           toggle does one slot to the left. -->
      <template v-else>
        <path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M5 21h14" />
      </template>
    </svg>
    <span v-if="variant === 'menuitem'">{{ label }}</span>
    <span
      v-else-if="state === 'downloading' && pct > 0"
      class="absolute -bottom-4 text-[10px] tabular-nums text-muted"
      >{{ pct }}%</span
    >
  </button>
</template>
