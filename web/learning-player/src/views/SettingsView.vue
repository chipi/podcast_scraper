<script setup lang="ts">
/**
 * Settings / About (#8) — a real destination for app-level info and options, reached from a gear in
 * Profile. Today it surfaces the build identity (version / sha / built-at / platform, and the
 * dev↔prod target on internal builds) plus a Help link and a one-tap "copy build info" for bug
 * reports.
 *
 * Scope note (#1905): options that belong to the DEVICE live in the profile's Device section
 * (`components/DeviceSettings.vue`), not here — they are shared by every account that signs in on
 * the phone, and grouping them with the account's own settings is what makes that legible. This
 * view is build identity and help; it is not the home for every future option.
 */
import BackIcon from '../components/BackIcon.vue'
import DeviceSettings from '../components/DeviceSettings.vue'
import ConnectedAgents from '../components/ConnectedAgents.vue'
import { computed, defineAsyncComponent, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAuthStore } from '../stores/auth'
import { useVoiceInput } from '../composables/useVoiceInput'
import { useOnline } from '../composables/useOnline'
import { usePlayerStore } from '../stores/player'
import Tabs from '../components/Tabs.vue'
import type { TabSpec } from '../components/tabs'
import { RouterLink, useRouter } from 'vue-router'
import { collectDebugInfo } from '../utils/debugInfo'
import { useGuidedStart } from '../composables/useGuidedStart'
import { Capacitor } from '@capacitor/core'
import { isNative } from '../services/native'
import { Browser } from '@capacitor/browser'
import { getTier, isInternalBuild } from '../services/tier'
import { CACHE_KEYS, clearCached } from '../services/contentCache'
import { clearAllDownloads } from '../services/downloads'
import { formatPublishDate } from '../utils/format'
import { identify, resolveSession, setUserOptedOut, userOptedOut } from '../services/analytics'

const { t, locale } = useI18n()
const auth = useAuthStore()
const { enabled: voiceEnabled, setEnabled: setVoiceEnabled } = useVoiceInput()
const { forcedOffline, setForcedOffline } = useOnline()
const router = useRouter()
const guided = useGuidedStart()

/** Run Home's guided start again from step 1 (operator 2026-10-08), then go to Home to do it. */
async function restartGuidedStart(): Promise<void> {
  await guided.restart()
  await router.push({ name: 'home' })
}

// Usage analytics (#2265). The toggle is ON by default; OFF writes Umami's own `umami.disabled`
// flag, which silences the tracker both through our gate and inside its own bundle.
//
// This exists because the beta's closing-interview script tells every participant "analytics
// continues unless you turn it off in Settings" — without the control that sentence is false.
//
// Server-side listen events are deliberately NOT affected: they power the listening stats the user
// can see in their own profile, so silencing them would remove a feature rather than telemetry.
const shareAnalytics = ref(!userOptedOut())
function onShareAnalyticsChange(next: boolean): void {
  shareAnalytics.value = next
  setUserOptedOut(!next)
  // Turning it back ON must re-attach the identity now. Nothing else would until the next
  // `auth.refresh()`, so the sessions in between would be recorded anonymously and drop out of the
  // per-person view the beta check-ins read.
  const aid = auth.user?.analytics_id
  if (next && aid) identify(aid, resolveSession())
}

const analyticsId = computed(() => auth.user?.analytics_id ?? '')
const analyticsIdCopied = ref(false)
async function copyAnalyticsId(): Promise<void> {
  try {
    await navigator.clipboard.writeText(analyticsId.value)
    analyticsIdCopied.value = true
    setTimeout(() => (analyticsIdCopied.value = false), 1500)
  } catch {
    // Clipboard denied (it needs a secure context and a user gesture). The id is on screen and
    // selectable, so the copy button is a convenience, not the only route to it.
  }
}

// Playback volume (low/med/high) — a persisted in-app multiplier over the OS volume (player store).
const player = usePlayerStore()
type VolumeLevel = 'low' | 'medium' | 'high'
const volumeOptions = computed<TabSpec<VolumeLevel>[]>(() => [
  { key: 'low', label: t('settings.volume_low') },
  { key: 'medium', label: t('settings.volume_medium') },
  { key: 'high', label: t('settings.volume_high') },
])

// Config actions (operator 2026-09-09). Offline-mode is a testing switch (forces the whole app
// offline on a live network); the two "clear" actions free space + let you re-fetch fresh.
const native = isNative()
const busy = ref<'' | 'cache' | 'downloads'>('')
const cleared = ref<'' | 'cache' | 'downloads'>('')
async function clearCache(): Promise<void> {
  busy.value = 'cache'
  try {
    await clearCached(CACHE_KEYS)
    cleared.value = 'cache'
    window.setTimeout(() => (cleared.value = ''), 1500)
  } finally {
    busy.value = ''
  }
}
async function clearDownloads(): Promise<void> {
  busy.value = 'downloads'
  try {
    await clearAllDownloads()
    cleared.value = 'downloads'
    window.setTimeout(() => (cleared.value = ''), 1500)
  } finally {
    busy.value = ''
  }
}

const HELP_URL = 'https://closelistening.app'
const SUPPORT_URL = 'https://closelistening.app/support'

async function openSupport(): Promise<void> {
  if (Capacitor.isNativePlatform()) {
    await Browser.open({ url: SUPPORT_URL }).catch(() => {})
  } else {
    window.open(SUPPORT_URL, '_blank', 'noopener')
  }
}


const version = __APP_VERSION__
const sha = (__BUILD_SHA__ || '').slice(0, 7)
const builtAt = formatPublishDate(__BUILD_TIME__, locale.value) ?? __BUILD_TIME__
const platform = Capacitor.getPlatform() // 'ios' | 'android' | 'web'
const internal = isInternalBuild()

/**
 * The dev↔prod tier switch, moved here from the masthead (operator 2026-09-29).
 *
 * It belongs beside the build identity it changes — version, sha, platform, target — rather than
 * in the app's top bar, where it sat on every screen of every internal build and was one mistap
 * away from pointing a phone at a private dev API.
 *
 * Imported DYNAMICALLY behind the RAW build-time constant, which is the part that actually
 * matters. `vite.config.ts` claimed a release build "tree-shakes the switch out"; it does not,
 * and never did — a static import plus a runtime `v-if` inside the component puts the component
 * in the bundle unconditionally and gates only its RENDERING. Verified on a real prod-locked
 * artifact, where `tier-switch` was still present.
 *
 * `__MOBILE_INTERNAL__`, not `isInternalBuild()`: Vite substitutes the constant for a literal, so
 * the ternary becomes `false ? … : null` and Rollup drops the import as unreachable. The helper
 * would compute the same answer, but it is a cross-module function call — the bundler cannot see
 * through it, and the chunk is emitted anyway. Also verified: with the helper the switch got its
 * own chunk and still shipped; with the constant it is gone.
 */
const TierSwitch = __MOBILE_INTERNAL__
  ? defineAsyncComponent(() => import('../components/TierSwitch.vue'))
  : null
const target = getTier() // 'dev' | 'prod'

const copied = ref(false)
async function copyDiagnostics(): Promise<void> {
  const info = `Close Listening ${version} · ${sha} · ${platform} · built ${__BUILD_TIME__}`
  try {
    await navigator.clipboard.writeText(info)
    copied.value = true
    window.setTimeout(() => (copied.value = false), 1500)
  } catch {
    /* clipboard blocked (insecure context / denied) — no-op, the info is still on screen */
  }
}

/**
 * "Copy debug info" (operator 2026-10-08): device, OS, WebView, GPU and memory as one pasteable block,
 * so a tester who sees something odd sends it straight from here. The screen recorded is the one
 * they CAME FROM — the one that misbehaved — not Settings.
 */
const debugCopied = ref(false)
async function copyDebugInfo(): Promise<void> {
  const back = (window.history.state as { back?: string } | null)?.back ?? '(unknown)'
  const text = await collectDebugInfo({
    version,
    sha: sha || '—',
    builtAt: String(__BUILD_TIME__),
    platform,
    target,
    route: back,
    userId: auth.user?.user_id ?? null,
  })
  try {
    await navigator.clipboard.writeText(text)
    debugCopied.value = true
    window.setTimeout(() => (debugCopied.value = false), 1500)
  } catch {
    /* clipboard blocked — nothing to fall back to without showing the block */
  }
}

async function openHelp(): Promise<void> {
  if (Capacitor.isNativePlatform()) {
    await Browser.open({ url: HELP_URL }).catch(() => {})
  } else {
    window.open(HELP_URL, '_blank', 'noopener')
  }
}
</script>

<template>
  <section class="lp-page lp-focus pb-8 pt-4" data-testid="settings-view">
    <RouterLink
      :to="{ name: 'profile' }"
      class="mb-4 inline-flex items-center gap-1 text-sm font-medium text-muted no-underline transition hover:text-canvas-foreground"
    >
      <BackIcon /> {{ t('profile.title') }}
    </RouterLink>
    <h1 class="mb-1 font-display text-3xl font-extrabold tracking-tight">{{ t('settings.title') }}</h1>
    <p class="mb-5 text-sm text-muted">{{ t('settings.subtitle') }}</p>

    <!-- Device (#1905, moved here from the profile) — download network policy, size cap, storage.
         Its own contract is "settings that belong to THIS PHONE rather than to the account", which
         is this page's subject and not the profile's: the profile is who you are, and these are
         shared by every account that signs in on this handset. First on the page, because what you
         can CHANGE outranks the version number you can only read. -->
    <DeviceSettings />

    <!-- Voice input (operator 2026-09-09) — opt-in, default OFF. Gates note dictation so the mic
         never listens unless the user turns it on here. -->
    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('settings.voice') }}</h2>
      <label class="flex items-center justify-between gap-3">
        <span class="min-w-0">
          <span class="block text-sm font-semibold text-canvas-foreground">{{ t('settings.voiceInput') }}</span>
          <span class="mt-0.5 block text-xs text-muted">{{ t('settings.voiceInputHint') }}</span>
        </span>
        <!-- NAMED ON THE INPUT, not only by the wrapping <label> (2026-09-28, measured).
             The implicit label association names this for the DOM. Android's accessibility bridge
             reported the node with EVERY name field empty — text, contentDescription, labeledBy,
             hintText, stateDescription — so TalkBack announced a bare checkbox with no idea what it
             toggles, and `Journey.setOfflineMode` cannot address it by name at all: it finds the row
             text and walks to `nearestCheckable`, the heuristic that drove this VOICE switch instead
             of Offline three times while reporting success (Journey.java:600-604).
             The string matches the row label exactly, so the two cannot drift. (#2156) -->
        <input
          type="checkbox"
          class="lp-check"
          data-testid="settings-voice-input"
          :aria-label="t('settings.voiceInput')"
          :checked="voiceEnabled"
          @change="setVoiceEnabled(($event.target as HTMLInputElement).checked)"
        />
      </label>
    </section>

    <!-- Playback (operator 2026-09-09): in-app volume level, a multiplier over the OS volume. -->
    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('settings.playback') }}</h2>
      <div class="flex items-center justify-between gap-3">
        <span class="min-w-0">
          <span class="block text-sm font-semibold text-canvas-foreground">{{ t('settings.volume') }}</span>
          <span class="mt-0.5 block text-xs text-muted">{{ t('settings.volumeHint') }}</span>
        </span>
        <Tabs
          :model-value="player.volumeLevel"
          :tabs="volumeOptions"
          :label="t('settings.volume')"
          id-prefix="settings-volume"
          variant="segment"
          pattern="radio"
          @update:model-value="player.setVolumeLevel"
        />
      </div>
    </section>

    <!-- Privacy (#2265): the usage-analytics opt-out, and the pseudonymous id behind it.
         Both are promises the beta makes out loud — the closing interview tells each participant
         the toggle exists, and the first session has the operator note the id down. -->
    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('settings.privacyHeading') }}</h2>

      <label class="flex items-center justify-between gap-3">
        <span class="min-w-0">
          <span class="block text-sm font-semibold text-canvas-foreground">{{
            t('settings.shareAnalytics')
          }}</span>
          <span class="mt-0.5 block text-xs text-muted">{{ t('settings.shareAnalyticsHint') }}</span>
        </span>
        <!-- Named on the INPUT, not only by the wrapping label: the Android accessibility bridge
             reported such a node with every name field empty, so TalkBack announced a bare
             checkbox with no idea what it toggles (#2156). -->
        <input
          type="checkbox"
          class="lp-check"
          data-testid="settings-share-analytics"
          :aria-label="t('settings.shareAnalytics')"
          :checked="shareAnalytics"
          @change="onShareAnalyticsChange(($event.target as HTMLInputElement).checked)"
        />
      </label>

      <div v-if="analyticsId" class="mt-4 border-t border-border pt-4">
        <span class="block text-sm font-semibold text-canvas-foreground">{{
          t('settings.analyticsId')
        }}</span>
        <span class="mt-0.5 block text-xs text-muted">{{ t('settings.analyticsIdHint') }}</span>
        <div class="mt-2 flex items-center justify-between gap-3">
          <!-- Selectable, so it is recoverable even where the clipboard API is denied. -->
          <code
            class="min-w-0 flex-1 truncate font-mono text-xs text-muted select-all"
            data-testid="settings-analytics-id"
            >{{ analyticsId }}</code
          >
          <button
            type="button"
            class="shrink-0 text-xs font-semibold text-accent"
            data-testid="settings-copy-analytics-id"
            @click="copyAnalyticsId"
          >
            {{ analyticsIdCopied ? t('settings.analyticsIdCopied') : t('settings.analyticsIdCopy') }}
          </button>
        </div>
      </div>
    </section>

    <!-- Config (operator 2026-09-09): offline-mode testing switch + space reclaim. -->
    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('settings.config') }}</h2>

      <label class="flex items-center justify-between gap-3">
        <span class="min-w-0">
          <span class="block text-sm font-semibold text-canvas-foreground">{{ t('settings.offlineMode') }}</span>
          <span class="mt-0.5 block text-xs text-muted">{{ t('settings.offlineModeHint') }}</span>
        </span>
        <!-- Named on the input for the reason on the Voice switch above. This is the control that
             heuristic actually drives, so it is the one whose namelessness cost three silent
             wrong-control flips. -->
        <input
          type="checkbox"
          class="lp-check"
          data-testid="settings-offline-mode"
          :aria-label="t('settings.offlineMode')"
          :checked="forcedOffline"
          @change="setForcedOffline(($event.target as HTMLInputElement).checked)"
        />
      </label>

      <div class="mt-4 flex flex-col gap-2 border-t border-border pt-4">
        <button
          v-if="auth.isAuthenticated"
          type="button"
          class="flex items-center justify-between gap-3 text-left text-sm font-semibold text-canvas-foreground"
          data-testid="settings-guided-restart"
          @click="restartGuidedStart"
        >
          <span>{{ t('settings.guidedRestart') }}</span>
          <span class="shrink-0 text-xs font-normal text-muted">{{ t('settings.guidedRestartHint') }}</span>
        </button>
      </div>
      <!-- Clear cache and Remove downloads: two small buttons side by side (operator 2026-10-08), each
           with its one-line consequence under it; the label says "Cleared" once done. -->
      <div class="mt-3 grid gap-2" :class="native ? 'grid-cols-2' : 'grid-cols-1'" data-testid="settings-reclaim-row">
        <div>
          <button
            type="button"
            class="w-full rounded-2xl border border-border px-2 py-2.5 text-xs font-semibold text-muted transition hover:text-canvas-foreground disabled:opacity-50"
            data-testid="settings-clear-cache"
            :disabled="busy === 'cache'"
            @click="clearCache"
          >
            {{ cleared === 'cache' ? t('settings.cleared') : t('settings.clearCache') }}
          </button>
          <p class="mt-1 text-center text-[11px] leading-snug text-muted">{{ t('settings.clearCacheHint') }}</p>
        </div>
        <div v-if="native">
          <button
            type="button"
            class="w-full rounded-2xl border border-border px-2 py-2.5 text-xs font-semibold text-muted transition hover:text-danger disabled:opacity-50"
            data-testid="settings-clear-downloads"
            :disabled="busy === 'downloads'"
            @click="clearDownloads"
          >
            {{ cleared === 'downloads' ? t('settings.cleared') : t('settings.clearDownloads') }}
          </button>
          <p class="mt-1 text-center text-[11px] leading-snug text-muted">{{ t('settings.clearDownloadsHint') }}</p>
        </div>
      </div>
    </section>

    <!-- Connected agents (RFC-112 §5) — app-level MCP connections belong with app settings, not on
         the profile (ST.2). Only for users with the mcp_access entitlement. -->
    <ConnectedAgents v-if="auth.user?.mcp_access" class="mt-6" />

    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('settings.about') }}</h2>
      <dl class="flex flex-col gap-2 text-sm">
        <div class="flex items-center justify-between gap-3">
          <dt class="text-muted">{{ t('settings.version') }}</dt>
          <dd class="font-semibold tabular-nums" data-testid="settings-version">v{{ version }}</dd>
        </div>
        <div class="flex items-center justify-between gap-3">
          <dt class="text-muted">{{ t('settings.build') }}</dt>
          <dd class="font-mono text-xs" data-testid="settings-build">{{ sha || '—' }}</dd>
        </div>
        <div class="flex items-center justify-between gap-3">
          <dt class="text-muted">{{ t('settings.builtAt') }}</dt>
          <dd class="text-right">{{ builtAt }}</dd>
        </div>
        <div class="flex items-center justify-between gap-3">
          <dt class="text-muted">{{ t('settings.platform') }}</dt>
          <dd class="font-semibold capitalize">{{ platform }}</dd>
        </div>
        <div v-if="internal" class="flex items-center justify-between gap-3">
          <dt class="text-muted">{{ t('settings.target') }}</dt>
          <dd
            class="font-bold uppercase"
            :class="target === 'dev' ? 'text-danger' : 'text-canvas-foreground'"
          >
            {{ target }}
          </dd>
        </div>
      </dl>
      <!-- The switch that CHANGES the target shown above, next to the target itself. `component
           :is` because the import is build-gated to null on a release build; `v-if` on the value,
           not on `internal`, so there is exactly one condition rather than two that can disagree. -->
      <!-- The tier switch and Copy build info on one row, the same size (operator 2026-10-08). -->
      <div class="mt-4 flex flex-wrap items-center gap-2">
        <component :is="TierSwitch" v-if="TierSwitch" />
        <button
          type="button"
          class="rounded-full border border-border px-4 py-1.5 text-sm font-bold transition hover:bg-overlay"
          data-testid="settings-copy"
          @click="copyDiagnostics"
        >
          {{ copied ? t('settings.copied') : t('settings.copyDiagnostics') }}
        </button>
        <button
          type="button"
          class="rounded-full border border-border px-4 py-1.5 text-sm font-bold transition hover:bg-overlay"
          data-testid="settings-copy-debug"
          @click="copyDebugInfo"
        >
          {{ debugCopied ? t('settings.copied') : t('settings.copyDebug') }}
        </button>
      </div>
      <p class="mt-2 text-xs text-muted">{{ t('settings.copyDebugHint') }}</p>
    </section>

    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('settings.help') }}</h2>
      <button
        type="button"
        class="flex w-full items-center justify-between gap-3 text-left"
        data-testid="settings-help"
        @click="openHelp"
      >
        <span class="text-sm font-semibold text-canvas-foreground">{{ t('settings.helpDesc') }}</span>
        <span class="shrink-0 text-muted" aria-hidden="true">›</span>
      </button>
    </section>

    <!-- About & legal (operator 2026-09-09): Support is an external link; the other three are
         in-app placeholder pages (empty content for now, copy drops in later). -->
    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('settings.aboutLegal') }}</h2>
      <div class="flex flex-col">
        <button
          type="button"
          class="flex items-center justify-between gap-3 border-b border-border py-2.5 text-left text-sm font-semibold text-canvas-foreground"
          data-testid="settings-support"
          @click="openSupport"
        >
          <span>{{ t('about.support') }}</span>
          <span class="shrink-0 text-muted" aria-hidden="true">↗</span>
        </button>
        <RouterLink
          :to="{ name: 'about-page', params: { page: 'third-party' } }"
          class="flex items-center justify-between gap-3 border-b border-border py-2.5 text-sm font-semibold text-canvas-foreground no-underline"
          data-testid="settings-third-party"
        >
          <span>{{ t('about.thirdParty') }}</span>
          <span class="shrink-0 text-muted" aria-hidden="true">›</span>
        </RouterLink>
        <RouterLink
          :to="{ name: 'about-page', params: { page: 'privacy' } }"
          class="flex items-center justify-between gap-3 border-b border-border py-2.5 text-sm font-semibold text-canvas-foreground no-underline"
          data-testid="settings-privacy"
        >
          <span>{{ t('about.privacy') }}</span>
          <span class="shrink-0 text-muted" aria-hidden="true">›</span>
        </RouterLink>
        <RouterLink
          :to="{ name: 'about-page', params: { page: 'terms' } }"
          class="flex items-center justify-between gap-3 py-2.5 text-sm font-semibold text-canvas-foreground no-underline"
          data-testid="settings-terms"
        >
          <span>{{ t('about.terms') }}</span>
          <span class="shrink-0 text-muted" aria-hidden="true">›</span>
        </RouterLink>
      </div>
    </section>

  </section>
</template>
