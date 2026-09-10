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
import DeviceSettings from '../components/DeviceSettings.vue'
import ConnectedAgents from '../components/ConnectedAgents.vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAuthStore } from '../stores/auth'
import { useVoiceInput } from '../composables/useVoiceInput'
import { useOnline } from '../composables/useOnline'
import { usePlayerStore } from '../stores/player'
import Tabs from '../components/Tabs.vue'
import type { TabSpec } from '../components/tabs'
import { RouterLink } from 'vue-router'
import { Capacitor } from '@capacitor/core'
import { Browser } from '@capacitor/browser'
import { getTier, isInternalBuild } from '../services/tier'
import { CACHE_KEYS, clearCached } from '../services/contentCache'
import { clearAllDownloads } from '../services/downloads'
import { formatPublishDate } from '../utils/format'

const { t, locale } = useI18n()
const auth = useAuthStore()
const { enabled: voiceEnabled, setEnabled: setVoiceEnabled } = useVoiceInput()
const { forcedOffline, setForcedOffline } = useOnline()

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
const native = Capacitor.isNativePlatform()
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

async function openHelp(): Promise<void> {
  if (Capacitor.isNativePlatform()) {
    await Browser.open({ url: HELP_URL }).catch(() => {})
  } else {
    window.open(HELP_URL, '_blank', 'noopener')
  }
}
</script>

<template>
  <section class="mx-auto max-w-2xl px-4 pb-8 pt-4" data-testid="settings-view">
    <RouterLink
      :to="{ name: 'profile' }"
      class="mb-4 inline-flex items-center gap-1 text-sm font-medium text-muted no-underline transition hover:text-canvas-foreground"
    >
      ‹ {{ t('profile.title') }}
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
        <input
          type="checkbox"
          class="h-5 w-5 shrink-0 accent-accent"
          data-testid="settings-voice-input"
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

    <!-- Config (operator 2026-09-09): offline-mode testing switch + space reclaim. -->
    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('settings.config') }}</h2>

      <label class="flex items-center justify-between gap-3">
        <span class="min-w-0">
          <span class="block text-sm font-semibold text-canvas-foreground">{{ t('settings.offlineMode') }}</span>
          <span class="mt-0.5 block text-xs text-muted">{{ t('settings.offlineModeHint') }}</span>
        </span>
        <input
          type="checkbox"
          class="h-5 w-5 shrink-0 accent-accent"
          data-testid="settings-offline-mode"
          :checked="forcedOffline"
          @change="setForcedOffline(($event.target as HTMLInputElement).checked)"
        />
      </label>

      <div class="mt-4 flex flex-col gap-2 border-t border-border pt-4">
        <button
          type="button"
          class="flex items-center justify-between gap-3 text-left text-sm font-semibold text-canvas-foreground disabled:opacity-50"
          data-testid="settings-clear-cache"
          :disabled="busy === 'cache'"
          @click="clearCache"
        >
          <span>{{ t('settings.clearCache') }}</span>
          <span class="shrink-0 text-xs font-normal text-muted">{{ cleared === 'cache' ? t('settings.cleared') : t('settings.clearCacheHint') }}</span>
        </button>
        <button
          v-if="native"
          type="button"
          class="flex items-center justify-between gap-3 text-left text-sm font-semibold text-canvas-foreground disabled:opacity-50"
          data-testid="settings-clear-downloads"
          :disabled="busy === 'downloads'"
          @click="clearDownloads"
        >
          <span>{{ t('settings.clearDownloads') }}</span>
          <span class="shrink-0 text-xs font-normal text-muted">{{ cleared === 'downloads' ? t('settings.cleared') : t('settings.clearDownloadsHint') }}</span>
        </button>
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
      <button
        type="button"
        class="mt-4 rounded-full border border-border px-4 py-1.5 text-sm font-bold transition hover:bg-overlay"
        data-testid="settings-copy"
        @click="copyDiagnostics"
      >
        {{ copied ? t('settings.copied') : t('settings.copyDiagnostics') }}
      </button>
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
