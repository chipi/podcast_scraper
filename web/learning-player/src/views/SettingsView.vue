<script setup lang="ts">
/**
 * Settings / About (#8) — a real destination for app-level info and options, reached from a gear in
 * Profile. Today it surfaces the build identity (version / sha / built-at / platform, and the
 * dev↔prod target on internal builds) plus a Help link and a one-tap "copy build info" for bug
 * reports. It is the scaffold the operator asked for: future app options land here, not buried in
 * the per-user Profile prefs.
 */
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { Capacitor } from '@capacitor/core'
import { Browser } from '@capacitor/browser'
import { getTier, isInternalBuild } from '../services/tier'
import { formatPublishDate } from '../utils/format'

const { t, locale } = useI18n()

const HELP_URL = 'https://closelistening.app'

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

    <section class="rounded-2xl border border-border p-5">
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

    <p class="mt-6 text-center text-xs text-muted">{{ t('settings.optionsSoon') }}</p>
  </section>
</template>
