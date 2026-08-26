<script setup lang="ts">
/**
 * Trend-window selector (RFC-103 R2) — a compact segmented control (1M · 3M · 6M · 1Y) that picks
 * the window the trending velocity is measured over. Default 3M (the browse/catch-up cadence).
 * `v-model` is the window key; the parent refetches `getTrending(..., window)` on change.
 */
import { useI18n } from 'vue-i18n'
import type { TrendWindow } from '../services/api'

const { t } = useI18n()
const model = defineModel<TrendWindow>({ required: true })

const WINDOWS: { key: TrendWindow; label: string }[] = [
  { key: '1m', label: 'browse.window.1m' },
  { key: '3m', label: 'browse.window.3m' },
  { key: '6m', label: 'browse.window.6m' },
  { key: '1y', label: 'browse.window.1y' },
]
</script>

<template>
  <div
    role="tablist"
    :aria-label="t('browse.windowLabel')"
    class="inline-flex items-center gap-0.5 rounded-full border border-border bg-surface p-0.5"
    data-testid="trend-window-tabs"
  >
    <button
      v-for="w in WINDOWS"
      :key="w.key"
      type="button"
      role="tab"
      :aria-selected="model === w.key"
      :data-testid="`trend-window-${w.key}`"
      class="rounded-full px-2.5 py-1 text-xs font-bold transition"
      :class="
        model === w.key
          ? 'bg-accent text-accent-foreground'
          : 'text-muted hover:text-canvas-foreground'
      "
      @click="model = w.key"
    >
      {{ t(w.label) }}
    </button>
  </div>
</template>
