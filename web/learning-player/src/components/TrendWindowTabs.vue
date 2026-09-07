<script setup lang="ts">
/**
 * Trend-window selector (RFC-103 R2) — a compact segmented control (1M · 3M · 6M · 1Y) that picks
 * the window the trending velocity is measured over. Default 3M (the browse/catch-up cadence).
 * `v-model` is the window key; the parent refetches `getTrending(..., window)` on change.
 *
 * ## Why this is a radiogroup and not a tablist (#1594 item 7)
 *
 * It was marked up as `role="tablist"` with `role="tab"` options, and it switches no panel: the
 * rail it re-queries belongs to the PARENT. Adding the `aria-controls` this issue asks for would
 * have meant naming an element this component cannot see, and a dangling reference is worse than
 * the missing one it replaces. "Set the window to one of four values" is a radiogroup.
 *
 * The markup, roving tabindex and arrow keys all come from {@link Tabs.vue} — the keyboard contract
 * is identical for both patterns, which is why one component serves them.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { TrendWindow } from '../services/api'
import Tabs from './Tabs.vue'
import type { TabSpec } from './tabs'

const { t } = useI18n()
const model = defineModel<TrendWindow>({ required: true })

const WINDOWS: { key: TrendWindow; labelKey: string }[] = [
  { key: '1m', labelKey: 'browse.window.1m' },
  { key: '3m', labelKey: 'browse.window.3m' },
  { key: '6m', labelKey: 'browse.window.6m' },
  { key: '1y', labelKey: 'browse.window.1y' },
]

const options = computed<TabSpec<TrendWindow>[]>(() =>
  WINDOWS.map((w) => ({ key: w.key, label: t(w.labelKey), testid: `trend-window-${w.key}` })),
)
</script>

<template>
  <Tabs
    v-model="model"
    :tabs="options"
    :label="t('browse.windowLabel')"
    id-prefix="trend-window"
    variant="pill"
    pattern="radio"
    data-testid="trend-window-tabs"
  />
</template>
