<script setup lang="ts">
/**
 * Resurfacing inbox (P3 Consolidation, #1125 / RFC-101 §5) — your past highlights, resurfaced on a
 * spaced schedule, each with a reflection prompt + one-tap jump back to the moment. Pacing controls
 * (pause/resume) live here. Read-time: the server decides what's due; this just renders + dismisses.
 * Embedded in the Library "Revisit" tab. Auth-gated (empty signed out).
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getResurfacing, markSurfaced, putResurfacingSettings } from '../services/api'
import type { ResurfacingItem } from '../services/types'
import { formatTime } from '../player/transcriptSync'

const { t } = useI18n()

const items = ref<ResurfacingItem[]>([])
const paused = ref(false)
const loaded = ref(false)

async function load(): Promise<void> {
  const resp = await getResurfacing()
  items.value = resp.items
  paused.value = resp.paused
  loaded.value = true
}

/** Mark a highlight seen → drop it from the list (the server advances its ladder). */
async function dismiss(item: ResurfacingItem): Promise<void> {
  items.value = items.value.filter((i) => i.highlight.id !== item.highlight.id)
  await markSurfaced(item.highlight.id)
}

async function togglePause(): Promise<void> {
  const next = !paused.value
  paused.value = next
  await putResurfacingSettings(next)
  await load() // pausing empties the due list; resuming re-fills it
}

function jumpQuery(item: ResurfacingItem): Record<string, string> {
  const ms = item.highlight.start_ms
  return ms != null ? { t: String(Math.floor(ms / 1000)) } : {}
}

function label(item: ResurfacingItem): string {
  const h = item.highlight
  return h.kind === 'moment' ? t('revisit.moment') : (h.quote_text ?? t('revisit.span'))
}

onMounted(load)
</script>

<template>
  <div>
    <div class="mb-4 flex items-center justify-between gap-3">
      <p class="text-sm text-muted">{{ t('revisit.intro') }}</p>
      <button
        type="button"
        class="shrink-0 rounded-full border border-border px-3 py-1 text-sm font-bold transition hover:bg-overlay"
        :aria-pressed="paused"
        @click="togglePause"
      >{{ paused ? t('revisit.resume') : t('revisit.pause') }}</button>
    </div>

    <p v-if="paused" class="text-muted">{{ t('revisit.paused') }}</p>
    <p v-else-if="loaded && !items.length" class="text-muted">{{ t('revisit.empty') }}</p>

    <ul v-else class="flex flex-col gap-3">
      <li
        v-for="item in items"
        :key="item.highlight.id"
        class="rounded-xl border border-border p-3"
      >
        <p class="lp-kicker">{{ item.reflection_prompt }}</p>
        <p class="mt-1 text-sm font-semibold leading-snug">{{ label(item) }}</p>
        <div class="mt-2 flex items-center gap-3">
          <RouterLink
            :to="{ name: 'player', params: { slug: item.highlight.episode_slug }, query: jumpQuery(item) }"
            class="font-mono text-xs font-bold text-accent no-underline"
          >▶ {{ item.highlight.start_ms != null ? formatTime(item.highlight.start_ms / 1000) : t('revisit.open') }}</RouterLink>
          <button
            type="button"
            class="text-xs text-muted transition hover:text-canvas-foreground"
            @click="dismiss(item)"
          >{{ t('revisit.dismiss') }}</button>
        </div>
      </li>
    </ul>
  </div>
</template>
