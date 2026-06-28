<script setup lang="ts">
/**
 * Play queue (PRD-039 FR2.3) — reorder / remove / play. Auth-gated (meta.requiresAuth).
 * The API stores ordered slugs; this view hydrates titles via episode detail (small queues).
 */
import { onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { getEpisode } from '../services/api'
import type { EpisodeDetail } from '../services/types'
import { useQueueStore } from '../stores/queue'
import { summaryFromDetail } from '../utils/episode'
import EpisodeCard from '../components/EpisodeCard.vue'

// `hideTitle` lets the Library hub embed this as the "Queue" tab without a duplicate heading.
defineProps<{ hideTitle?: boolean }>()
const { t } = useI18n()
const queue = useQueueStore()
const details = ref<Record<string, EpisodeDetail>>({})
const loading = ref(true)

async function hydrate(): Promise<void> {
  await queue.ensureLoaded()
  const missing = queue.items.filter((s) => !details.value[s])
  const fetched = await Promise.all(
    missing.map((s) =>
      getEpisode(s)
        .then((d) => [s, d] as const)
        .catch(() => null),
    ),
  )
  for (const f of fetched) if (f) details.value[f[0]] = f[1]
  loading.value = false
}

onMounted(hydrate)
watch(() => queue.items.slice(), hydrate)
</script>

<template>
  <section>
    <h1 v-if="!hideTitle" class="mb-5 font-display text-3xl font-extrabold tracking-tight">{{ t('queue.title') }}</h1>

    <p v-if="loading && queue.count === 0" class="text-muted">{{ t('catalog.loading') }}</p>
    <p v-else-if="queue.count === 0" class="text-muted">{{ t('queue.empty') }}</p>

    <!-- Showcase each queued episode through the shared card (UXS-014 — one card, every surface).
         The card's own queue toggle is the remove affordance; reorder lives in the ↑/↓ rail. -->
    <div v-else class="flex flex-col">
      <div v-for="(slug, i) in queue.items" :key="slug" class="flex items-stretch gap-1">
        <div class="flex flex-col items-center justify-center gap-1 pt-2 text-muted">
          <button type="button" :disabled="i === 0" :aria-label="t('queue.up')" class="px-1 leading-none disabled:opacity-30 hover:text-canvas-foreground" @click="queue.move(slug, -1)">↑</button>
          <span class="font-mono text-xs tabular-nums">{{ i + 1 }}</span>
          <button type="button" :disabled="i === queue.items.length - 1" :aria-label="t('queue.down')" class="px-1 leading-none disabled:opacity-30 hover:text-canvas-foreground" @click="queue.move(slug, 1)">↓</button>
        </div>
        <EpisodeCard v-if="details[slug]" :episode="summaryFromDetail(details[slug])" class="flex-1" />
        <div v-else class="flex-1 border-b border-border py-5 text-sm text-muted">…</div>
      </div>
    </div>
  </section>
</template>
