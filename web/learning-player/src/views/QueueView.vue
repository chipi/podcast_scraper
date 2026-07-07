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
         The card's own queue toggle is the remove affordance; reorder ↑/↓ sit in the card's icon
         row (via its #actions slot), consistent small rounded buttons — no layout-shifting side rail. -->
    <div v-else class="flex flex-col">
      <template v-for="(slug, i) in queue.items" :key="slug">
        <EpisodeCard v-if="details[slug]" :episode="summaryFromDetail(details[slug])">
          <template #actions>
            <button
              type="button"
              :disabled="i === 0"
              :aria-label="t('queue.up')"
              :title="t('queue.up')"
              class="relative z-30 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground disabled:opacity-30"
              @click="queue.move(slug, -1)"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><path d="m18 15-6-6-6 6" /></svg>
            </button>
            <button
              type="button"
              :disabled="i === queue.items.length - 1"
              :aria-label="t('queue.down')"
              :title="t('queue.down')"
              class="relative z-30 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground disabled:opacity-30"
              @click="queue.move(slug, 1)"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg>
            </button>
          </template>
        </EpisodeCard>
        <div v-else class="border-b border-border py-5 text-sm text-muted">…</div>
      </template>
    </div>
  </section>
</template>
