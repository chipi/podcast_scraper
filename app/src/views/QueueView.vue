<script setup lang="ts">
/**
 * Play queue (PRD-039 FR2.3) — reorder / remove / play. Auth-gated (meta.requiresAuth).
 * The API stores ordered slugs; this view hydrates titles via episode detail (small queues).
 */
import { onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getEpisode } from '../services/api'
import type { EpisodeDetail } from '../services/types'
import { useQueueStore } from '../stores/queue'

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
    <h1 class="mb-5 font-display text-3xl font-extrabold tracking-tight">{{ t('queue.title') }}</h1>

    <p v-if="loading && queue.count === 0" class="text-muted">{{ t('catalog.loading') }}</p>
    <p v-else-if="queue.count === 0" class="text-muted">{{ t('queue.empty') }}</p>

    <ul v-else class="flex flex-col">
      <li
        v-for="(slug, i) in queue.items"
        :key="slug"
        class="flex items-center gap-3 border-b border-border py-3"
      >
        <span class="font-mono text-xs text-disabled tabular-nums">{{ i + 1 }}</span>
        <div class="min-w-0 flex-1">
          <RouterLink
            :to="{ name: 'player', params: { slug } }"
            class="block truncate font-display font-bold text-canvas-foreground no-underline"
          >
            {{ details[slug]?.title ?? slug }}
          </RouterLink>
          <span v-if="details[slug]?.podcast_title" class="lp-kicker">{{ details[slug]?.podcast_title }}</span>
        </div>
        <div class="flex shrink-0 items-center gap-1 text-muted">
          <button type="button" :disabled="i === 0" :aria-label="t('queue.up')" class="px-2 disabled:opacity-30" @click="queue.move(slug, -1)">↑</button>
          <button type="button" :disabled="i === queue.items.length - 1" :aria-label="t('queue.down')" class="px-2 disabled:opacity-30" @click="queue.move(slug, 1)">↓</button>
          <button type="button" :aria-label="t('queue.remove')" class="px-2 text-danger" @click="queue.remove(slug)">✕</button>
        </div>
      </li>
    </ul>
  </section>
</template>
