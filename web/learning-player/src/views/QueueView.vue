<script setup lang="ts">
/**
 * Play queue (PRD-039 FR2.3) — reorder / remove / play. Auth-gated (meta.requiresAuth).
 * The API stores ordered slugs; this view hydrates titles via episode detail (small queues).
 */
import { onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { getEpisode } from '../services/api'
import { readCached, writeCached } from '../services/contentCache'
import type { EpisodeDetail } from '../services/types'
import { useQueueStore } from '../stores/queue'
import { summaryFromDetail } from '../utils/episode'
import EpisodeCard from '../components/EpisodeCard.vue'
import SectionStatus from '../components/SectionStatus.vue'

// `hideTitle` lets the Library hub embed this as the "Queue" tab without a duplicate heading.
defineProps<{ hideTitle?: boolean }>()
const { t } = useI18n()
const queue = useQueueStore()
const details = ref<Record<string, EpisodeDetail>>({})
const loading = ref(true)

/**
 * The queue's episode titles, cached (operator 2026-09-19).
 *
 * The STORE already keeps the ordered slugs readable offline (#1909), but a slug is not something
 * a person recognises: offline the panel rendered four rows of "…" — the queue was intact and
 * completely unreadable, which is worse than an empty one because it shows there is something
 * there and refuses to say what. Titles are what makes a queue a queue, so they are cached
 * alongside it and read back whenever the network cannot answer.
 *
 * One blob rather than a key per episode: it is written and read as a set, it is bounded by the
 * queue itself, and it stays in step with the `queue` key the store writes beside it.
 */
const DETAILS_KEY = 'queue.details'
const isDetailMap = (v: unknown): boolean =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

async function hydrate(): Promise<void> {
  // ensureLoaded no longer throws, but it can report failure — offline this left `loading` true
  // forever and the Queue tab was a permanent spinner (#1906).
  await queue.ensureLoaded()

  // Paint from cache FIRST, so the titles are on screen whether or not the network answers. A
  // successful fetch below overwrites each entry with the fresh copy.
  const cached = await readCached<Record<string, EpisodeDetail>>(DETAILS_KEY, isDetailMap)
  if (cached) details.value = { ...cached, ...details.value }

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

  // Keep only what is still queued, so a removed episode does not linger in the cache for ever.
  const keep: Record<string, EpisodeDetail> = {}
  for (const slug of queue.items) if (details.value[slug]) keep[slug] = details.value[slug]
  if (Object.keys(keep).length) void writeCached(DETAILS_KEY, keep)
}

// Belt and braces: any unexpected rejection must still clear the spinner.
function hydrateSafely(): void {
  void hydrate().catch(() => {
    loading.value = false
  })
}

onMounted(hydrateSafely)
watch(() => queue.items.slice(), hydrateSafely)
</script>

<template>
  <section>
    <h1 v-if="!hideTitle" class="mb-5 font-display text-3xl font-extrabold tracking-tight">{{ t('queue.title') }}</h1>

    <!-- `stale` = the cached copy, never revalidated. Adding and removing still work (item-level,
         replayed from the outbox); REORDERING does not, because it goes through a whole-list PUT
         that would delete whatever the server actually holds. Say which is which rather than
         leaving the arrows to do nothing silently (#1925). -->
    <p v-if="queue.stale" class="mb-4 rounded-lg border border-border bg-surface px-3 py-2 text-sm text-muted">
      {{ t('queue.offline') }}
    </p>

    <!-- F1.3: reserve the queue's list shape while loading (no jump). Offline resolves to the
         stale notice above + the cached queue, so there is no hard error state here. -->
    <SectionStatus v-if="loading && queue.count === 0" phase="loading" :rows="4" />
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
              :disabled="i === 0 || queue.stale"
              :aria-label="t('queue.up')"
              :title="queue.stale ? t('queue.offlineDisabled') : t('queue.up')"
              class="lp-tap z-30 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground disabled:opacity-30"
              @click="queue.move(slug, -1)"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><path d="m18 15-6-6-6 6" /></svg>
            </button>
            <button
              type="button"
              :disabled="i === queue.items.length - 1 || queue.stale"
              :aria-label="t('queue.down')"
              :title="queue.stale ? t('queue.offlineDisabled') : t('queue.down')"
              class="lp-tap z-30 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground disabled:opacity-30"
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
