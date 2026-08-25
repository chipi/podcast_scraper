<script setup lang="ts">
/**
 * Browse hub (#14) — one destination that gathers the three corpus indexes that were previously
 * scattered as separate in-content links on Home: Episodes (the catalogue), Topics, and People.
 * Reached from the mobile bottom-nav Browse tab (so it's one tap from anywhere, including Search —
 * #6) and the desktop header. Each card routes to the existing index; this view is just the fan-out.
 */
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
defineOptions({ name: 'BrowseView' }) // stable name for <keep-alive :include> (App.vue)

const { t } = useI18n()

const ENTRIES = [
  { to: 'catalog', title: 'browse.episodes', desc: 'browse.episodesDesc', testid: 'browse-hub-episodes' },
  { to: 'browse-topics', title: 'browse.topics', desc: 'browse.topicsDesc', testid: 'browse-hub-topics' },
  { to: 'browse-people', title: 'browse.people', desc: 'browse.peopleDesc', testid: 'browse-hub-people' },
] as const
</script>

<template>
  <section class="mx-auto max-w-3xl px-4 pb-8 pt-4" data-testid="browse-view">
    <h1 class="mb-1 font-display text-3xl font-extrabold tracking-tight">
      {{ t('browse.hubTitle') }}
    </h1>
    <p class="mb-5 text-sm text-muted">{{ t('browse.hubSubtitle') }}</p>

    <ul class="flex flex-col gap-3">
      <li v-for="e in ENTRIES" :key="e.to">
        <RouterLink
          :to="{ name: e.to }"
          :data-testid="e.testid"
          class="flex items-center justify-between gap-3 rounded-2xl border border-border bg-surface px-5 py-4 no-underline transition hover:bg-overlay"
        >
          <span class="min-w-0">
            <span class="block font-display text-lg font-bold text-canvas-foreground">
              {{ t(e.title) }}
            </span>
            <span class="block text-sm text-muted">{{ t(e.desc) }}</span>
          </span>
          <span class="shrink-0 text-xl text-muted" aria-hidden="true">›</span>
        </RouterLink>
      </li>
    </ul>
  </section>
</template>
