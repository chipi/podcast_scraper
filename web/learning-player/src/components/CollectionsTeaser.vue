<script setup lang="ts">
/**
 * A few of the user's boards, beside the ask box on Home (operator 2026-09-18).
 *
 * Collections were reachable only through Library → Boards, which means the thing the user
 * ASSEMBLED was the least visible thing they own — the opposite of how a curation surface earns
 * repeat visits. This is a teaser, not the tab: a handful of covers as quick access, with "See
 * all" for the rest.
 *
 * Cover + name + count, because a board is recognised by its picture first and named second; the
 * count is what tells you whether there is anything in it yet.
 *
 * Most recently CHANGED first (`updated_at`), not alphabetical and not the manual board order: on
 * Home the question is "what am I working on", and the board you added to yesterday answers it.
 * The manual `position` ordering is the Boards tab's own affordance and stays there.
 */
import { computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { useCollectionsStore } from '../stores/collections'

/** How many fit the space beside the ask box without the row becoming its own section. */
const MAX = 4

const { t } = useI18n()
const collections = useCollectionsStore()

onMounted(() => {
  void collections.ensureLoaded()
})

const items = computed(() =>
  [...collections.items]
    .sort((a, b) => (b.updated_at ?? b.created_at ?? 0) - (a.updated_at ?? a.created_at ?? 0))
    .slice(0, MAX),
)
</script>

<template>
  <section v-if="items.length" class="mt-7" data-testid="home-collections-teaser">
    <div class="mb-3 flex items-baseline justify-between gap-3">
      <h2 class="lp-section">{{ t('home.collectionsTitle') }}</h2>
      <RouterLink
        :to="{ name: 'library', query: { tab: 'collections' } }"
        class="shrink-0 text-xs font-semibold text-accent no-underline"
        data-testid="home-collections-see-all"
      >{{ t('home.collectionsSeeAll') }}</RouterLink>
    </div>

    <ul class="grid grid-cols-4 gap-3">
      <li v-for="c in items" :key="c.id" class="min-w-0">
        <RouterLink
          :to="{ name: 'library', query: { tab: 'collections', board: c.id } }"
          class="block no-underline text-canvas-foreground"
          data-testid="home-collection-tile"
        >
          <!-- The cover is derived from the board's first member, so a brand-new board has none.
               A flat tile then, not a broken image or a placeholder icon pretending to be art. -->
          <img
            v-if="c.cover_url"
            :src="c.cover_url"
            alt=""
            aria-hidden="true"
            loading="lazy"
            class="aspect-square w-full rounded-xl bg-elevated object-cover"
          />
          <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
          <div class="mt-1.5 line-clamp-2 text-xs font-bold leading-tight">{{ c.name }}</div>
          <div class="lp-kicker mt-0.5">
            {{ t('collections.count', c.count, { named: { count: c.count } }) }}
          </div>
        </RouterLink>
      </li>
    </ul>
  </section>
</template>
