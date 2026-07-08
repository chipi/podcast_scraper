<script setup lang="ts">
/**
 * Momentum rail (RFC-103) — one generic "trending now" rail for any entity kind, powered by the
 * read-time momentum endpoint (GET /api/app/trending). Each chip shows the entity's label, its
 * velocity (↑ rising), a weekly sparkline, and — for interest-token kinds (topic / cluster /
 * storyline / person) — a one-tap follow. Emits `open` with the entity so the parent decides how to
 * open it. Hides when nothing is trending.
 */
import { computed, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { getTrending } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import type { TrendingEntity } from '../services/types'
import Sparkline from './Sparkline.vue'
import { trendArrow, trendColor } from './trending'

const props = withDefaults(
  defineProps<{ kind: string; title: string; scope?: 'corpus' | 'mine'; limit?: number }>(),
  { scope: 'corpus', limit: 12 },
)
const emit = defineEmits<{ (e: 'open', entity: TrendingEntity): void }>()

const auth = useAuthStore()
const interests = useInterestsStore()
const { ids: followedIds } = storeToRefs(interests)
if (auth.isAuthenticated) void interests.ensureLoaded().catch(() => {})

// Only interest tokens are followable (topic: / tc: / thc: / person:); episodes/shows/insights aren't.
const _FOLLOWABLE = /^(topic:|tc:|thc:|person:)/
function isFollowable(id: string): boolean {
  return auth.isAuthenticated && _FOLLOWABLE.test(id)
}
function isFollowed(id: string): boolean {
  return followedIds.value.includes(id)
}
function onFollow(id: string): void {
  void interests.toggle(id)
}

const items = ref<TrendingEntity[]>([])
void getTrending(props.kind, props.scope, props.limit)
  .then((rows) => (items.value = rows))
  .catch(() => (items.value = []))
const hasAny = computed(() => items.value.length > 0)
</script>

<template>
  <section v-if="hasAny" class="mt-7" :data-testid="`momentum-rail-${kind}`">
    <h2 class="lp-section mb-2">{{ title }}</h2>
    <div class="flex flex-wrap gap-1.5">
      <div
        v-for="e in items"
        :key="e.entity_id"
        class="inline-flex items-center gap-2 rounded-full bg-overlay px-1 text-sm text-topic transition hover:bg-elevated"
        data-testid="momentum-chip"
      >
        <button
          type="button"
          class="inline-flex items-center gap-1.5 py-1.5 pl-2"
          @click="emit('open', e)"
        >
          <span class="font-semibold">{{ e.label }}</span>
          <Sparkline :values="e.series" :width="44" :height="16" />
          <span class="text-xs font-semibold" :style="{ color: trendColor(e.velocity) }"
            >{{ trendArrow(e.velocity) }} {{ Math.round(e.velocity * 10) / 10 }}×</span
          >
        </button>
        <button
          v-if="isFollowable(e.entity_id)"
          type="button"
          class="rounded-r-full py-1.5 pl-0.5 pr-2.5 text-base leading-none transition"
          :class="isFollowed(e.entity_id) ? 'text-accent' : 'text-muted hover:text-accent'"
          data-testid="momentum-follow"
          :aria-pressed="isFollowed(e.entity_id)"
          @click="onFollow(e.entity_id)"
        >{{ isFollowed(e.entity_id) ? '✓' : '＋' }}</button>
      </div>
    </div>
  </section>
</template>
