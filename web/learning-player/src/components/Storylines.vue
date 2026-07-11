<script setup lang="ts">
/**
 * Storylines (option B) — theme clusters (topics discussed together) as a browsable Home rail,
 * sibling of TrendingTopics. Each chip opens the storyline's anchor topic card (whose "discussed
 * together" set is the whole storyline) and carries a one-tap follow (＋/✓) that adds the theme
 * cluster (thc:…) to your interests — the same store the entity-card + trending follows use, so a
 * storyline re-ranks discovery. Reads /api/app/theme-clusters; hides when the corpus has none.
 */
import { computed, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useI18n } from 'vue-i18n'
import { getStorylines } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import type { Storyline } from '../services/types'

const emit = defineEmits<{ (e: 'open', id: string): void }>()
const { t } = useI18n()

const auth = useAuthStore()
const interests = useInterestsStore()
const { ids: followedIds } = storeToRefs(interests)
const canFollow = computed(() => auth.isAuthenticated)
if (auth.isAuthenticated) void interests.ensureLoaded().catch(() => {})
function isFollowed(id: string): boolean {
  return followedIds.value.includes(id)
}
function onFollow(id: string): void {
  void interests.toggle(id)
}

const storylines = ref<Storyline[]>([])
void getStorylines(12)
  .then((s) => (storylines.value = s))
  .catch(() => (storylines.value = []))
const hasAny = computed(() => storylines.value.length > 0)
</script>

<template>
  <section v-if="hasAny" class="mt-7" data-testid="home-storylines">
    <h2 class="lp-section">{{ t('home.storylines') }}</h2>
    <p class="mb-2 text-sm text-muted">{{ t('home.storylinesHint') }}</p>
    <div class="flex flex-wrap gap-1.5">
      <div
        v-for="s in storylines"
        :key="s.id"
        class="lp-theme-chip inline-flex min-w-0 max-w-[calc(50%-0.375rem)] items-center rounded-full text-sm text-surface-foreground transition sm:max-w-none"
        data-testid="storyline-chip"
      >
        <button
          type="button"
          class="inline-flex min-w-0 items-center gap-1.5 py-1.5 pl-3"
          :class="canFollow ? 'pr-1.5' : 'rounded-full pr-3'"
          :aria-label="t('home.storylineOpen', { label: s.label, count: s.size })"
          @click="emit('open', s.anchor_topic_id)"
        >
          <span class="truncate font-semibold">{{ s.label }}</span>
          <span class="shrink-0 text-xs opacity-80">{{
            t('home.storylineSize', s.size, { named: { count: s.size } })
          }}</span>
        </button>
        <button
          v-if="canFollow"
          type="button"
          class="rounded-r-full py-1.5 pl-1 pr-3 text-base leading-none transition"
          :class="isFollowed(s.id) ? 'opacity-100' : 'opacity-60 hover:opacity-100'"
          data-testid="storyline-follow"
          :aria-pressed="isFollowed(s.id)"
          :aria-label="
            isFollowed(s.id)
              ? t('home.storylineFollowing', { label: s.label })
              : t('home.storylineFollow', { label: s.label })
          "
          @click="onFollow(s.id)"
        >{{ isFollowed(s.id) ? '✓' : '＋' }}</button>
      </div>
    </div>
  </section>
</template>
